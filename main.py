import logging
import math
import socket
import sys
import time
from enum import Enum, auto

from camera import Camera
from robot import Robot, Command

# ── EV3 connection ──────────────────────────────────────────────────────────
EV3_HOST = "10.187.118.18"
EV3_PORT = 9999

_sock = None


def _ev3_backend(command: Command) -> None:
    """Send command over TCP; auto-reconnect on failure."""
    global _sock
    payload = (command.value + "\n").encode()
    for _ in range(2):
        try:
            if _sock is None:
                _sock = socket.create_connection(
                    (EV3_HOST, EV3_PORT), timeout=3)
                logging.getLogger(__name__).info(
                    "Connected to EV3 at %s:%d", EV3_HOST, EV3_PORT)
            _sock.sendall(payload)
            _sock.recv(16)
            return
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "EV3 send failed (%s); reconnecting…", exc)
            try:
                _sock.close()
            except Exception:
                pass
            _sock = None
    logging.getLogger(__name__).error(
        "Could not send %r to EV3 after retry.", command.value)


class _ReadlineAwareHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            self.stream.write("\r" + msg + "\n")
            self.stream.flush()
        except Exception:
            self.handleError(record)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[_ReadlineAwareHandler(sys.stderr)],
)

log = logging.getLogger(__name__)

# ── Class names ──────────────────────────────────────────────────────────────
BALL_CLASSES = ("WhiteBall", "OrangeBall", "orange_ball", "white_ball")
ROBOT_CLASS = "Car"
GOAL_CLASSES = ("small_goal", "big_goal")

# ── Navigation constants ─────────────────────────────────────────────────────
_APPROACH_BALL_PX = 90    # px: stop when this close to ball centroid
_APPROACH_GOAL_PX = 55    # px: stop when this close to goal centroid
_ALIGN_DEG = 10.0  # °: drive forward when heading error is within ±10°
_NOISE_PX = 4     # px: minimum displacement to trust as real movement
# Camera runs inference in a background thread.  Without this settle delay,
# robot.assess() returns the frame captured *before* the motor ran — so
# displacement is always ~0 px and heading measurement always fails.
_CMD_SETTLE_S = 0.20

# ── Module-level navigation state ────────────────────────────────────────────
# Kept global so heading earned in APPROACH_BALL is reused in APPROACH_GOAL.
_heading:     tuple[float, float] | None = None   # unit vector (hx, hy)
_right_is_cw: bool | None = None                  # True if "right" rotates CW

FRAME_W = 640
FRAME_H = 480


# ── FSM ───────────────────────────────────────────────────────────────────────
class RobotState(Enum):
    SEARCH_BALL = auto()
    APPROACH_BALL = auto()
    PICKUP = auto()
    SEARCH_GOAL = auto()
    APPROACH_GOAL = auto()
    DROP = auto()
    DONE = auto()


def _transition(cur: RobotState, nxt: RobotState, reason: str = "") -> RobotState:
    log.info("[FSM] %s → %s  %s", cur.name, nxt.name, reason)
    return nxt


# ── Helpers ───────────────────────────────────────────────────────────────────

def _all_balls(state: dict) -> list:
    return [d for cls in BALL_CLASSES
            for d in state["objects_by_class"].get(cls, [])]


def _pick_goal(state: dict) -> str | None:
    for goal in GOAL_CLASSES:
        if goal in state["objects_by_class"]:
            return goal
    return None


def _send(robot: Robot, cmd: str) -> None:
    """Log, send, then pause so the camera thread captures a fresh frame."""
    log.info("    → CMD: %s", cmd.upper())
    robot.send(cmd)
    time.sleep(_CMD_SETTLE_S)


def _read_pos(robot: Robot) -> tuple[float | None, float | None, dict]:
    """Return (rx, ry, full_state).  rx/ry are None if robot not detected."""
    state = robot.assess()
    r = state["objects_by_class"].get(ROBOT_CLASS, [])
    if not r:
        return None, None, state
    return float(r[0]["cx"]), float(r[0]["cy"]), state


def _store_heading(rx0: float, ry0: float,
                   rx1: float, ry1: float) -> bool:
    """
    Compute heading from displacement (rx0,ry0)→(rx1,ry1) and store it
    in the global _heading.  Returns False if displacement is below the
    noise floor (measurement would be unreliable).
    """
    global _heading
    d = math.hypot(rx1 - rx0, ry1 - ry0)
    if d < _NOISE_PX:
        log.warning("  Disp %.1f px < %d px noise floor — heading NOT updated",
                    d, _NOISE_PX)
        return False
    _heading = ((rx1 - rx0) / d, (ry1 - ry0) / d)
    log.info("  Disp %.1f px  (%.0f,%.0f)→(%.0f,%.0f)  New heading: %.0f°",
             d, rx0, ry0, rx1, ry1,
             math.degrees(math.atan2(_heading[1], _heading[0])))
    return True


def _bearing_error(tx: float, ty: float, rx: float, ry: float) -> float:
    """
    Signed angle from current heading to direct bearing toward (tx, ty).
      Positive → target is clockwise  (right) of heading → turn CW.
      Negative → target is counter-clockwise (left) → turn CCW.
    Range: [-180, 180].  Uses direct bearing; no potential field.
    """
    bearing = math.degrees(math.atan2(ty - ry, tx - rx))
    hdg = math.degrees(math.atan2(_heading[1], _heading[0]))
    raw = bearing - hdg
    return ((raw + 180) % 360) - 180


def _overlay(robot: Robot, rx: float, ry: float,
             tx: float, ty: float) -> None:
    """
    Update the camera preview path overlay:
      robot → heading-arrow-tip (80 px) → target
    This draws the heading direction and the target in the same pass.
    """
    det = robot._evaluator._detector
    if not hasattr(det, "set_current_path"):
        return
    if _heading is not None:
        tip = [int(rx + _heading[0] * 80), int(ry + _heading[1] * 80)]
        det.set_current_path([[int(rx), int(ry)], tip, [int(tx), int(ty)]])
    else:
        det.set_current_path([[int(rx), int(ry)], [int(tx), int(ty)]])


# ── Core navigation ───────────────────────────────────────────────────────────

def _navigate_to(robot: Robot, tx: float, ty: float,
                 approach_px: float, track_ball: bool = False) -> bool:
    """
    Simplest possible controller — no A*, no potential field:

      Each iteration:
        1. Read robot position.
        2. If track_ball: snap target to nearest currently-visible ball.
        3. Log: position, heading, target, dist, bearing, error.
        4. If dist < approach_px → ARRIVED.
        5. If heading unknown → FORWARD to bootstrap heading.
        6. If |error| <= _ALIGN_DEG → FORWARD (aligned).
        7. Else:
             a. If _right_is_cw unknown → calibrate (RIGHT + probe FORWARD).
             b. Otherwise → turn in the correct direction.
             After turning, do ONE probe FORWARD to re-measure heading.

    Heading is stored in the module-level _heading so it persists across calls.
    """
    global _heading, _right_is_cw
    cal_tries = 0

    for step in range(300):

        # ── 1. Read position ─────────────────────────────────────────────────
        rx, ry, state = _read_pos(robot)
        if rx is None:
            log.warning("[nav %03d] Robot not detected — waiting", step)
            time.sleep(0.1)
            continue

        # ── 2. Refresh target from nearest visible ball ──────────────────────
        if track_ball:
            fresh = _all_balls(state)
            if fresh:
                nb = min(fresh,
                         key=lambda b: math.hypot(float(b["cx"]) - rx,
                                                  float(b["cy"]) - ry))
                ntx, nty = float(nb["cx"]), float(nb["cy"])
                if abs(ntx - tx) > 3 or abs(nty - ty) > 3:
                    log.info("  Retarget (%.0f,%.0f) → (%.0f,%.0f)",
                             tx, ty, ntx, nty)
                tx, ty = ntx, nty

        # ── 3. Compute metrics ───────────────────────────────────────────────
        dist = math.hypot(tx - rx, ty - ry)
        bearing_deg = math.degrees(math.atan2(ty - ry, tx - rx))
        hdg_deg = (math.degrees(math.atan2(_heading[1], _heading[0]))
                   if _heading else None)
        error = _bearing_error(tx, ty, rx, ry) if _heading else None

        log.info(
            "[nav %03d] "
            "Pos:(%.0f,%.0f)  Hdg:%s  "
            "Target:(%.0f,%.0f)  Dist:%.0fpx  "
            "Bearing:%.0f°  Error:%s  CW:%s",
            step,
            rx, ry,
            f"{hdg_deg:.0f}°" if hdg_deg is not None else "?",
            tx, ty, dist,
            bearing_deg,
            f"{error:.1f}°" if error is not None else "?",
            _right_is_cw,
        )

        _overlay(robot, rx, ry, tx, ty)

        # ── 4. Arrival check ─────────────────────────────────────────────────
        if dist < approach_px:
            log.info("  ✓ ARRIVED  dist=%.0f < %.0f px", dist, approach_px)
            return True

        # ── 5. Bootstrap: drive forward once to get initial heading ──────────
        if _heading is None:
            log.info("  Action: BOOTSTRAP_FORWARD (heading unknown)")
            _send(robot, "forward")
            rx2, ry2, _ = _read_pos(robot)
            if rx2 is not None:
                _store_heading(rx, ry, rx2, ry2)
            continue

        # ── 6. Aligned: drive forward ────────────────────────────────────────
        if abs(error) <= _ALIGN_DEG:
            log.info("  Action: FORWARD  (error=%.1f° within ±%.0f°)",
                     error, _ALIGN_DEG)
            _send(robot, "forward")
            rx2, ry2, _ = _read_pos(robot)
            if rx2 is not None:
                _store_heading(rx, ry, rx2, ry2)
            continue

        # ── 7. Misaligned: turn ──────────────────────────────────────────────
        need_cw = error > 0   # positive error = target is right = need CW turn

        # 7a. Calibrate polarity (once per run)
        if _right_is_cw is None:
            cal_tries += 1
            if cal_tries > 6:
                log.error("  Calibration failed %d× — forcing right_is_cw=True",
                          cal_tries)
                _right_is_cw = True
                continue

            log.info("  Action: CALIBRATE [try %d]  RIGHT + probe FORWARD",
                     cal_tries)
            h_before = _heading

            # Send right turn
            _send(robot, "right")
            rxc, ryc, _ = _read_pos(robot)
            if rxc is None:
                continue

            # Probe forward to measure heading direction after the turn
            _send(robot, "forward")
            rxc2, ryc2, _ = _read_pos(robot)
            if rxc2 is None:
                continue

            if _store_heading(rxc, ryc, rxc2, ryc2):
                # cross product h_before × h_after:
                #   > 0 → heading rotated CW → "right" is CW
                #   < 0 → heading rotated CCW → "right" is CCW
                cross = h_before[0] * _heading[1] - h_before[1] * _heading[0]
                _right_is_cw = cross > 0
                log.info(
                    "  Calibrated: right_is_cw=%s  cross=%.3f  "
                    "before=%.0f°  after=%.0f°",
                    _right_is_cw, cross,
                    math.degrees(math.atan2(h_before[1], h_before[0])),
                    math.degrees(math.atan2(_heading[1], _heading[0])),
                )
            continue

        # 7b. Turn toward the target
        #   need_cw=T, right_is_cw=T → "right"   need_cw=T, right_is_cw=F → "left"
        #   need_cw=F, right_is_cw=T → "left"    need_cw=F, right_is_cw=F → "right"
        turn = "right" if (need_cw == _right_is_cw) else "left"
        log.info("  Action: TURN_%s  (error=%.1f°, need_cw=%s, right_is_cw=%s)",
                 turn.upper(), error, need_cw, _right_is_cw)
        _send(robot, turn)

        # Probe FORWARD to re-measure heading after the turn.
        # This is intentional: we need the robot to move in order to observe
        # the new heading direction from its displacement.
        rxp, ryp, _ = _read_pos(robot)
        if rxp is not None:
            log.info("  Probe FORWARD to re-measure heading after turn")
            _send(robot, "forward")
            rxp2, ryp2, _ = _read_pos(robot)
            if rxp2 is not None:
                _store_heading(rxp, ryp, rxp2, ryp2)

    log.warning("_navigate_to: 300 steps exhausted without arriving")
    return False


# ── Autonomous mission ────────────────────────────────────────────────────────

def run_autonomous(robot: Robot) -> None:
    global _heading, _right_is_cw
    _heading = None
    _right_is_cw = None

    log.info("Waiting 3 s for camera warmup…")
    time.sleep(3.0)
    _send(robot, "open_gripper")
    time.sleep(0.3)

    fsm: RobotState = RobotState.SEARCH_BALL
    ball_target: tuple | None = None
    goal_target: tuple | None = None

    log.info("=== Autonomous mission start  state=%s ===", fsm.name)

    while fsm != RobotState.DONE:
        log.info("--- [FSM: %s] ---", fsm.name)

        # ── SEARCH_BALL ──────────────────────────────────────────────────────
        if fsm == RobotState.SEARCH_BALL:
            frame = robot.assess()
            balls = _all_balls(frame)

            if not balls:
                log.info("No balls visible — mission complete!")
                fsm = _transition(fsm, RobotState.DONE, "(no balls)")
                continue

            robot_dets = frame["objects_by_class"].get(ROBOT_CLASS, [])
            if not robot_dets:
                log.warning("Robot not detected — retrying")
                time.sleep(0.2)
                continue

            rx = float(robot_dets[0]["cx"])
            ry = float(robot_dets[0]["cy"])
            nearest = min(
                balls,
                key=lambda b: math.hypot(
                    float(b["cx"]) - rx, float(b["cy"]) - ry),
            )
            ball_target = (float(nearest["cx"]), float(nearest["cy"]))
            dist = math.hypot(ball_target[0] - rx, ball_target[1] - ry)
            log.info("Target ball: (%.0f,%.0f)  dist=%.0f px  [%d balls]",
                     *ball_target, dist, len(balls))
            fsm = _transition(fsm, RobotState.APPROACH_BALL,
                              f"target={ball_target}")

        # ── APPROACH_BALL ────────────────────────────────────────────────────
        elif fsm == RobotState.APPROACH_BALL:
            assert ball_target is not None
            log.info("Navigating to ball (%.0f,%.0f)  approach=%.0f px",
                     *ball_target, _APPROACH_BALL_PX)
            arrived = _navigate_to(robot, *ball_target,
                                   approach_px=_APPROACH_BALL_PX,
                                   track_ball=True)
            if arrived:
                fsm = _transition(fsm, RobotState.PICKUP, "(arrived)")
            else:
                log.warning("Could not reach ball — re-searching")
                fsm = _transition(fsm, RobotState.SEARCH_BALL, "(nav failed)")

        # ── PICKUP ───────────────────────────────────────────────────────────
        elif fsm == RobotState.PICKUP:
            log.info("Gripping ball…")
            _send(robot, "open_gripper")
            time.sleep(0.3)
            _send(robot, "forward")       # seat ball into gripper arms
            _send(robot, "close_gripper")
            log.info("Ball gripped.")
            fsm = _transition(fsm, RobotState.SEARCH_GOAL)

        # ── SEARCH_GOAL ──────────────────────────────────────────────────────
        elif fsm == RobotState.SEARCH_GOAL:
            log.info("Searching for goal (up to 360° spin)…")
            goal_target = None
            for spin in range(24):
                frame = robot.assess()
                goal_cls = _pick_goal(frame)
                if goal_cls:
                    dets = frame["objects_by_class"].get(goal_cls, [])
                    if dets:
                        goal_target = (
                            float(dets[0]["cx"]), float(dets[0]["cy"]))
                        log.info("Goal '%s' at (%.0f,%.0f) after %d spins",
                                 goal_cls, *goal_target, spin)
                        break
                log.info("  Spin %d: no goal visible", spin)
                _send(robot, "right")

            if goal_target is None:
                log.warning("Goal not found — dropping ball and resetting")
                _send(robot, "open_gripper")
                fsm = _transition(fsm, RobotState.SEARCH_BALL,
                                  "(goal not found, ball dropped)")
            else:
                fsm = _transition(fsm, RobotState.APPROACH_GOAL,
                                  f"goal={goal_target}")

        # ── APPROACH_GOAL ────────────────────────────────────────────────────
        elif fsm == RobotState.APPROACH_GOAL:
            assert goal_target is not None
            log.info("Navigating to goal (%.0f,%.0f)  approach=%.0f px",
                     *goal_target, _APPROACH_GOAL_PX)
            arrived = _navigate_to(robot, *goal_target,
                                   approach_px=_APPROACH_GOAL_PX)
            if arrived:
                fsm = _transition(fsm, RobotState.DROP, "(arrived)")
            else:
                log.warning("Could not reach goal — re-searching")
                fsm = _transition(fsm, RobotState.SEARCH_GOAL, "(nav failed)")

        # ── DROP ─────────────────────────────────────────────────────────────
        elif fsm == RobotState.DROP:
            log.info("Depositing ball at goal.")
            _send(robot, "open_gripper")
            _send(robot, "backward")
            _send(robot, "backward")
            log.info("Ball deposited.")
            fsm = _transition(fsm, RobotState.SEARCH_BALL, "(ready for next)")

    log.info("=== Mission complete ===")


if __name__ == "__main__":
    camera = Camera()
    robot = Robot(detector=camera._detector, backend=_ev3_backend)

    camera.start_detection(model_path="best.pt", confidence=0.40)
    camera.preview()

    try:
        run_autonomous(robot)
    finally:
        camera.close()
