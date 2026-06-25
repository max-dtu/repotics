import logging
import math
import socket
import sys
import time

from camera import Camera
from robot import Robot, Command

# ── EV3 connection ──────────────────────────────────────────────────────────
EV3_HOST = "10.45.151.18"   # EV3 IP shown in its startup banner
EV3_PORT = 9999

_sock = None


def _ev3_backend(command: Command) -> None:
    """Send command over TCP; auto-reconnect on failure."""
    global _sock
    payload = (command.value + "\n").encode()
    for attempt in range(2):
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
# ────────────────────────────────────────────────────────────────────────────


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

# ── Class names — adjust to match your best.pt labels ───────────────────────
BALL_CLASSES = ("WhiteBall", "OrangeBall", "orange_ball", "white_ball")
ROBOT_CLASS = "Car"
# preferred order (more points first)
GOAL_CLASSES = ("small_goal", "big_goal")
# ────────────────────────────────────────────────────────────────────────────


def _pick_goal(state: dict) -> str | None:
    """Return the best visible goal class name, or None."""
    for goal in GOAL_CLASSES:
        if goal in state["objects_by_class"]:
            return goal
    return None


# ── Navigation constants ─────────────────────────────────────────────────────
_APPROACH_PX = 55    # default stop distance (px) from target centroid
_ALIGN_DEG   = 20.0  # acceptable heading error before driving forward
_NOISE_PX    = 8     # min displacement (px) required to trust as heading
# ─────────────────────────────────────────────────────────────────────────────

# ── Field / obstacle avoidance ───────────────────────────────────────────────
FRAME_W        = 640   # camera frame width  (px)
FRAME_H        = 480   # camera frame height (px)
_BORDER_MARGIN = 80    # px from edge to activate border repulsion
_OBS_MARGIN    = 110   # px from YOLO obstacle centroid to activate repulsion
_BORDER_W      = 1.5   # border repulsion weight vs attraction
_OBS_W         = 2.5   # obstacle repulsion weight vs attraction

# Hardcoded obstacles (cx, cy, radius_px).
# Add the cross here if YOLO does not detect it, e.g.:
#   _FIXED_OBSTACLES = [(320, 240, 70)]  # cross at frame centre
_FIXED_OBSTACLES: list[tuple[float, float, float]] = []
# ─────────────────────────────────────────────────────────────────────────────

# True  = "right" command is clockwise in the camera image
# False = "right" is counter-clockwise
# None  = not yet determined (calibrated on the first turn needed)
_right_is_cw: bool | None = None


def _steer_direction(
    rx: float, ry: float,
    tx: float, ty: float,
    state: dict,
) -> tuple[float, float]:
    """
    Return a normalised (dx, dy) steering direction blending:
      - attraction toward (tx, ty)
      - repulsion from the four field borders
      - repulsion from YOLO-detected obstacles (non-ball / non-goal / non-robot)
      - repulsion from hardcoded _FIXED_OBSTACLES (e.g. the cross)
    """
    dist = math.hypot(tx - rx, ty - ry)
    if dist < 1e-3:
        return (0.0, 0.0)

    ax = (tx - rx) / dist
    ay = (ty - ry) / dist

    # Border repulsion
    bx, by = 0.0, 0.0
    if rx < _BORDER_MARGIN:
        bx += (_BORDER_MARGIN - rx) / _BORDER_MARGIN
    if rx > FRAME_W - _BORDER_MARGIN:
        bx -= (rx - (FRAME_W - _BORDER_MARGIN)) / _BORDER_MARGIN
    if ry < _BORDER_MARGIN:
        by += (_BORDER_MARGIN - ry) / _BORDER_MARGIN
    if ry > FRAME_H - _BORDER_MARGIN:
        by -= (ry - (FRAME_H - _BORDER_MARGIN)) / _BORDER_MARGIN

    # YOLO obstacle repulsion (any class not recognised as ball / goal / robot)
    known = frozenset(BALL_CLASSES) | frozenset(GOAL_CLASSES) | {ROBOT_CLASS}
    ox, oy = 0.0, 0.0
    for det in state.get("detections", []):
        if det.get("class_name", "") in known:
            continue
        dcx, dcy = float(det["cx"]), float(det["cy"])
        odist = math.hypot(rx - dcx, ry - dcy)
        if 0 < odist < _OBS_MARGIN:
            mag = (_OBS_MARGIN - odist) / _OBS_MARGIN
            ox += (rx - dcx) / odist * mag
            oy += (ry - dcy) / odist * mag

    # Hardcoded obstacle repulsion (cross, walls, etc.)
    for (fcx, fcy, frad) in _FIXED_OBSTACLES:
        odist = math.hypot(rx - fcx, ry - fcy)
        if 0 < odist < frad:
            mag = (frad - odist) / frad
            ox += (rx - fcx) / odist * mag
            oy += (ry - fcy) / odist * mag

    cx = ax + _BORDER_W * bx + _OBS_W * ox
    cy = ay + _BORDER_W * by + _OBS_W * oy

    cmag = math.hypot(cx, cy)
    if cmag < 1e-6:
        return (ax, ay)
    return (cx / cmag, cy / cmag)


def _drive_to(robot: Robot, tx: float, ty: float,
              approach_px: float = _APPROACH_PX) -> bool:
    """
    Drive to pixel target (tx, ty) using heading-aware steering.

    Sequence each iteration:
      1. Read position, update path preview, check arrival (dist < approach_px).
      2. Bootstrap (heading=None): drive forward once to measure initial heading.
      3. Compute signed angular error to the potential-field steering direction.
         cross = hx*dy - hy*dx; positive → target is clockwise from heading.
      4. Misaligned: turn first, then probe forward to measure real heading.
         - First turn ever: calibration probe (right + forward) sets _right_is_cw.
         - Subsequent turns: send correct turn, then probe forward for real heading.
           No dead reckoning — error cannot accumulate.
      5. Aligned: drive forward and update heading from measured displacement.
    """
    global _right_is_cw
    detector = robot._evaluator._detector

    heading: tuple[float, float] | None = None

    for _ in range(100):
        # ── 1. Read position ────────────────────────────────────────────────
        state = robot.assess()
        r     = state["objects_by_class"].get(ROBOT_CLASS, [])
        if not r:
            time.sleep(0.05)
            continue

        rx, ry = float(r[0]["cx"]), float(r[0]["cy"])
        dist   = math.hypot(tx - rx, ty - ry)

        if hasattr(detector, "set_current_path"):
            detector.set_current_path([[int(rx), int(ry)], [int(tx), int(ty)]])

        log.info("drive_to: pos=(%.0f,%.0f)  dist=%.0fpx  hdg=%s",
                 rx, ry, dist,
                 "(%.2f,%.2f)" % heading if heading else "?")

        if dist < approach_px:
            return True

        # ── 2. Bootstrap: drive forward once to get initial heading ─────────
        if heading is None:
            robot.send("forward")
            s2 = robot.assess()
            r2 = s2["objects_by_class"].get(ROBOT_CLASS, [])
            if r2:
                rx2, ry2 = float(r2[0]["cx"]), float(r2[0]["cy"])
                d = math.hypot(rx2 - rx, ry2 - ry)
                if d >= _NOISE_PX:
                    heading = ((rx2 - rx) / d, (ry2 - ry) / d)
                    log.info("  Bootstrap heading: (%.2f, %.2f)", *heading)
            continue

        # ── 3. Angular error using potential-field steering direction ───────
        dx, dy    = _steer_direction(rx, ry, tx, ty, state)
        cross     = heading[0] * dy - heading[1] * dx
        dot       = heading[0] * dx + heading[1] * dy
        error_deg = math.degrees(math.atan2(cross, dot))
        log.info("  error=%.1f°", error_deg)

        # ── 4. Misaligned: turn first ────────────────────────────────────────
        if abs(error_deg) > _ALIGN_DEG:
            if _right_is_cw is None:
                # One-time calibration: probe with "right" then forward
                h_before = heading
                robot.send("right")
                sc1 = robot.assess()
                rc1 = sc1["objects_by_class"].get(ROBOT_CLASS, [])
                if not rc1:
                    continue
                rxc, ryc = float(rc1[0]["cx"]), float(rc1[0]["cy"])
                robot.send("forward")
                sc2 = robot.assess()
                rc2 = sc2["objects_by_class"].get(ROBOT_CLASS, [])
                if not rc2:
                    continue
                rxc2, ryc2 = float(rc2[0]["cx"]), float(rc2[0]["cy"])
                d = math.hypot(rxc2 - rxc, ryc2 - ryc)
                if d >= _NOISE_PX:
                    h_after      = ((rxc2 - rxc) / d, (ryc2 - ryc) / d)
                    cal_cross    = h_before[0] * h_after[1] - h_before[1] * h_after[0]
                    _right_is_cw = cal_cross > 0
                    heading      = h_after
                    log.info("  Calibrated: right_is_cw=%s (cross=%.3f)",
                             _right_is_cw, cal_cross)
                continue

            # Steer with calibrated polarity
            need_cw  = error_deg > 0
            turn_cmd = ("right" if need_cw else "left") if _right_is_cw \
                  else ("left"  if need_cw else "right")
            robot.send(turn_cmd)

            # Probe forward to get ground-truth heading (no dead reckoning)
            sp1 = robot.assess()
            rp1 = sp1["objects_by_class"].get(ROBOT_CLASS, [])
            if rp1:
                rxp, ryp = float(rp1[0]["cx"]), float(rp1[0]["cy"])
                robot.send("forward")
                sp2 = robot.assess()
                rp2 = sp2["objects_by_class"].get(ROBOT_CLASS, [])
                if rp2:
                    rxp2, ryp2 = float(rp2[0]["cx"]), float(rp2[0]["cy"])
                    dp = math.hypot(rxp2 - rxp, ryp2 - ryp)
                    if dp >= _NOISE_PX:
                        heading = ((rxp2 - rxp) / dp, (ryp2 - ryp) / dp)
                        log.info("  Heading after turn (real): (%.2f, %.2f)", *heading)
            continue

        # ── 5. Aligned: drive forward and update heading ─────────────────────
        robot.send("forward")
        s3 = robot.assess()
        r3 = s3["objects_by_class"].get(ROBOT_CLASS, [])
        if r3:
            rx3, ry3 = float(r3[0]["cx"]), float(r3[0]["cy"])
            d = math.hypot(rx3 - rx, ry3 - ry)
            if d >= _NOISE_PX:
                heading = ((rx3 - rx) / d, (ry3 - ry) / d)
                log.info("  Heading from disp: (%.2f, %.2f)  d=%.1fpx", *heading, d)

    log.warning("_drive_to: max steps reached.")
    return False


def _all_balls(state: dict) -> list:
    return [d for cls in BALL_CLASSES for d in state["objects_by_class"].get(cls, [])]


def _plan_sequence(balls: list, start: tuple) -> list:
    """
    Greedy nearest-neighbour ordering of all balls from start position.
    Returns a new list ordered closest-first.
    """
    remaining = list(balls)
    ordered = []
    pos = start
    while remaining:
        nearest = min(remaining, key=lambda d: math.hypot(
            d["cx"] - pos[0], d["cy"] - pos[1]))
        ordered.append(nearest)
        pos = (nearest["cx"], nearest["cy"])
        remaining.remove(nearest)
    return ordered


def _show_route(robot: Robot, start: tuple, balls: list) -> None:
    """Draw the full planned route through all remaining balls in the preview."""
    pts = [[int(start[0]), int(start[1])]]
    for b in balls:
        pts.append([int(b["cx"]), int(b["cy"])])
    detector = robot._evaluator._detector
    if hasattr(detector, "set_current_path"):
        detector.set_current_path(pts)


def run_autonomous(robot: Robot) -> None:
    global _right_is_cw
    _right_is_cw = None   # re-calibrate turn direction at the start of each run

    log.info("Waiting 3 s for camera to warm up...")
    time.sleep(3.0)

    robot.send("open_gripper")
    time.sleep(0.5)

    log.info("Autonomous collection started.")

    while True:
        state = robot.assess()
        balls = _all_balls(state)

        if not balls:
            log.info("No balls detected — mission complete!")
            break

        robot_dets = state["objects_by_class"].get(ROBOT_CLASS, [])
        if not robot_dets:
            time.sleep(0.2)
            continue

        robot_pos = (float(robot_dets[0]["cx"]), float(robot_dets[0]["cy"]))

        # ── Plan one route through all balls, closest first ─────────────────
        sequence = _plan_sequence(balls, robot_pos)
        log.info("Route planned: %d balls", len(sequence))
        _show_route(robot, robot_pos, sequence)

        # ── Follow the route ─────────────────────────────────────────────────
        for i, ball in enumerate(sequence):
            # Update preview to show only remaining balls in route
            state = robot.assess()
            robot_dets = state["objects_by_class"].get(ROBOT_CLASS, [])
            if robot_dets:
                robot_pos = (float(robot_dets[0]["cx"]), float(
                    robot_dets[0]["cy"]))
                _show_route(robot, robot_pos, sequence[i:])

            target = (float(ball["cx"]), float(ball["cy"]))
            log.info("Ball %d/%d at (%.0f, %.0f)",
                     i + 1, len(sequence), *target)

            # ── 1. Navigate to just outside gripper range ────────────────────
            if not _drive_to(robot, *target, approach_px=90):
                log.warning("Could not reach ball %d — skipping.", i + 1)
                continue

            # ── 2. Open gripper, advance to seat ball, then close ────────────
            log.info("Gripping ball...")
            robot.send("open_gripper")
            time.sleep(0.3)           # let gripper fully open
            robot.send("forward")     # advance ball into open gripper arms
            robot.send("close_gripper")

            # ── 3. Find goal — spin until one is visible ──────────────────────
            log.info("Looking for goal...")
            goal_pos = None
            for _ in range(24):          # up to ~360° of search
                state = robot.assess()
                goal_cls = _pick_goal(state)
                if goal_cls:
                    dets = state["objects_by_class"].get(goal_cls, [])
                    if dets:
                        goal_pos = (float(dets[0]["cx"]), float(dets[0]["cy"]))
                        log.info("Goal found: %s at (%.0f, %.0f)",
                                 goal_cls, *goal_pos)
                        break
                robot.send("right")

            if goal_pos is None:
                log.warning("Goal not found — dropping ball and continuing.")
                robot.send("open_gripper")
                continue

            # ── 4. Drive to goal ──────────────────────────────────────────────
            _drive_to(robot, *goal_pos)

            # ── 5. Deposit and back away ──────────────────────────────────────
            log.info("Depositing ball.")
            robot.send("open_gripper")
            robot.send("backward")
            robot.send("backward")

        # Loop back: replan from new position with any remaining/new balls


if __name__ == "__main__":
    camera = Camera()
    robot = Robot(detector=camera._detector, backend=_ev3_backend)

    camera.start_detection(model_path="best.pt", confidence=0.15)
    camera.preview()

    try:
        run_autonomous(robot)
    finally:
        camera.close()
