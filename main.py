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
                _sock = socket.create_connection((EV3_HOST, EV3_PORT), timeout=3)
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
BALL_CLASSES  = ("WhiteBall", "OrangeBall", "orange_ball", "white_ball")
ROBOT_CLASS   = "Car"
GOAL_CLASSES  = ("small_goal", "big_goal")   # preferred order (more points first)
# ────────────────────────────────────────────────────────────────────────────


def _pick_goal(state: dict) -> str | None:
    """Return the best visible goal class name, or None."""
    for goal in GOAL_CLASSES:
        if goal in state["objects_by_class"]:
            return goal
    return None


# ── Navigation constants ─────────────────────────────────────────────────────
_APPROACH_PX = 70   # stop when ball centroid is within this many pixels
_PROGRESS_PX = 4    # minimum pixels toward ball per step to count as progress
# ─────────────────────────────────────────────────────────────────────────────


def _drive_to(robot: Robot, tx: float, ty: float) -> bool:
    """
    Drive to pixel target (tx, ty) without any heading math.

    Each iteration:
      1. Send 'forward'.
      2. Measure the dot-product of the displacement with the direction to
         the target.  If the robot moved at least _PROGRESS_PX toward it,
         keep going.  Otherwise turn right and try again next step.

    No heading estimation, no dead reckoning, no sign assumptions about
    which physical direction the turn commands rotate.
    """
    detector = robot._evaluator._detector

    for _ in range(100):
        state = robot.assess()
        r     = state["objects_by_class"].get(ROBOT_CLASS, [])
        if not r:
            time.sleep(0.05)
            continue

        rx, ry = float(r[0]["cx"]), float(r[0]["cy"])
        dist   = math.hypot(tx - rx, ty - ry)

        if hasattr(detector, "set_current_path"):
            detector.set_current_path([[int(rx), int(ry)], [int(tx), int(ty)]])

        log.info("drive_to: pos=(%.0f,%.0f)  dist=%.0fpx", rx, ry, dist)

        if dist < _APPROACH_PX:
            return True

        # Drive forward, then measure progress toward target
        robot.send("forward")

        state2 = robot.assess()
        r2     = state2["objects_by_class"].get(ROBOT_CLASS, [])
        if not r2:
            continue
        rx2, ry2 = float(r2[0]["cx"]), float(r2[0]["cy"])

        # Dot product of displacement with direction-to-target
        # positive and large  → moved toward target
        # small or negative   → moving sideways or away
        progress = ((rx2 - rx) * (tx - rx) + (ry2 - ry) * (ty - ry)) / max(dist, 1)
        log.info("  progress=%.1fpx", progress)

        if progress < _PROGRESS_PX:
            robot.send("right")   # not heading toward ball — turn and retry

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
        nearest = min(remaining, key=lambda d: math.hypot(d["cx"] - pos[0], d["cy"] - pos[1]))
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
                robot_pos = (float(robot_dets[0]["cx"]), float(robot_dets[0]["cy"]))
                _show_route(robot, robot_pos, sequence[i:])

            target = (float(ball["cx"]), float(ball["cy"]))
            log.info("Ball %d/%d at (%.0f, %.0f)", i + 1, len(sequence), *target)

            # ── 1. Drive to ball ─────────────────────────────────────────────
            if not _drive_to(robot, *target):
                log.warning("Could not reach ball %d — skipping.", i + 1)
                continue

            # ── 2. Grab ──────────────────────────────────────────────────────
            log.info("Closing gripper...")
            robot.send("close_gripper")
            time.sleep(0.5)

            # ── 3. Drive to goal ─────────────────────────────────────────────
            state = robot.assess()
            goal = _pick_goal(state)
            if goal is None:
                log.warning("No goal visible — opening gripper.")
                robot.send("open_gripper")
                time.sleep(0.5)
                continue

            log.info("Depositing at %s...", goal)
            state      = robot.assess()
            goal_dets  = state["objects_by_class"].get(goal, [])
            if goal_dets:
                _drive_to(robot, float(goal_dets[0]["cx"]), float(goal_dets[0]["cy"]))

            # ── 4. Deposit ───────────────────────────────────────────────────
            robot.send("open_gripper")
            time.sleep(0.5)

        # Loop back: replan from new position with any remaining/new balls


if __name__ == "__main__":
    camera = Camera()
    robot  = Robot(detector=camera._detector, backend=_ev3_backend)

    camera.start_detection(model_path="best.pt", confidence=0.30)
    camera.preview()

    try:
        run_autonomous(robot)
    finally:
        camera.close()
