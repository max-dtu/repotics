import logging
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


def _any_balls(state: dict) -> bool:
    return any(cls in state["objects_by_class"] for cls in BALL_CLASSES)


def run_autonomous(robot: Robot) -> None:
    log.info("Waiting 3 s for camera to warm up...")
    time.sleep(3.0)

    # Reset gripper to open so we're ready to grab
    robot.send("open_gripper")
    time.sleep(0.5)

    log.info("Autonomous collection started.")

    while True:
        state = robot.assess()

        if not _any_balls(state):
            log.info("No balls detected — mission complete!")
            break

        # ── 1. Drive to nearest ball ────────────────────────────────────────
        log.info("Navigating to nearest ball...")
        path = robot.go_to_nearest_ball(
            ball_classes=BALL_CLASSES,
            robot_class=ROBOT_CLASS,
        )
        if not path:
            log.warning("Could not plan path to ball — retrying in 1 s...")
            time.sleep(1.0)
            continue

        # ── 2. Grab it ──────────────────────────────────────────────────────
        log.info("Closing gripper...")
        robot.send("close_gripper")
        time.sleep(0.5)

        # ── 3. Drive to goal ────────────────────────────────────────────────
        state = robot.assess()
        goal = _pick_goal(state)
        if goal is None:
            log.warning("No goal in view — opening gripper and retrying...")
            robot.send("open_gripper")
            time.sleep(1.0)
            continue

        log.info("Driving to %s...", goal)
        robot.find_path(ROBOT_CLASS, goal)

        # ── 4. Deposit ──────────────────────────────────────────────────────
        log.info("Opening gripper to deposit ball.")
        robot.send("open_gripper")
        time.sleep(0.5)


if __name__ == "__main__":
    camera = Camera()
    robot  = Robot(detector=camera._detector, backend=_ev3_backend)

    camera.start_detection(model_path="best.pt", confidence=0.30)
    camera.preview()

    try:
        run_autonomous(robot)
    finally:
        camera.close()
