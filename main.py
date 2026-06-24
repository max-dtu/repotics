import code
import logging
import socket
import sys

from camera import Camera
from robot import Robot, Command, Task, Subtask

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
            _sock.recv(16)   # consume 'ok\n' / 'error\n'
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
    """
    Logging handler for interactive REPL use.

    Background threads write log lines to the same terminal as the readline
    prompt. Prefixing each line with \\r (carriage return) jumps to column 0
    and overwrites any partially-displayed `>>> ` before printing the message,
    preventing garbled terminal output.
    """

    def emit(self, record):
        try:
            msg = self.format(record)
            self.stream.write("\r" + msg + "\n")
            self.stream.flush()
        except Exception:
            self.handleError(record)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(threadName)s) %(message)s",
    handlers=[_ReadlineAwareHandler(sys.stderr)],
)

camera = Camera()

# robot shares the same Detector instance — no duplicate subscriptions.
# Call camera.start_detection() before robot.assess() / run_until_complete()
# to get real detection results (stub returns [] until then).
robot = Robot(detector=camera._detector, backend=_ev3_backend)

BANNER = """
  robot.assess()                           — see all detected objects
  robot.go_to_nearest_ball()               — drive to the closest ball (auto)
  robot.go_to_nearest_ball(execute=False)  — plan path only, no movement
  robot.find_path("robot", "white_ball")   — drive robot to white ball
  robot.find_path("robot", "orange_ball")  — drive robot to orange ball
  robot.find_path("white_ball", "big_goal")— plan ball → goal path
  robot.find_path(..., execute=False)      — plan only (no movement)
  robot.send("forward")                    — send one manual command
  camera.preview()                         — live window (already open)
  camera.start_detection()                 — restart detection pipeline
  camera.close()                           — clean shutdown
"""

if __name__ == "__main__":
    camera.start_detection(model_path="best.pt", confidence=0.30)
    camera.preview()
    try:
        code.interact(banner=BANNER, local=globals())
    finally:
        camera.close()
