"""
ev3-side.py — Robot socket server for EV3 (ev3dev, Python 3.5+)

Deploy this file to the EV3 and run:
    python3 ev3-side.py

The MacBook connects and sends newline-terminated command strings
matching the Command enum in robot/commands.py.
"""

# ── Configurable parameters ─────────────────────────────────────────────────

HOST            = "0.0.0.0"   # Listen on all interfaces
PORT            = 9999         # Must match the port used by the MacBook
DRIVE_SPEED     = -400         # Motor speed for driving (deg/s). Negate if forward drives backward.
DRIVE_DURATION  = 0.3          # Seconds each drive command runs
TURN_SPEED      = 300          # Motor speed for turning (deg/s)
TURN_DURATION   = 0.2          # Seconds each turn command runs. Reduce to turn less per command.
GRIPPER_SPEED   = 200          # Motor speed for gripper (deg/s)
GRIPPER_OPEN_POS  = 120        # Gripper open position relative to close (deg)
GRIPPER_CLOSE_POS = -120        # Gripper close angle from open position (deg)

# Port A = left drive motor, Port B = right drive motor, Port C = gripper
MOTOR_LEFT_PORT    = "outA"
MOTOR_RIGHT_PORT   = "outB"
MOTOR_GRIPPER_PORT = "outC"

# ── Imports ──────────────────────────────────────────────────────────────────

import socket
import time
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("ev3-side")

try:
    import ev3dev.ev3 as ev3
except ImportError:
    log.error("ev3dev library not found. Run this script on the EV3.")
    sys.exit(1)

# ── Motor initialisation ─────────────────────────────────────────────────────

motor_left    = ev3.LargeMotor(MOTOR_LEFT_PORT)
motor_right   = ev3.LargeMotor(MOTOR_RIGHT_PORT)
motor_gripper = ev3.MediumMotor(MOTOR_GRIPPER_PORT)

def _assert_connected(motor, name):
    if not motor.connected:
        log.error("Motor '%s' not connected on %s", name, motor.address)
        sys.exit(1)

_assert_connected(motor_left,    "left")
_assert_connected(motor_right,   "right")
_assert_connected(motor_gripper, "gripper")

log.info("All motors connected.")

# ── Command handlers ─────────────────────────────────────────────────────────

def stop_drive():
    motor_left.stop(stop_action="brake")
    motor_right.stop(stop_action="brake")

def drive(left_sign, right_sign):
    motor_left.run_timed(
        speed_sp=left_sign * DRIVE_SPEED,
        time_sp=int(DRIVE_DURATION * 1000),
        stop_action="brake",
    )
    motor_right.run_timed(
        speed_sp=right_sign * DRIVE_SPEED,
        time_sp=int(DRIVE_DURATION * 1000),
        stop_action="brake",
    )
    time.sleep(DRIVE_DURATION)

def turn(left_sign, right_sign):
    motor_left.run_timed(
        speed_sp=left_sign * TURN_SPEED,
        time_sp=int(TURN_DURATION * 1000),
        stop_action="brake",
    )
    motor_right.run_timed(
        speed_sp=right_sign * TURN_SPEED,
        time_sp=int(TURN_DURATION * 1000),
        stop_action="brake",
    )
    time.sleep(TURN_DURATION)

def gripper_move(position_deg):
    motor_gripper.run_to_rel_pos(
        position_sp=position_deg,
        speed_sp=GRIPPER_SPEED,
        stop_action="hold",
    )
    motor_gripper.wait_while("running", timeout=3000)

HANDLERS = {
    "forward":       lambda: drive(+1, +1),
    "backward":      lambda: drive(-1, -1),
    "left":          lambda: turn(-1, +1),
    "right":         lambda: turn(+1, -1),
    "open_gripper":  lambda: gripper_move(GRIPPER_OPEN_POS),
    "close_gripper": lambda: gripper_move(GRIPPER_CLOSE_POS),
}

# ── Socket server ────────────────────────────────────────────────────────────

def handle_client(conn, addr):
    log.info("Client connected: %s", addr)
    buf = ""
    try:
        while True:
            chunk = conn.recv(256)
            if not chunk:
                break
            buf += chunk.decode("utf-8", errors="ignore")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                cmd = line.strip()
                if not cmd:
                    continue
                log.info("Received: %r", cmd)
                handler = HANDLERS.get(cmd)
                if handler:
                    try:
                        handler()
                        conn.sendall(b"ok\n")
                    except Exception as exc:
                        log.error("Handler error for %r: %s", cmd, exc)
                        conn.sendall(b"error\n")
                else:
                    log.warning("Unknown command: %r", cmd)
                    conn.sendall(b"unknown\n")
    except Exception as exc:
        log.error("Connection error: %s", exc)
    finally:
        conn.close()
        log.info("Client disconnected: %s", addr)

def _local_ips():
    """Return all non-loopback IPv4 addresses on this machine."""
    ips = []
    try:
        import subprocess
        out = subprocess.check_output(["hostname", "-I"]).decode()
        ips = [ip for ip in out.split() if not ip.startswith("127.")]
    except Exception:
        pass
    if not ips:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ips = [s.getsockname()[0]]
            s.close()
        except Exception:
            ips = ["<unknown>"]
    return ips


def main():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(1)

    ips = _local_ips()
    print("=" * 48)
    print("  EV3 robot server READY")
    print("  Port : {}".format(PORT))
    for ip in ips:
        print("  IP   : {}".format(ip))
    print("")
    print("  Connect from MacBook:")
    for ip in ips:
        print("    {}:{}".format(ip, PORT))
    print("=" * 48)

    while True:
        conn, addr = srv.accept()
        handle_client(conn, addr)  # single-client; one at a time

if __name__ == "__main__":
    main()

