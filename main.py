import code
import json
import logging
import os
import socket
import sys
import time

from camera import Camera
from robot  import Robot, Command, Task, Subtask

# ── EV3 connection ──────────────────────────────────────────────────────────
# Change this to the IP address printed by ev3-side.py on startup:
EV3_HOST = "10.45.151.18"   
EV3_PORT = 9999

_sock = None

def _ev3_backend(command: Command) -> bool:
    """Send command over TCP or ntfy.sh HTTP endpoint."""
    global _sock
    
    if EV3_HOST.startswith("dweet:") or EV3_HOST.startswith("ntfy:"):
        robot_id = EV3_HOST.split(":", 1)[1]
        import urllib.request
        
        url = "https://ntfy.sh/{}".format(robot_id)
        try:
            req = urllib.request.Request(
                url,
                data=command.value.encode("utf-8"),
                headers={"User-Agent": "Mozilla/5.0"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                resp.read()
            return True
        except Exception as exc:
            logging.getLogger(__name__).error("Failed to send command to ntfy.sh: %s", exc)
            return False

    payload = (command.value + "\n").encode()
    for attempt in range(2):
        try:
            if _sock is None:
                _sock = socket.create_connection((EV3_HOST, EV3_PORT), timeout=3)
                logging.getLogger(__name__).info("Connected to EV3 at %s:%d", EV3_HOST, EV3_PORT)
            _sock.sendall(payload)
            _sock.recv(16)   # consume 'ok\n' / 'error\n'
            return True
        except Exception as exc:
            logging.getLogger(__name__).warning("EV3 send failed (%s); reconnecting…", exc)
            try:
                _sock.close()
            except Exception:
                pass
            _sock = None
    logging.getLogger(__name__).error("Could not send %r to EV3 after retry.", command.value)
    return False
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

def apply_logging_config():
    config_path = "logging_config.json"
    default_config = {
        "detection": False,
        "path_finding": True,
        "follow_path": True,
        "robot": True
    }
    
    if not os.path.exists(config_path):
        try:
            with open(config_path, "w") as f:
                json.dump(default_config, f, indent=4)
            config = default_config
        except Exception as e:
            logging.getLogger(__name__).warning("Could not write default logging config: %s", e)
            config = default_config
    else:
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
        except Exception as e:
            logging.getLogger(__name__).warning("Could not read logging config, using defaults: %s", e)
            config = default_config

    stage_loggers = {
        "detection": [
            "camera.detector",
            "camera.camera",
            "camera.frame_reader",
            "camera.recorder",
            "camera.preview",
        ],
        "path_finding": [
            "robot.planner",
            "robot.spatial",
        ],
        "follow_path": [
            "robot.agent",
            "robot.evaluator",
        ],
        "robot": [
            "robot.robot",
            "robot.commander",
            "robot.task",
        ],
    }

    for stage, loggers in stage_loggers.items():
        enabled = config.get(stage, True)
        level = logging.INFO if enabled else logging.WARNING
        for name in loggers:
            logging.getLogger(name).setLevel(level)
            
    logging.getLogger(__name__).info("Applied logging config: %s", config)

def reload_logging():
    """Reload and apply the logging configuration from logging_config.json."""
    apply_logging_config()

apply_logging_config()

camera = Camera()

# robot shares the same Detector instance — no duplicate subscriptions.
# Call camera.start_detection() before robot.assess() / run_until_complete()
# to get real detection results (stub returns [] until then).
robot  = Robot(detector=camera._detector, backend=_ev3_backend)

BANNER = """
  camera.preview()                       — live window (already open)
  camera.get_detections()                — raw detection list
  robot.assess()                         — structured world state
  robot.send("forward")                  — send one command
  robot.find_path("object_0", "object_1")— plan + execute A* path
  robot.find_path(..., execute=False)    — plan only (no movement)
  camera.start_detection()               — restart detection pipeline
  reload_logging()                       — reload logging config from logging_config.json
  camera.close()                         — clean shutdown
"""

if __name__ == "__main__":
    camera.start_detection(model_path="best.pt")
    camera.preview()
    try:
        print("\n[INFO] Running automatic object detection to locate Car and WhiteBall...")
        frame = camera.capture()
        from ultralytics import YOLO
        import numpy as np
        
        # Load YOLO model
        yolo_model = YOLO("best.pt")
        results = yolo_model(frame, verbose=False)[0]
        
        car_box = None
        ball_box = None
        
        for box in results.boxes:
            cls_idx = int(box.cls[0])
            cls_name = yolo_model.names[cls_idx]
            bx1, by1, bx2, by2 = map(int, box.xyxy[0])
            if cls_name == "Car" and car_box is None:
                car_box = (bx1, by1, bx2, by2)
            elif cls_name == "WhiteBall" and ball_box is None:
                ball_box = (bx1, by1, bx2, by2)
                
        if car_box is None:
            print("[WARNING] Could not automatically detect 'Car' in the frame! Using fallback coordinates.")
            car_box = (357, 777, 437, 796)
            car_heading = (334, 762, 485, 688)
        else:
            cx, cy = (car_box[0] + car_box[2]) // 2, (car_box[1] + car_box[3]) // 2
            car_heading = (cx, cy, cx + 50, cy)  # default heading facing right
            
        if ball_box is None:
            print("[WARNING] Could not automatically detect 'WhiteBall' in the frame! Using fallback coordinates.")
            ball_box = (700, 679, 723, 720)
            ball_heading = (707, 670, 696, 705)
        else:
            cx, cy = (ball_box[0] + ball_box[2]) // 2, (ball_box[1] + ball_box[3]) // 2
            ball_heading = (cx, cy, cx + 10, cy + 10)
            
        print(f"[INFO] Initializing Car box at: {car_box}")
        print(f"[INFO] Initializing WhiteBall box at: {ball_box}")
        
        # Register targets programmatically
        camera._detector.set_box_target(car_box[0], car_box[1], car_box[2], car_box[3], 
                                        car_heading[0], car_heading[1], car_heading[2], car_heading[3])
        camera._detector.set_box_target(ball_box[0], ball_box[1], ball_box[2], ball_box[3],
                                        ball_heading[0], ball_heading[1], ball_heading[2], ball_heading[3])

        print("[INFO] Starting live tracking...")
        camera._detector.start_tracking()
        
        # Wait for the first frame to be processed
        print("[INFO] Waiting for the first camera frame to be processed...")
        camera._detector.wait_for_fresh_detections(0, timeout=10.0)

        # Force class names to guarantee they are registered correctly
        with camera._detector._state_lock:
            for obj in camera._detector._target_objects:
                if obj["id"] == 0:
                    obj["class_name"] = "Car"
                elif obj["id"] == 1:
                    obj["class_name"] = "WhiteBall"

        # Let the detector thread run another cycle to update centroids/snappings
        cnt = camera._detector.get_detections_counter()
        camera._detector.wait_for_fresh_detections(cnt, timeout=5.0)
            
        # Assess state to locate the annotated objects
        state = robot.assess()
        objects_by_class = state.get("objects_by_class", {})
        
        # Find car and white ball instances
        car_classes = [c for c in objects_by_class if c.startswith("Car")]
        whiteball_classes = [c for c in objects_by_class if c.startswith("WhiteBall")]
        
        if not car_classes:
            print("\n[ERROR] No 'Car' object found in annotations!")
        elif not whiteball_classes:
            print("\n[ERROR] No 'WhiteBall' object found in annotations!")
        else:
            import random
            car_cls = car_classes[0]
            whiteball_cls = random.choice(whiteball_classes)
            print(f"\n[INFO] Found Car(s): {car_classes}")
            print(f"[INFO] Found WhiteBall(s): {whiteball_classes}")
            print(f"[INFO] Selected starting robot: {car_cls}")
            print(f"[INFO] Selected destination ball: {whiteball_cls}")
            print(f"[INFO] Planning and executing path from {car_cls} to {whiteball_cls}...\n")
            
            # This plans and executes the path using the A* path planner
            robot.find_path(car_cls, whiteball_cls)
            
        print("\nDropping to interactive python shell...")
        code.interact(banner=BANNER, local=globals())
    finally:
        camera.close()








