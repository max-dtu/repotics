import code
import logging
import sys

from camera import Camera
from robot  import Robot, Command, Task, Subtask


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
robot  = Robot(detector=camera._detector)

BANNER = """
repotics REPL  —  camera + robot agent
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CAMERA
  camera.preview()              Live preview window (BLOCKS until closed)
                                   r  start recording        (auto-named .mp4)
                                   s  stop  recording
                                   c  capture frame          (auto-named .jpg)
                                   d  toggle detection       (bounding boxes)
                                   q  quit preview & shut down camera
  camera.capture("f.jpg")       Save a single frame to disk
  camera.capture()              Return latest frame as NumPy array (BGR)
  camera.record("v.mp4")        Start background recording
  camera.stop_recording()       Stop and finalize recording
  camera.start_detection()      Start object detection pipeline
  camera.stop_detection()       Stop  object detection pipeline
  camera.get_detections()       Latest detections [{x,y,w,h,cx,cy,class,conf}]
  camera.close()                Release all hardware and threads

ROBOT — single commands
  robot.send("forward")         Send one command directly
  robot.send(Command.LEFT)      Same using enum (tab-completable)
  robot.step()                  Assess → pick random command → send → return it
  robot.step(policy=fn)         Same with custom policy: fn(state) -> Command
  robot.run_until_complete(     Loop: assess → check goal → step, until done
      goal,                        goal: callable (state)->bool
      max_steps=100,
      step_delay=0.5,
      policy="random")
  robot.find_path(src, dst)     Plan path src→dst, execute it  (stub: returns [])
  robot.find_best_path(         Plan best path src→dst (lowest cost), execute it
      src, dst, cost_fn=None)
  robot.is_path_clear(src, dst) Check if direct line src→dst is clear of obstacles
  robot.assess()                Current world state from camera detections
  robot.available_commands      List all commands
  robot.spatial                 Geometric queries helper (e.g. find_nearest)

TASK MISSIONS — ordered subtask sequences
  task = Task("mission name")   Create a mission  (fail_fast=True by default)
  task.then(Subtask(            Append a subtask:
      name="approach",
      description="Come within 1 cm of the ball",
      goal=lambda s: ...,       termination condition (state dict → bool)
      max_steps=60,             hard cap (default 100)
      step_delay=0.3,           pause between steps (default 0.5 s)
      policy="random",          "random" or fn(state)->Command
      on_success=fn,            optional callback(SubtaskResult)
      on_failure=fn,            optional callback(SubtaskResult)
  ))
  robot.run_task(task)          Execute all subtasks in order → TaskResult
  result.print_summary()        Print a formatted outcome table
  result.success                True iff all subtasks succeeded
  result.subtask_results        List of SubtaskResult objects

  NOTE: call camera.start_detection() before run_task() to get real
        detection results (stub returns [] until a model is wired in).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

if __name__ == "__main__":
    code.interact(banner=BANNER, local=globals())