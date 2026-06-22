import code
import logging
import sys

from camera import Camera


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

BANNER = """
Camera REPL  —  available commands:

  camera.preview()          Open live preview window (BLOCKS until closed)
                              r  start recording     (auto-named .mp4)
                              s  stop  recording
                              c  capture frame       (auto-named .jpg)
                              q  quit preview & shut down camera

  camera.capture("f.jpg")   Save a single frame to disk
  camera.capture()          Return latest frame as a NumPy array (BGR)
  camera.record("v.mp4")    Start background recording
  camera.stop_recording()   Finalize and stop recording
  camera.close()            Release all hardware and threads
"""

if __name__ == "__main__":
    code.interact(banner=BANNER, local=globals())