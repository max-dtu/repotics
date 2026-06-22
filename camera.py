""" 
A minimal REPL-driven camera interface for capturing images, previewing video, and recording clips using OpenCV.

The Camera class exposes a simple, state-safe interface for interacting with a webcam from a Python REPL. It enforces exclusive camera usage, meaning only one operation (preview, record, or capture) can access the camera at a time.

camera.record(vid.mp4)  cannot run while preview is active.
camera.preview() can not run while recording.
camera.capture() can not run while recording or previewing.

Valid modes:

idle
previewing
recording
capture (instant, non-continuous)

"""

import cv2
import threading


class Camera:
    def __init__(self, index=0):
        self.index = index
        self.cap = None

        self._lock = threading.Lock()

        self._recording = False
        self._previewing = False

        self._writer = None
        self._thread = None

    def open(self):
        with self._lock:
            if self.cap is None:
                self.cap = cv2.VideoCapture(self.index)

    def preview(self):
        if self._recording or self._previewing:
            raise RuntimeError("Camera already in use")

        self.open()

        self._previewing = True

        def loop():
            import cv2

            while self._previewing:
                with self._lock:
                    ok, frame = self.cap.read()

                if not ok:
                    break

                cv2.imshow("camera", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            cv2.destroyAllWindows()
            self._previewing = False

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def record(self, path, fps=30):
        if self._recording or self._previewing:
            raise RuntimeError("Camera already in use")

        self.open()

        with self._lock:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            self._writer = cv2.VideoWriter(
                path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )

        def loop():
            while self._recording:
                with self._lock:
                    ok, frame = self.cap.read()

                if not ok:
                    break

                self._writer.write(frame)

        self._recording = True  # MUST be before thread starts
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop_recording(self):
        if not self._recording:
            return

        self._recording = False

        if self._thread:
            self._thread.join()
            self._thread = None

        if self._writer:
            self._writer.release()
            self._writer = None

    def close(self):
        self._recording = False
        self._previewing = False

        if self._thread:
            self._thread.join(timeout=2)

        with self._lock:
            if self._writer:
                self._writer.release()
                self._writer = None

            if self.cap is not None:
                self.cap.release()
                self.cap = None

        cv2.destroyAllWindows()

    def capture(self, path=None):
        if self._recording or self._previewing:
            raise RuntimeError("Camera already in use")

        self.open()

        with self._lock:
            ok, frame = self.cap.read()

        if not ok:
            return None

        if path:
            cv2.imwrite(path, frame)

        return frame