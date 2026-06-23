import cv2
import logging

from .frame_reader import FrameReader
from .recorder import VideoRecorder
from .detector import Detector
from .preview import PreviewManager

logger = logging.getLogger(__name__)


class Camera:
    """
    Thin facade that composes FrameReader, VideoRecorder, Detector, and
    PreviewManager.  The public API is identical to the original monolithic
    Camera class, extended with detection methods.

    Composition
    -----------
    FrameReader      — hardware, frame loop, subscriber fan-out
    VideoRecorder    — frame queue → VideoWriter pipeline
    Detector         — inference pipeline, get_detections()
    PreviewManager   — subprocess IPC, feeder thread, cmd dispatcher
    """

    def __init__(self, index: int = 0) -> None:
        self._reader   = FrameReader(index)
        self._recorder = VideoRecorder(self._reader)
        self._detector = Detector(self._reader)
        self._preview  = PreviewManager(self._reader, self._recorder, self._detector)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Opens the camera device and starts frame acquisition.  Idempotent."""
        self._reader.open()

    def close(self) -> None:
        """
        Shuts down in a safe, deterministic order:
        1. Stop and drain the recording pipeline.
        2. Stop the preview subprocess.
        3. Stop the detector.
        4. Stop frame acquisition and release the camera hardware.
        """
        logger.info("Shutting down Camera...")
        self._recorder.stop()
        self._preview.stop()
        self._detector.stop()
        self._reader.close()
        logger.info("Camera shutdown complete.")

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def get_dimensions(self) -> tuple:
        """Returns (width, height) of captured frames.  Thread-safe."""
        return self._reader.get_dimensions()

    def capture(self, path: str | None = None, timeout: float = 5.0):
        """
        Returns the latest frame as a NumPy array (BGR).
        Blocks until the first frame arrives or *timeout* seconds elapse.
        Optionally saves the frame to *path*.
        """
        self.open()
        frame = self._reader.get_frame(timeout=timeout)

        if path:
            logger.info(f"Saving frame to: {path}")
            try:
                ok = cv2.imwrite(path, frame)
                if not ok:
                    raise RuntimeError(f"cv2.imwrite returned False for path: '{path}'")
            except Exception as e:
                raise RuntimeError(f"Failed to save image to '{path}': {e}") from e
            logger.info(f"Frame saved to {path}.")

        return frame

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def preview(self) -> None:
        """
        Opens a live preview window and BLOCKS until the window is closed.

        Keyboard shortcuts in the window:
            r — start recording   (auto-named .mp4)
            s — stop  recording
            c — capture frame     (auto-named .jpg)
            d — toggle detection  (start / stop; boxes drawn when running)
            q — close preview and shut down the camera
        """
        self.open()
        self._preview.start_and_block()
        self.close()

    def stop_preview(self) -> None:
        """Stops the preview subprocess."""
        self._preview.stop()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, path: str, fps: int = 30) -> None:
        """Starts background recording to *path*.  Returns immediately."""
        self.open()
        self._recorder.start(path, fps)

    def stop_recording(self) -> None:
        """Finalizes and stops the current recording."""
        self._recorder.stop()

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def start_detection(
        self,
        model_path: str | None = None,
        *,
        confidence: float = 0.5,
        classes: list[str] | None = None,
    ) -> None:
        """
        Starts the object detection pipeline.

        Parameters
        ----------
        model_path:  Path passed through to Detector._run_inference.
                     Ignored by the built-in stub.
        confidence:  Minimum score threshold for reported detections.
        classes:     Optional allow-list of class names (None = all classes).
        """
        self.open()
        self._detector.start(model_path, confidence=confidence, classes=classes)

    def stop_detection(self) -> None:
        """Stops the object detection pipeline."""
        self._detector.stop()

    def get_detections(self) -> list[dict]:
        """
        Returns the latest detections as a list of dicts::

            [{x, y, w, h, cx, cy, class_name, confidence}, ...]

        Returns an empty list if detection is not running or no frame has been
        processed yet.  Thread-safe.
        """
        return self._detector.get_detections()
