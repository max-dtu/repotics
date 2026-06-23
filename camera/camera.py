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

    def preview(self, block: bool = False) -> None:
        """
        Opens a live preview window.

        By default, runs in the background (non-blocking), allowing you to interact with the
        Python prompt to adjust settings dynamically while the preview is open.
        Pass block=True if you want to block the REPL until the window is closed.

        Keyboard shortcuts in the window:
            r — start recording   (auto-named .mp4)
            s — stop  recording
            c — capture frame     (auto-named .jpg)
            d — toggle detection  (start / stop; boxes drawn when running)
            q — close preview
        """
        self.open()
        if block:
            self._preview.start(block=True)
            self.close()
        else:
            self._preview.start(block=False)

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
        imgsz: int | tuple | None = None,
    ) -> None:
        """
        Starts the object detection pipeline.

        Parameters
        ----------
        model_path:  Path passed through to Detector._run_inference.
        confidence:  Minimum score threshold for reported detections.
        classes:     Optional allow-list of class names (None = all classes).
        imgsz:       Optional image size for inference (e.g., 320, 640).
        """
        self.open()
        self._detector.start(model_path, confidence=confidence, classes=classes, imgsz=imgsz)

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

    # ------------------------------------------------------------------
    # Dynamic parameter configuration properties
    # ------------------------------------------------------------------

    @property
    def model_path(self) -> str | None:
        """Returns the currently configured model path."""
        return self._detector.model_path

    @model_path.setter
    def model_path(self, value: str | None) -> None:
        """Dynamically updates the model path on the fly."""
        self._detector.model_path = value

    @property
    def confidence(self) -> float:
        """Returns the currently configured detection confidence threshold."""
        return self._detector.confidence

    @confidence.setter
    def confidence(self, value: float) -> None:
        """Dynamically updates the detection confidence threshold on the fly."""
        self._detector.confidence = value

    @property
    def classes(self) -> list[str] | None:
        """Returns the currently configured class filter."""
        return self._detector.classes

    @classes.setter
    def classes(self, value: list[str] | None) -> None:
        """Dynamically updates the class filter on the fly."""
        self._detector.classes = value

    @property
    def imgsz(self) -> int | tuple | None:
        """Returns the currently configured image size."""
        return self._detector.imgsz

    @imgsz.setter
    def imgsz(self, value: int | tuple | None) -> None:
        """Dynamically updates the image size on the fly."""
        self._detector.imgsz = value
