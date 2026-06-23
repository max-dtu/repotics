import threading
import queue
import logging

from .frame_reader import FrameReader

logger = logging.getLogger(__name__)


class Detector:
    """
    Owns the object detection inference pipeline.

    Subscribes to FrameReader and runs inference on each incoming frame in a
    dedicated daemon thread.  The latest detections are exposed via
    get_detections() and are safe to call from any thread.

    Detection dict format
    ---------------------
    Each entry in the list returned by get_detections() is a plain dict::

        {
            "x":          int,    # bounding-box left edge (pixels)
            "y":          int,    # bounding-box top edge  (pixels)
            "w":          int,    # bounding-box width     (pixels)
            "h":          int,    # bounding-box height    (pixels)
            "cx":         int,    # centre x
            "cy":         int,    # centre y
            "class_name": str,    # detected class label
            "confidence": float,  # model confidence [0, 1]
        }

    Swapping in a real backend
    --------------------------
    Override (or monkey-patch) ``_run_inference`` to plug in any model::

        from ultralytics import YOLO

        _model = YOLO("yolov8n.pt")

        def _my_backend(self, frame, model_path, confidence, classes):
            results = _model(frame, conf=confidence)[0]
            detections = []
            for box in results.boxes:
                x, y, w, h = map(int, box.xywh[0])
                detections.append({
                    "x": x - w // 2, "y": y - h // 2, "w": w, "h": h,
                    "cx": x, "cy": y,
                    "class_name": _model.names[int(box.cls)],
                    "confidence": float(box.conf),
                })
            return detections

        Detector._run_inference = _my_backend
    """

    def __init__(self, reader: FrameReader) -> None:
        self._reader = reader
        self._state_lock = threading.Lock()

        self._running = False
        self._thread: threading.Thread | None = None
        self._queue: queue.Queue | None = None
        self._stop_event: threading.Event | None = None

        self._detections_lock = threading.Lock()
        self._latest_detections: list[dict] = []

        self._model_path: str | None = None
        self._confidence: float = 0.5
        self._classes: list[str] | None = None
        self._imgsz: int | tuple | None = None
        self._first_frame = False

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def model_path(self) -> str | None:
        return self._model_path

    @model_path.setter
    def model_path(self, value: str | None) -> None:
        with self._state_lock:
            if self._model_path != value:
                self._model_path = value
                logger.info(f"Detector model path updated to: {value}")

    @property
    def confidence(self) -> float:
        return self._confidence

    @confidence.setter
    def confidence(self, value: float) -> None:
        with self._state_lock:
            if self._confidence != value:
                self._confidence = value
                logger.info(f"Detector confidence threshold updated to: {value}")

    @property
    def classes(self) -> list[str] | None:
        return self._classes

    @classes.setter
    def classes(self, value: list[str] | None) -> None:
        with self._state_lock:
            if self._classes != value:
                self._classes = value
                logger.info(f"Detector class filter updated to: {value}")

    @property
    def imgsz(self) -> int | tuple | None:
        return self._imgsz

    @imgsz.setter
    def imgsz(self, value: int | tuple | None) -> None:
        with self._state_lock:
            if self._imgsz != value:
                self._imgsz = value
                logger.info(f"Detector imgsz updated to: {value}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(
        self,
        model_path: str | None = None,
        *,
        confidence: float = 0.5,
        classes: list[str] | None = None,
        imgsz: int | tuple | None = None,
    ) -> None:
        """
        Start the detection pipeline.  Idempotent — safe to call if already running.

        Parameters
        ----------
        model_path:  Path to model weights (passed through to _run_inference).
                     Defaults to 'yolov8n.pt' if not provided.
        confidence:  Minimum confidence threshold for reported detections.
        classes:     Optional allow-list of class names (None = all classes).
        imgsz:       Optional image size for inference (e.g., 320, 640).
        """
        with self._state_lock:
            if self._thread and self._thread.is_alive():
                logger.warning("Previous detector thread is still active. Waiting for it to terminate...")
                self._thread.join(timeout=5.0)
                if self._thread.is_alive():
                    logger.error("Cannot start: previous detector thread is still running.")
                    return

            if self._running:
                logger.info("Detector is already running.")
                return

            self._model_path = model_path
            self._confidence = confidence
            self._classes = classes
            self._imgsz = imgsz
            self._first_frame = True

            q: queue.Queue = queue.Queue(maxsize=1)
            stop_event = threading.Event()
            self._queue = q
            self._stop_event = stop_event
            self._running = True

            self._reader.subscribe(q, drop_stale=True)

            self._thread = threading.Thread(
                target=self._inference_loop,
                args=(q, stop_event),
                name="DetectorThread",
                daemon=True,
            )
            self._thread.start()
            logger.info("Detector thread started.")

    def stop(self) -> None:
        """Stop the detection pipeline and clear the latest detections."""
        with self._state_lock:
            if not self._running:
                return
            logger.info("Stopping detector...")
            self._running = False

            if self._stop_event:
                self._stop_event.set()

            if self._queue:
                self._reader.unsubscribe(self._queue)
                try:
                    self._queue.put_nowait(None)  # stop sentinel
                except Exception:
                    pass

            thread = self._thread
            self._queue = None
            self._stop_event = None

        if thread and thread.is_alive():
            thread.join(timeout=5.0)

        with self._state_lock:
            if self._thread is thread:
                self._thread = None

        with self._detections_lock:
            self._latest_detections = []

        logger.info("Detector stopped.")

    def get_detections(self) -> list[dict]:
        """Returns a snapshot of the latest detections.  Thread-safe."""
        with self._detections_lock:
            return list(self._latest_detections)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _inference_loop(
        self,
        q: queue.Queue,
        stop_event: threading.Event,
    ) -> None:
        logger.info("Detector inference loop active.")
        try:
            while not stop_event.is_set():
                try:
                    frame = q.get(timeout=0.1)
                except queue.Empty:
                    continue

                if frame is None:
                    logger.info("Detector received stop sentinel.")
                    break

                with self._state_lock:
                    model_path = self._model_path
                    confidence = self._confidence
                    classes = self._classes
                    imgsz = self._imgsz

                detections = self._run_inference(frame, model_path, confidence, classes, imgsz)
                
                if self._first_frame:
                    self._first_frame = False
                    logger.info("Detector pipeline is running and processing frames.")

                with self._detections_lock:
                    self._latest_detections = detections

        except Exception:
            logger.exception("Unexpected exception in detector inference loop.")
        finally:
            with self._detections_lock:
                self._latest_detections = []
            logger.info("Detector inference loop stopped.")

    def _run_inference(
        self,
        frame,
        model_path: str | None,
        confidence: float,
        classes: list[str] | None,
        imgsz: int | tuple | None = None,
    ) -> list[dict]:
        """
        Runs object detection/segmentation using YOLOv8 and/or SAM models.
        Supports:
          - YOLOv8 object detection (e.g., 'yolov8n.pt')
          - YOLOv8 segmentation (e.g., 'yolov8n-seg.pt')
          - SAM automatic segmentation (e.g., 'mobile_sam.pt', 'sam_b.pt')
          - YOLOv8 detection + SAM prompted segmentation (e.g., 'yolov8n.pt+mobile_sam.pt')
        """
        # Lazy imports
        from ultralytics import YOLO, SAM
        import numpy as np

        # Decide default model path if none is provided
        if not model_path:
            model_path = "yolov8n.pt"

        # Parse model paths if we have a combined YOLO + SAM (e.g., "yolov8n.pt+mobile_sam.pt")
        yolo_path = None
        sam_path = None

        if "+" in model_path:
            parts = model_path.split("+")
            yolo_path = parts[0].strip()
            sam_path = parts[1].strip()
        elif "sam" in model_path.lower():
            sam_path = model_path
        else:
            yolo_path = model_path

        # Load and cache YOLO model if needed
        if yolo_path:
            if not hasattr(self, "_cached_yolo_model") or getattr(self, "_cached_yolo_path", None) != yolo_path:
                logger.info(f"Loading YOLO model from {yolo_path}...")
                self._cached_yolo_model = YOLO(yolo_path)
                self._cached_yolo_path = yolo_path
                logger.info(f"YOLO model {yolo_path} loaded successfully.")

        # Load and cache SAM model if needed
        if sam_path:
            if not hasattr(self, "_cached_sam_model") or getattr(self, "_cached_sam_path", None) != sam_path:
                logger.info(f"Loading SAM model from {sam_path}...")
                self._cached_sam_model = SAM(sam_path)
                self._cached_sam_path = sam_path
                logger.info(f"SAM model {sam_path} loaded successfully.")

        detections = []

        # Mode 1: Combined YOLO + SAM
        if yolo_path and sam_path:
            yolo_kwargs = {"conf": confidence, "verbose": False}
            if imgsz is not None:
                yolo_kwargs["imgsz"] = imgsz
            yolo_results = self._cached_yolo_model(frame, **yolo_kwargs)[0]
            bboxes = []
            yolo_dets = []

            for box in yolo_results.boxes:
                x_c, y_c, w, h = map(float, box.xywh[0])
                cls_idx = int(box.cls[0])
                class_name = self._cached_yolo_model.names[cls_idx]
                conf = float(box.conf[0])

                if classes is not None and class_name not in classes:
                    continue

                x1, y1, x2, y2 = map(float, box.xyxy[0])
                bboxes.append([x1, y1, x2, y2])

                yolo_dets.append({
                    "x": int(x_c - w / 2),
                    "y": int(y_c - h / 2),
                    "w": int(w),
                    "h": int(h),
                    "cx": int(x_c),
                    "cy": int(y_c),
                    "class_name": class_name,
                    "confidence": conf,
                })

            if bboxes:
                # Prompt SAM with YOLO bounding boxes
                sam_results = self._cached_sam_model(frame, bboxes=bboxes, verbose=False)[0]
                if sam_results.masks is not None:
                    for i, det in enumerate(yolo_dets):
                        if i < len(sam_results.masks.xy):
                            polygon = sam_results.masks.xy[i]
                            if len(polygon) > 0:
                                det["polygon"] = polygon.tolist()

            detections = yolo_dets

        # Mode 2: YOLO Only (Detection or Segmentation)
        elif yolo_path:
            yolo_kwargs = {"conf": confidence, "verbose": False}
            if imgsz is not None:
                yolo_kwargs["imgsz"] = imgsz
            results = self._cached_yolo_model(frame, **yolo_kwargs)[0]
            has_masks = results.masks is not None

            for i, box in enumerate(results.boxes):
                x_c, y_c, w, h = map(float, box.xywh[0])
                cls_idx = int(box.cls[0])
                class_name = self._cached_yolo_model.names[cls_idx]
                conf = float(box.conf[0])

                if classes is not None and class_name not in classes:
                    continue

                det = {
                    "x": int(x_c - w / 2),
                    "y": int(y_c - h / 2),
                    "w": int(w),
                    "h": int(h),
                    "cx": int(x_c),
                    "cy": int(y_c),
                    "class_name": class_name,
                    "confidence": conf,
                }

                if has_masks and i < len(results.masks.xy):
                    polygon = results.masks.xy[i]
                    if len(polygon) > 0:
                        det["polygon"] = polygon.tolist()

                detections.append(det)

        # Mode 3: SAM Only (Automatic Segmentation)
        elif sam_path:
            results = self._cached_sam_model(frame, verbose=False)[0]
            if results.masks is not None:
                for i, polygon in enumerate(results.masks.xy):
                    if len(polygon) > 0:
                        x_min, y_min = np.min(polygon, axis=0)
                        x_max, y_max = np.max(polygon, axis=0)
                        w = x_max - x_min
                        h = y_max - y_min
                        detections.append({
                            "x": int(x_min),
                            "y": int(y_min),
                            "w": int(w),
                            "h": int(h),
                            "cx": int(x_min + w / 2),
                            "cy": int(y_min + h / 2),
                            "class_name": f"object_{i}",
                            "confidence": 1.0,
                            "polygon": polygon.tolist(),
                        })

        return detections
