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
        self._pending_clicks: list[tuple] = []
        self._target_objects: list[dict] = []
        self._next_object_id: int = 0
        self._current_path: list[list[int]] | None = None

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
            self._pending_clicks = []
            self._target_objects = []
            self._next_object_id = 0
            self._current_path = None

        with self._detections_lock:
            self._latest_detections = []

        logger.info("Detector stopped.")

    def get_detections(self) -> list[dict]:
        """Returns a snapshot of the latest detections.  Thread-safe."""
        with self._detections_lock:
            dets = list(self._latest_detections)
        with self._state_lock:
            current_path = getattr(self, "_current_path", None)
        if current_path:
            dets.append({
                "class_name": "path",
                "points": current_path,
                "confidence": 1.0,
            })
        return dets

    def set_click_target(self, x, y) -> None:
        """Adds coordinates of a new target object to track."""
        with self._state_lock:
            self._pending_clicks.append((x, y))
            logger.info(f"Detector click target added: x={x}, y={y}")

    def clear_tracked_objects(self) -> None:
        """Clears all tracked targets."""
        with self._state_lock:
            self._target_objects = []
            self._next_object_id = 0
            self._current_path = None
            logger.info("Cleared all tracked objects and path.")

    def set_current_path(self, points: list[list[int]] | None) -> None:
        """Sets the current planned path waypoints to show in preview."""
        with self._state_lock:
            self._current_path = points
            logger.info(f"Detector path updated: {len(points) if points else 0} points")

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
          - SAM 2 + DINOv2 visual tracking ('SAM2+DINOv2')
        """
        if model_path == "SAM2+DINOv2":
            import torch
            import torchvision.transforms.functional as TF
            from ultralytics import SAM
            import numpy as np

            # Lazy load models
            if not hasattr(self, "_cached_dino_model"):
                logger.info("Loading DINOv2 model (dinov2_vits14)...")
                self._cached_dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
                self._cached_dino_model.eval()
                logger.info("DINOv2 model loaded successfully.")

            if not hasattr(self, "_cached_sam_model") or getattr(self, "_cached_sam_path", None) != "sam2_t.pt":
                logger.info("Loading SAM model from sam2_t.pt...")
                self._cached_sam_model = SAM("sam2_t.pt")
                self._cached_sam_path = "sam2_t.pt"
                logger.info("SAM model sam2_t.pt loaded successfully.")

            h_orig, w_orig = frame.shape[:2]

            with self._state_lock:
                clicks = list(self._pending_clicks)
                self._pending_clicks.clear()

            if len(clicks) > 0 or len(self._target_objects) > 0:
                device = torch.device("cpu")
                self._cached_dino_model.to(device)

                img_t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                img_t = img_t[[2, 1, 0], :, :]  # BGR to RGB
                img_t = TF.resize(img_t, (448, 448), antialias=True)
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_t = (img_t - mean) / std
                img_t = img_t.unsqueeze(0).to(device)

                with torch.no_grad():
                    features = self._cached_dino_model.forward_features(img_t)
                    patch_tokens = features["x_norm_patchtokens"]  # [1, 1024, 384]

                # Process any new clicks to register targets
                for click_x, click_y in clicks:
                    click_x_448 = click_x * (448.0 / w_orig)
                    click_y_448 = click_y * (448.0 / h_orig)

                    col = int(click_x_448 // 14)
                    row = int(click_y_448 // 14)
                    col = max(0, min(31, col))
                    row = max(0, min(31, row))

                    patch_idx = row * 32 + col
                    target_token = patch_tokens[0, patch_idx]
                    target_token = target_token / (target_token.norm(dim=-1, keepdim=True) + 1e-8)
                    
                    with self._state_lock:
                        obj_id = self._next_object_id
                        self._next_object_id += 1
                        self._target_objects.append({
                            "id": obj_id,
                            "token": target_token
                        })
                    logger.info(f"Target token initialized for object_{obj_id} from click ({click_x}, {click_y}) -> patch row={row}, col={col}")

                # Track all active target objects
                detections = []
                if len(self._target_objects) > 0:
                    patch_tokens_norm = patch_tokens / (patch_tokens.norm(dim=-1, keepdim=True) + 1e-8)
                    for obj in self._target_objects:
                        obj_id = obj["id"]
                        obj_token = obj["token"]

                        similarity = torch.matmul(patch_tokens_norm[0], obj_token)
                        max_val, max_idx = torch.max(similarity, dim=0)
                        score = max_val.item()

                        if score >= confidence:
                            matched_row = max_idx.item() // 32
                            matched_col = max_idx.item() % 32

                            x_448 = (matched_col + 0.5) * 14
                            y_448 = (matched_row + 0.5) * 14

                            matched_x = int(x_448 * (w_orig / 448.0))
                            matched_y = int(y_448 * (h_orig / 448.0))

                            sam_results = self._cached_sam_model(frame, points=[[matched_x, matched_y]], labels=[1], verbose=False)[0]
                            
                            if sam_results.masks is not None and len(sam_results.masks.xy) > 0:
                                polygon = sam_results.masks.xy[0]
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
                                        "cx": int(matched_x),
                                        "cy": int(matched_y),
                                        "class_name": f"object_{obj_id}",
                                        "confidence": float(score),
                                        "polygon": polygon.tolist(),
                                    })
                        else:
                            logger.info(f"Object_{obj_id} similarity ({score:.3f}) below confidence threshold ({confidence}). Object lost.")

                return detections
            
            return []

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
