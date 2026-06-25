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
        self._pending_boxes: list[tuple] = []
        self._target_objects: list[dict] = []
        self._next_object_id: int = 0
        self._current_path: list[list[int]] | None = None
        # Maps object_id → stored click point (fallback if no drag heading given).
        self._click_headings: dict[int, tuple[int, int]] = {}
        # Maps object_id → np.ndarray shape (2,): the heading unit vector.
        # Seeded from the user's drag; updated each frame by applying only the
        # rotation DELTA observed in the SAM mask PCA, so two objects with the
        # same shape never interfere with each other.
        self._heading_vecs: dict = {}   # object_id → np.ndarray
        # Maps object_id → last raw PCA axis (for computing rotation delta).
        self._pca_axes: dict = {}       # object_id → np.ndarray | None
        # Maps object_id → last tracked mask centroid (to stabilize SAM prompted segmentation point).
        self._last_centroids: dict = {} # object_id → tuple[float, float]
        # Maps object_id → last tracked bounding box size (to detect stationary state).
        self._last_sizes: dict = {}     # object_id → tuple[int, int]
        self._tracking_active = False
        # True while the object has no current-frame match (keeps last known position).
        self._lost_flags: dict[int, bool] = {}
        # EMA-smoothed bbox per object: (cx, cy, w, h) — reduces jitter.
        self._smoothed_bboxes: dict[int, tuple] = {}
        self._detections_counter = 0
        self._detections_updated = threading.Condition()

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
            self._lost_flags = {}
            self._smoothed_bboxes = {}

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

    def get_detections_counter(self) -> int:
        with self._detections_updated:
            return self._detections_counter

    def wait_for_fresh_detections(self, last_count: int, timeout: float = 2.0) -> bool:
        with self._detections_updated:
            return self._detections_updated.wait_for(
                lambda: self._detections_counter > last_count,
                timeout=timeout
            )

    def set_click_target(self, x, y, heading_x=None, heading_y=None) -> None:
        """Register a new click target.

        Parameters
        ----------
        x, y : int
            Pixel where the user *pressed* — used to locate the DINOv2 patch
            token for the object to track.
        heading_x, heading_y : int, optional
            Pixel where the user *released* — defines the initial heading
            direction (drag vector).  The system updates this automatically
            each frame via mask PCA; the drag is only used to seed the
            direction and resolve the 180° ambiguity.
        """
        import numpy as np
        hx = int(heading_x) if heading_x is not None else int(x)
        hy = int(heading_y) if heading_y is not None else int(y)
        # Compute initial heading unit vector from drag direction
        drag_vec = np.array([hx - x, hy - y], dtype=np.float32)
        mag = np.linalg.norm(drag_vec)
        init_vec = drag_vec / mag if mag > 1e-6 else np.array([1.0, 0.0], dtype=np.float32)
        with self._state_lock:
            # Pre-assign the ID here so rapid successive clicks each get a unique key.
            obj_id = self._next_object_id
            self._next_object_id += 1
            self._click_headings[obj_id] = (hx, hy)
            self._heading_vecs[obj_id] = init_vec
            self._pending_clicks.append((obj_id, x, y))
            logger.info(
                f"Detector click target added: object_{obj_id} "
                f"press=({x},{y}) heading=({hx},{hy}) "
                f"init_vec=({init_vec[0]:.2f},{init_vec[1]:.2f})"
            )

    def set_box_target(self, x1, y1, x2, y2, hx1, hy1, hx2, hy2) -> None:
        """Register a new box target with a heading.

        Parameters
        ----------
        x1, y1, x2, y2 : int
            Bounding box coordinates enclosing the target object.
        hx1, hy1, hx2, hy2 : int
            Initial heading drag coordinates.
        """
        import numpy as np
        # Compute initial heading unit vector from drag direction
        drag_vec = np.array([hx2 - hx1, hy2 - hy1], dtype=np.float32)
        mag = np.linalg.norm(drag_vec)
        init_vec = drag_vec / mag if mag > 1e-6 else np.array([1.0, 0.0], dtype=np.float32)
        with self._state_lock:
            # Pre-assign the ID here so rapid successive box registrations each get a unique key.
            obj_id = self._next_object_id
            self._next_object_id += 1
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            self._click_headings[obj_id] = (cx, cy)
            self._heading_vecs[obj_id] = init_vec
            self._pending_boxes.append((obj_id, x1, y1, x2, y2))
            logger.info(
                f"Detector box target added: object_{obj_id} "
                f"box=({x1},{y1},{x2},{y2}) heading=(({hx1},{hy1})->({hx2},{hy2})) "
                f"init_vec=({init_vec[0]:.2f},{init_vec[1]:.2f})"
            )

    def get_click_heading(self, object_id: int) -> tuple[int, int] | None:
        """Return the original click (x, y) that registered *object_id*, or None."""
        with self._state_lock:
            return self._click_headings.get(object_id)

    def clear_tracked_objects(self) -> None:
        """Clears all tracked targets and their headings."""
        with self._state_lock:
            self._target_objects = []
            self._next_object_id = 0
            self._current_path = None
            self._click_headings = {}
            self._heading_vecs = {}
            self._pca_axes = {}
            self._last_centroids = {}
            self._last_sizes = {}
            self._pending_clicks = []
            self._pending_boxes = []
            self._tracking_active = False
            self._lost_flags = {}
            self._smoothed_bboxes = {}
            logger.info("Cleared all tracked objects, headings, and path.")

    def start_tracking(self) -> None:
        """Enables live tracking of the registered target objects."""
        import sys
        with self._state_lock:
            self._tracking_active = True
            logger.info("Live tracking confirmed and started.")
            for obj in self._target_objects:
                obj_id = obj["id"]
                class_name = obj.get("class_name", "object")
                logger.info(f"Object {obj_id} is recognized as '{class_name}'")
                sys.stdout.write(f"\robject is {class_name}\n")
                sys.stdout.flush()

    def save_targets(self, filepath: str = "targets.json") -> None:
        """Saves current tracked target object signatures to a JSON file."""
        import json
        with self._state_lock:
            serializable_targets = []
            for obj in self._target_objects:
                obj_id = obj["id"]
                token_list = obj["token"].tolist() if hasattr(obj["token"], "tolist") else list(obj["token"])
                initial_polygon = obj.get("initial_polygon")
                class_name = obj.get("class_name", "object")
                
                heading_vec = self._heading_vecs.get(obj_id)
                heading_list = heading_vec.tolist() if hasattr(heading_vec, "tolist") else (list(heading_vec) if heading_vec is not None else None)
                
                centroid = self._last_centroids.get(obj_id)
                size = self._last_sizes.get(obj_id)
                
                serializable_targets.append({
                    "id": obj_id,
                    "token": token_list,
                    "class_name": class_name,
                    "initial_polygon": initial_polygon,
                    "heading_vec": heading_list,
                    "centroid": centroid,
                    "size": size
                })
            
            try:
                with open(filepath, "w") as f:
                    json.dump(serializable_targets, f, indent=4)
                logger.info(f"Successfully saved {len(serializable_targets)} target signatures to '{filepath}'")
            except Exception as e:
                logger.error(f"Failed to save target signatures to '{filepath}': {e}")

    def load_targets(self, filepath: str = "targets.json") -> None:
        """Loads target object signatures from a JSON file and starts tracking."""
        import json
        import torch
        import numpy as np
        import sys
        
        try:
            with open(filepath, "r") as f:
                loaded_targets = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load target signatures from '{filepath}': {e}")
            return

        with self._state_lock:
            # Clear existing tracking states
            self._target_objects = []
            self._click_headings = {}
            self._heading_vecs = {}
            self._pca_axes = {}
            self._last_centroids = {}
            self._last_sizes = {}
            self._pending_clicks = []
            self._pending_boxes = []
            self._next_object_id = 0
            
            for target in loaded_targets:
                obj_id = target["id"]
                token_tensor = torch.tensor(target["token"], dtype=torch.float32)
                class_name = target.get("class_name", "object")
                initial_polygon = target.get("initial_polygon")
                
                self._target_objects.append({
                    "id": obj_id,
                    "token": token_tensor,
                    "class_name": class_name,
                    "initial_polygon": initial_polygon
                })
                
                if target.get("centroid") is not None:
                    self._last_centroids[obj_id] = tuple(target["centroid"])
                if target.get("size") is not None:
                    self._last_sizes[obj_id] = tuple(target["size"])
                if target.get("heading_vec") is not None:
                    self._heading_vecs[obj_id] = np.array(target["heading_vec"], dtype=np.float32)
                    
                    # Compute initial arrow tip / click heading
                    if self._last_centroids.get(obj_id) is not None and self._last_sizes.get(obj_id) is not None:
                        cx, cy = self._last_centroids[obj_id]
                        w, h = self._last_sizes[obj_id]
                        h_vec = self._heading_vecs[obj_id]
                        arrow_len = float(np.sqrt((w / 2) ** 2 + (h / 2) ** 2)) * 1.3
                        tip_x = int(cx + h_vec[0] * arrow_len)
                        tip_y = int(cy + h_vec[1] * arrow_len)
                        self._click_headings[obj_id] = (tip_x, tip_y)
                
                self._next_object_id = max(self._next_object_id, obj_id + 1)
            
            self._tracking_active = True
            logger.info(f"Loaded {len(self._target_objects)} target signatures from '{filepath}'. Tracking activated.")
            for obj in self._target_objects:
                obj_id = obj["id"]
                class_name = obj.get("class_name", "object")
                logger.info(f"Object {obj_id} is recognized as '{class_name}'")
                sys.stdout.write(f"\robject is {class_name}\n")
                sys.stdout.flush()

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
                with self._detections_updated:
                    self._detections_counter += 1
                    self._detections_updated.notify_all()

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
        is_dino = False
        if model_path:
            if model_path in ("YOLO+DINOv2", "YOLO+DINO", "SAM2+DINOv2"):
                is_dino = True
            elif model_path.endswith(".pt") and "sam" not in model_path.lower():
                is_dino = True

        if is_dino:
            import torch
            import torchvision.transforms.functional as TF
            from ultralytics import YOLO
            import numpy as np
            import cv2

            # Lazy load models
            if not hasattr(self, "_cached_dino_model"):
                logger.info("Loading DINOv2 model (dinov2_vits14)...")
                self._cached_dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
                self._cached_dino_model.eval()
                logger.info("DINOv2 model loaded successfully.")

            h_orig, w_orig = frame.shape[:2]

            with self._state_lock:
                clicks = list(self._pending_clicks)
                self._pending_clicks.clear()
                boxes = list(self._pending_boxes)
                self._pending_boxes.clear()

            def get_dino_token_for_box(x1, y1, x2, y2, p_tokens):
                x1_448 = x1 * (448.0 / w_orig)
                x2_448 = x2 * (448.0 / w_orig)
                y1_448 = y1 * (448.0 / h_orig)
                y2_448 = y2 * (448.0 / h_orig)

                col_start = int(x1_448 // 14)
                col_end = int(x2_448 // 14)
                row_start = int(y1_448 // 14)
                row_end = int(y2_448 // 14)

                col_start = max(0, min(31, col_start))
                col_end = max(0, min(31, col_end))
                row_start = max(0, min(31, row_start))
                row_end = max(0, min(31, row_end))

                patch_indices = []
                for r in range(row_start, row_end + 1):
                    for c in range(col_start, col_end + 1):
                        patch_indices.append(r * 32 + c)

                if len(patch_indices) > 0:
                    tokens_to_avg = p_tokens[0, patch_indices]
                    token = tokens_to_avg.mean(dim=0)
                else:
                    cx_448 = (x1_448 + x2_448) / 2.0
                    cy_448 = (y1_448 + y2_448) / 2.0
                    col = max(0, min(31, int(cx_448 // 14)))
                    row = max(0, min(31, int(cy_448 // 14)))
                    token = p_tokens[0, row * 32 + col]

                token = token / (token.norm(dim=-1, keepdim=True) + 1e-8)
                return token

            if len(clicks) > 0 or len(boxes) > 0 or len(self._target_objects) > 0:
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

                # Dynamically resolve yolo model path
                yolo_model_path = "yolov8n-seg.pt"
                if model_path and model_path.endswith(".pt") and "sam" not in model_path.lower():
                    yolo_model_path = model_path

                if not hasattr(self, "_cached_yolo_model") or getattr(self, "_cached_yolo_path", None) != yolo_model_path:
                    logger.info(f"Loading YOLO model ({yolo_model_path}) for DINOv2 visual tracking...")
                    self._cached_yolo_model = YOLO(yolo_model_path)
                    self._cached_yolo_path = yolo_model_path
                    logger.info("YOLO model loaded successfully.")

                yolo_results = self._cached_yolo_model(frame, verbose=False)[0]
                yolo_candidates = []
                has_masks = yolo_results.masks is not None

                for i, box in enumerate(yolo_results.boxes):
                    bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                    cls_idx = int(box.cls[0])
                    class_name = self._cached_yolo_model.names[cls_idx]
                    conf = float(box.conf[0])

                    polygon = None
                    if has_masks and i < len(yolo_results.masks.xy):
                        polygon = yolo_results.masks.xy[i]
                    if polygon is None or len(polygon) == 0:
                        polygon = np.array([[bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]], dtype=np.float32)

                    cand_token = get_dino_token_for_box(bx1, by1, bx2, by2, patch_tokens)

                    yolo_candidates.append({
                        "box": (bx1, by1, bx2, by2),
                        "class_name": class_name,
                        "confidence": conf,
                        "polygon": polygon,
                        "token": cand_token,
                        "cx": (bx1 + bx2) / 2.0,
                        "cy": (by1 + by2) / 2.0,
                        "w": bx2 - bx1,
                        "h": by2 - by1,
                    })

                # Process clicks — obj_id was pre-assigned in set_click_target
                for obj_id, click_x, click_y in clicks:
                    matched_cand = None
                    for cand in yolo_candidates:
                        bx1, by1, bx2, by2 = cand["box"]
                        if bx1 <= click_x <= bx2 and by1 <= click_y <= by2:
                            if matched_cand is None:
                                matched_cand = cand
                            else:
                                area_prev = (matched_cand["box"][2] - matched_cand["box"][0]) * (matched_cand["box"][3] - matched_cand["box"][1])
                                area_curr = (bx2 - bx1) * (by2 - by1)
                                if area_curr < area_prev:
                                    matched_cand = cand

                    if matched_cand is not None:
                        box_class = matched_cand["class_name"]
                        target_token = matched_cand["token"]
                        cx, cy = matched_cand["cx"], matched_cand["cy"]
                        w, h = matched_cand["w"], matched_cand["h"]
                        poly_list = matched_cand["polygon"].tolist() if hasattr(matched_cand["polygon"], "tolist") else matched_cand["polygon"]
                    else:
                        box_class = "object"
                        click_x_448 = click_x * (448.0 / w_orig)
                        click_y_448 = click_y * (448.0 / h_orig)
                        col = max(0, min(31, int(click_x_448 // 14)))
                        row = max(0, min(31, int(click_y_448 // 14)))
                        target_token = patch_tokens[0, row * 32 + col]
                        target_token = target_token / (target_token.norm(dim=-1, keepdim=True) + 1e-8)
                        cx, cy = float(click_x), float(click_y)
                        w, h = 40.0, 40.0
                        poly_list = [[click_x - 20, click_y - 20], [click_x + 20, click_y - 20], [click_x + 20, click_y + 20], [click_x - 20, click_y + 20]]

                    with self._state_lock:
                        self._target_objects.append({
                            "id": obj_id,
                            "token": target_token,
                            "class_name": box_class,
                            "initial_polygon": poly_list
                        })
                        self._last_centroids[obj_id] = (cx, cy)
                        self._last_sizes[obj_id] = (w, h)
                    logger.info(f"Target token initialized for object_{obj_id} ({box_class}) from click ({click_x}, {click_y})")

                # Process boxes — obj_id was pre-assigned in set_box_target
                for obj_id, x1, y1, x2, y2 in boxes:
                    snapped_cand = None
                    best_iou = 0.0

                    for cand in yolo_candidates:
                        bx1, by1, bx2, by2 = cand["box"]
                        ix1 = max(x1, bx1)
                        iy1 = max(y1, by1)
                        ix2 = min(x2, bx2)
                        iy2 = min(y2, by2)

                        iw = max(0, ix2 - ix1)
                        ih = max(0, iy2 - iy1)
                        inter_area = iw * ih

                        if inter_area > 0:
                            area_u = (x2 - x1) * (y2 - y1)
                            area_b = (bx2 - bx1) * (by2 - by1)
                            union_area = area_u + area_b - inter_area
                            iou = inter_area / (union_area + 1e-8)
                            if iou > best_iou:
                                best_iou = iou
                                snapped_cand = cand

                    if best_iou < 0.1 and len(yolo_candidates) > 0:
                        cx_u, cy_u = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                        best_dist = float("inf")
                        for cand in yolo_candidates:
                            dist = np.sqrt((cx_u - cand["cx"])**2 + (cy_u - cand["cy"])**2)
                            if dist < best_dist:
                                best_dist = dist
                                snapped_cand = cand

                        if best_dist > 150.0:
                            snapped_cand = None

                    if snapped_cand is not None:
                        box_class = snapped_cand["class_name"]
                        target_token = snapped_cand["token"]
                        cx, cy = snapped_cand["cx"], snapped_cand["cy"]
                        w, h = snapped_cand["w"], snapped_cand["h"]
                        poly_list = snapped_cand["polygon"].tolist() if hasattr(snapped_cand["polygon"], "tolist") else snapped_cand["polygon"]
                        logger.info(f"[Snapping] Snapped user box ({x1},{y1},{x2},{y2}) to YOLO object '{box_class}' at {snapped_cand['box']}")
                    else:
                        box_class = "object"
                        target_token = get_dino_token_for_box(x1, y1, x2, y2, patch_tokens)
                        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                        w, h = float(x2 - x1), float(y2 - y1)
                        poly_list = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                        logger.info(f"[Snapping] No matching YOLO object found near user box ({x1},{y1},{x2},{y2}). Using drawn box.")

                    with self._state_lock:
                        self._target_objects.append({
                            "id": obj_id,
                            "token": target_token,
                            "class_name": box_class,
                            "initial_polygon": poly_list
                        })
                        self._last_centroids[obj_id] = (cx, cy)
                        self._last_sizes[obj_id] = (w, h)
                    logger.info(f"Target token initialized for object_{obj_id} ({box_class}) from box")

                # Track all active target objects
                detections = []
                with self._state_lock:
                    tracking_active = self._tracking_active

                if tracking_active:
                    for obj in self._target_objects:
                        obj_id = obj["id"]
                        obj_token = obj["token"]
                        obj_class = obj.get("class_name", "object")

                        # ── Candidate selection: similarity-primary, proximity tiebreak ──
                        best_match = None
                        best_sim = -1.0
                        prev_centroid = self._last_centroids.get(obj_id)

                        for cand in yolo_candidates:
                            sim = float(torch.dot(obj_token, cand["token"]).item())
                            if best_match is None or sim > best_sim:
                                best_sim = sim
                                best_match = cand
                            elif prev_centroid is not None and (sim - best_sim) > -0.05:
                                # Scores within 0.05 — tiebreak by proximity to last centroid
                                prev_cx, prev_cy = prev_centroid
                                new_dist = np.hypot(cand["cx"] - prev_cx, cand["cy"] - prev_cy)
                                best_dist = np.hypot(best_match["cx"] - prev_cx, best_match["cy"] - prev_cy)
                                if new_dist < best_dist:
                                    best_sim = sim
                                    best_match = cand

                        if best_sim >= 0.55 and best_match is not None:
                            # ── Successful match ──────────────────────────────────────────
                            self._lost_flags[obj_id] = False

                            matched_cx = best_match["cx"]
                            matched_cy = best_match["cy"]
                            matched_w = best_match["w"]
                            matched_h = best_match["h"]
                            matched_polygon = best_match["polygon"]

                            prev_size = self._last_sizes.get(obj_id)
                            self._last_centroids[obj_id] = (matched_cx, matched_cy)
                            self._last_sizes[obj_id] = (matched_w, matched_h)

                            # EMA bbox smoothing — reduces per-frame jitter (alpha = 0.5)
                            if obj_id in self._smoothed_bboxes:
                                scx, scy, sw, sh = self._smoothed_bboxes[obj_id]
                                scx = 0.5 * matched_cx + 0.5 * scx
                                scy = 0.5 * matched_cy + 0.5 * scy
                                sw  = 0.5 * matched_w  + 0.5 * sw
                                sh  = 0.5 * matched_h  + 0.5 * sh
                            else:
                                scx, scy, sw, sh = matched_cx, matched_cy, matched_w, matched_h
                            self._smoothed_bboxes[obj_id] = (scx, scy, sw, sh)

                            # Slow token update on very confident matches to track appearance drift
                            if best_sim >= 0.80:
                                new_tok = 0.05 * best_match["token"] + 0.95 * obj_token
                                obj["token"] = new_tok / (float(new_tok.norm()) + 1e-8)

                            det = {
                                "x": int(scx - sw / 2),
                                "y": int(scy - sh / 2),
                                "w": int(sw),
                                "h": int(sh),
                                "cx": int(scx),
                                "cy": int(scy),
                                "class_name": f"{obj_class}_{obj_id}",
                                "confidence": float(best_sim),
                                "polygon": matched_polygon.tolist() if hasattr(matched_polygon, "tolist") else matched_polygon,
                                "lost": False,
                            }

                            # ── Heading tracking (unchanged logic, smoothed bbox for arrow length) ──
                            heading_vec = self._heading_vecs.get(obj_id)
                            if heading_vec is not None:
                                is_stationary = False
                                if prev_centroid is not None and prev_size is not None:
                                    prev_cx, prev_cy = prev_centroid
                                    prev_w, prev_h = prev_size
                                    centroid_dist = np.sqrt((matched_cx - prev_cx)**2 + (matched_cy - prev_cy)**2)
                                    w_diff = abs(matched_w - prev_w)
                                    h_diff = abs(matched_h - prev_h)
                                    if centroid_dist < 2.0 and w_diff < 2.0 and h_diff < 2.0:
                                        is_stationary = True

                                # Translation-based heading tracking fallback:
                                # If the object translated significantly, align heading vector with motion direction.
                                if prev_centroid is not None:
                                    prev_cx, prev_cy = prev_centroid
                                    dx_move = matched_cx - prev_cx
                                    dy_move = matched_cy - prev_cy
                                    move_dist = np.sqrt(dx_move**2 + dy_move**2)
                                    if move_dist > 3.0:
                                        move_vec = np.array([dx_move, dy_move], dtype=np.float32) / move_dist
                                        dot_prod = np.dot(move_vec, heading_vec)
                                        # Only update if the motion is collinear (forward/backward)
                                        if abs(dot_prod) > 0.5:
                                            target_heading = move_vec if dot_prod > 0 else -move_vec
                                            # Apply EMA smoothing to the heading vector
                                            beta = 0.25
                                            heading_vec = beta * target_heading + (1.0 - beta) * heading_vec
                                            heading_vec = heading_vec / (np.linalg.norm(heading_vec) + 1e-8)
                                            self._heading_vecs[obj_id] = heading_vec

                                rect = cv2.minAreaRect(matched_polygon.astype(np.float32))
                                box_pts = cv2.boxPoints(rect)
                                v1 = box_pts[1] - box_pts[0]
                                v2 = box_pts[2] - box_pts[1]
                                norm_v1 = np.linalg.norm(v1)
                                norm_v2 = np.linalg.norm(v2)

                                last_pca = self._pca_axes.get(obj_id)

                                if norm_v1 > 1e-6 and norm_v2 > 1e-6:
                                    v1 = v1 / norm_v1
                                    v2 = v2 / norm_v2
                                    pca_candidates = [v1, -v1, v2, -v2]

                                    if last_pca is None:
                                        new_pca = max(pca_candidates, key=lambda c: np.dot(c, heading_vec))
                                        if np.dot(new_pca, heading_vec) < 0:
                                            new_pca = -new_pca
                                    else:
                                        new_pca = max(pca_candidates, key=lambda c: np.dot(c, last_pca))
                                        if np.dot(new_pca, last_pca) < 0:
                                            new_pca = -new_pca
                                else:
                                    new_pca = heading_vec.copy()

                                if last_pca is None:
                                    self._pca_axes[obj_id] = new_pca
                                    smoothed_pca = new_pca
                                else:
                                    alpha = 0.15
                                    smoothed_pca = alpha * new_pca + (1.0 - alpha) * last_pca
                                    smoothed_pca = smoothed_pca / (np.linalg.norm(smoothed_pca) + 1e-8)

                                cos_a = float(np.clip(np.dot(last_pca if last_pca is not None else new_pca, smoothed_pca), -1.0, 1.0))
                                cross = float((last_pca[0] if last_pca is not None else new_pca[0]) * smoothed_pca[1] - (last_pca[1] if last_pca is not None else new_pca[1]) * smoothed_pca[0])
                                delta = np.arctan2(cross, cos_a)

                                if is_stationary:
                                    delta = 0.0

                                _DB = 0.0
                                _MAX = 0.524
                                if abs(delta) < _DB:
                                    delta = 0.0
                                elif abs(delta) > _MAX:
                                    delta = float(np.sign(delta)) * _MAX

                                if delta != 0.0:
                                    c_val, s_val = np.cos(delta), np.sin(delta)
                                    heading_vec = np.array([
                                        c_val * heading_vec[0] - s_val * heading_vec[1],
                                        s_val * heading_vec[0] + c_val * heading_vec[1],
                                    ], dtype=np.float32)
                                    self._heading_vecs[obj_id] = heading_vec

                                self._pca_axes[obj_id] = smoothed_pca

                                # Arrow length uses smoothed bbox so it doesn't flicker
                                arrow_len = float(np.sqrt((sw / 2) ** 2 + (sh / 2) ** 2)) * 1.3
                                tip_x = int(scx + heading_vec[0] * arrow_len)
                                tip_y = int(scy + heading_vec[1] * arrow_len)
                                det["heading"] = [tip_x, tip_y]
                                with self._state_lock:
                                    self._click_headings[obj_id] = (tip_x, tip_y)

                            logger.info(
                                f"[{det['class_name']}] matched sim={best_sim:.3f} "
                                f"centroid=({int(scx)},{int(scy)}) bbox={int(sw)}x{int(sh)}"
                            )
                            detections.append(det)

                        else:
                            # ── Lost: preserve spatial state, emit at last known position ──
                            self._lost_flags[obj_id] = True
                            logger.info(f"Object_{obj_id} sim={best_sim:.3f} below threshold — lost (last pos kept).")

                            scx, scy, sw, sh = self._smoothed_bboxes.get(
                                obj_id,
                                (*self._last_centroids.get(obj_id, (0.0, 0.0)),
                                 *self._last_sizes.get(obj_id, (40, 40)))
                            )

                            heading_vec = self._heading_vecs.get(obj_id)
                            heading_tip = None
                            if heading_vec is not None:
                                arrow_len = float(np.sqrt((sw / 2) ** 2 + (sh / 2) ** 2)) * 1.3
                                heading_tip = [
                                    int(scx + heading_vec[0] * arrow_len),
                                    int(scy + heading_vec[1] * arrow_len),
                                ]

                            det = {
                                "x": int(scx - sw / 2),
                                "y": int(scy - sh / 2),
                                "w": int(sw),
                                "h": int(sh),
                                "cx": int(scx),
                                "cy": int(scy),
                                "class_name": f"{obj_class}_{obj_id}",
                                "confidence": float(best_sim),
                                "lost": True,
                                "heading": heading_tip,
                            }
                            detections.append(det)

                else:
                    # ── Draft phase: live re-anchor each object per frame ─────────────────
                    for obj in self._target_objects:
                        obj_id = obj["id"]
                        obj_token = obj["token"]
                        obj_class = obj.get("class_name", "object")

                        # Re-match against current YOLO candidates so draft position stays live
                        best_cand = None
                        best_sim = -1.0
                        prev_centroid = self._last_centroids.get(obj_id)

                        for cand in yolo_candidates:
                            sim = float(torch.dot(obj_token, cand["token"]).item())
                            if best_cand is None or sim > best_sim:
                                best_sim = sim
                                best_cand = cand
                            elif prev_centroid is not None and (sim - best_sim) > -0.05:
                                prev_cx, prev_cy = prev_centroid
                                if (np.hypot(cand["cx"] - prev_cx, cand["cy"] - prev_cy) <
                                        np.hypot(best_cand["cx"] - prev_cx, best_cand["cy"] - prev_cy)):
                                    best_sim = sim
                                    best_cand = cand

                        if best_cand is not None and best_sim >= 0.50:
                            # Live position available — update stored state
                            cx, cy = best_cand["cx"], best_cand["cy"]
                            w, h = best_cand["w"], best_cand["h"]
                            self._last_centroids[obj_id] = (cx, cy)
                            self._last_sizes[obj_id] = (w, h)
                            poly = best_cand["polygon"]
                            obj["initial_polygon"] = poly.tolist() if hasattr(poly, "tolist") else poly
                        else:
                            # Object temporarily not visible — fall back to stored position
                            cx, cy = self._last_centroids.get(obj_id, (0.0, 0.0))
                            w, h = self._last_sizes.get(obj_id, (0, 0))

                        heading_vec = self._heading_vecs.get(obj_id)
                        heading_tip = None
                        if heading_vec is not None:
                            arrow_len = float(np.sqrt((w / 2) ** 2 + (h / 2) ** 2)) * 1.3
                            heading_tip = [
                                int(cx + heading_vec[0] * arrow_len),
                                int(cy + heading_vec[1] * arrow_len),
                            ]

                        det = {
                            "x": int(cx - w / 2),
                            "y": int(cy - h / 2),
                            "w": int(w),
                            "h": int(h),
                            "cx": int(cx),
                            "cy": int(cy),
                            "class_name": f"{obj_class}_{obj_id} (draft)",
                            "confidence": 1.0,
                            "polygon": obj.get("initial_polygon"),
                            "heading": heading_tip,
                        }
                        detections.append(det)

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
