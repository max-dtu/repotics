import cv2
import threading
import multiprocessing
import queue
import time
import logging
import sys

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Preview subprocess — must be a top-level picklable function for spawn context
# ---------------------------------------------------------------------------

def _run_preview_process(frame_queue, cmd_queue, exit_event):
    """
    Runs in a dedicated subprocess.

    Receives ``(frame, detections)`` tuples from the main process, draws
    bounding boxes, and displays via cv2.imshow.  Keypresses are forwarded
    back as command strings through cmd_queue.

    Keyboard map
    ------------
    r — "record"
    s — "stop_recording"
    c — "capture"
    d — "toggle_detect"
    q — set exit_event and exit
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] (PreviewProcess) %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )
    log = logging.getLogger("camera.preview")
    log.info("Preview process started.")

    _KEY_ACTIONS = {
        ord("r"): "record",
        ord("s"): "stop_recording",
        ord("c"): "capture",
        ord("d"): "toggle_detect",
        ord("x"): "clear_tracked",
        9: "cycle_detector",  # Tab key
    }

    def _get_class_color(class_name):
        import random
        state = random.getstate()
        random.seed(class_name)
        color = (random.randint(50, 240), random.randint(50, 240), random.randint(50, 240))
        random.setstate(state)
        return color

    def _handle_key(key):
        """Dispatch a keypress.  Returns True if the preview should exit."""
        if key == ord("q"):
            log.info("'q' pressed. Closing preview.")
            try:
                cmd_queue.put_nowait("quit")
            except Exception:
                pass
            exit_event.set()
            return True
        action = _KEY_ACTIONS.get(key)
        if action:
            log.info(f"'{chr(key)}' pressed — sending '{action}' command.")
            try:
                cmd_queue.put_nowait(action)
            except Exception as e:
                log.warning(f"Could not enqueue command '{action}': {e}")
        return False

    window_name = "Camera Preview [r=record  s=stop  c=capture  d=detect  x=clear  click=track  q=quit]"
    error_count = 0
    max_errors = 10

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                log.info(f"Mouse clicked at x={x}, y={y} — sending click command.")
                try:
                    cmd_queue.put_nowait(f"click:{x},{y}")
                except Exception as e:
                    log.warning(f"Could not enqueue click command: {e}")

        cv2.setMouseCallback(window_name, mouse_callback)

        while not exit_event.is_set():
            try:
                payload = frame_queue.get(timeout=0.1)
            except queue.Empty:
                # No new frame — still pump the GUI event loop
                key = cv2.waitKey(1) & 0xFF
                if _handle_key(key):
                    break
                continue
            except (EOFError, ConnectionError, KeyboardInterrupt):
                log.info("IPC channel closed. Exiting.")
                break

            if payload is None:
                log.info("Received stop sentinel. Exiting.")
                break

            frame, detections = payload

            # Draw masks (polygons) on the overlay first
            overlay = frame.copy()
            has_masks = False
            for det in detections:
                if det.get("class_name") == "path":
                    continue
                polygon = det.get("polygon")
                if polygon:
                    has_masks = True
                    import numpy as np
                    pts = np.array(polygon, dtype=np.int32)
                    color = _get_class_color(det.get("class_name", "?"))
                    cv2.fillPoly(overlay, [pts], color)

            # Blend mask overlay into the frame if any masks were drawn
            if has_masks:
                cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

            # Draw bounding boxes and text labels on top (opaque)
            for det in detections:
                if det.get("class_name") == "path":
                    continue
                x = det.get("x", 0)
                y = det.get("y", 0)
                w = det.get("w", 0)
                h = det.get("h", 0)
                class_name = det.get("class_name", "?")
                confidence = det.get("confidence", 0.0)
                
                color = _get_class_color(class_name)
                label = f"{class_name} {confidence:.2f}"
                
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # Draw a nice background box for text label to make it readable
                (lbl_w, lbl_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                lbl_y_offset = max(y, lbl_h + baseline + 4)
                cv2.rectangle(frame, (x, lbl_y_offset - lbl_h - baseline - 2), (x + lbl_w, lbl_y_offset), color, cv2.FILLED)
                
                # Draw white text on class color background
                cv2.putText(
                    frame,
                    label,
                    (x, lbl_y_offset - baseline - 1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            # Draw path waypoints if present
            path_det = next((d for d in detections if d.get("class_name") == "path"), None)
            if path_det and path_det.get("points"):
                import numpy as np
                pts = np.array(path_det["points"], dtype=np.int32)
                cv2.polylines(frame, [pts], isClosed=False, color=(0, 255, 0), thickness=3)
                for pt in path_det["points"]:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)

            try:
                cv2.imshow(window_name, frame)
                error_count = 0
            except Exception as e:
                error_count += 1
                log.error(f"Failed to display frame ({error_count}/{max_errors}): {e}")
                if error_count >= max_errors:
                    log.critical("Too many display errors. Exiting.")
                    break

            try:
                key = cv2.waitKey(1) & 0xFF
                if _handle_key(key):
                    break
            except Exception as e:
                log.error(f"waitKey error: {e}")
                break

    except Exception:
        log.exception("Unexpected exception in preview loop.")
    finally:
        log.info("Destroying preview windows.")
        try:
            cv2.destroyAllWindows()
            for _ in range(4):
                cv2.waitKey(1)
        except Exception as e:
            log.error(f"Error during window cleanup: {e}")
        log.info("Preview process terminated.")


# ---------------------------------------------------------------------------
# PreviewManager — main-process side of the preview subsystem
# ---------------------------------------------------------------------------

class PreviewManager:
    """
    Manages the preview subprocess and its two IPC channels.

    Architecture
    ------------
    * A **feeder thread** subscribes a queue to FrameReader, combines each
      frame with the latest detections from Detector, and pushes
      ``(frame, detections)`` tuples into the subprocess via a
      multiprocessing.Queue.
    * A **cmd-dispatcher thread** reads command strings from the subprocess
      and drives VideoRecorder and Detector accordingly.

    This design keeps FrameReader and Detector completely unaware of the
    preview subprocess.
    """

    def __init__(self, reader, recorder, detector) -> None:
        self._reader = reader
        self._recorder = recorder
        self._detector = detector

        self._state_lock = threading.Lock()
        self._previewing = False

        self._preview_process: multiprocessing.Process | None = None
        self._preview_queue = None          # multiprocessing.Queue → subprocess
        self._preview_stop_event = None     # multiprocessing.Event
        self._preview_cmd_queue = None      # multiprocessing.Queue ← subprocess
        self._cmd_thread: threading.Thread | None = None
        self._feeder_queue: queue.Queue | None = None   # in-process subscriber queue
        self._feeder_thread: threading.Thread | None = None

    @property
    def is_previewing(self) -> bool:
        return self._previewing

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, block: bool = False) -> None:
        """
        Spawns the preview subprocess.
        By default, runs in the background (non-blocking).
        Pass block=True to block the REPL until the window is closed.
        """
        with self._state_lock:
            if self._previewing:
                logger.info("Preview is already active.")
                return

            logger.info("Spawning preview process...")
            ctx = multiprocessing.get_context("spawn")
            preview_queue = ctx.Queue(maxsize=1)
            cmd_queue = ctx.Queue()
            stop_event = ctx.Event()

            proc = ctx.Process(
                target=_run_preview_process,
                args=(preview_queue, cmd_queue, stop_event),
                name=f"CameraPreviewProcess-{self._reader.index}",
                daemon=True,
            )
            try:
                proc.start()
            except Exception as e:
                raise RuntimeError(f"Failed to spawn preview process: {e}") from e

            # Feeder: in-process queue subscribed to FrameReader
            feeder_queue: queue.Queue = queue.Queue(maxsize=1)
            self._reader.subscribe(feeder_queue, drop_stale=True)

            feeder_thread = threading.Thread(
                target=self._feeder_loop,
                args=(feeder_queue, preview_queue, stop_event),
                name=f"PreviewFeederThread-{self._reader.index}",
                daemon=True,
            )
            feeder_thread.start()

            cmd_thread = threading.Thread(
                target=self._cmd_dispatcher_loop,
                args=(cmd_queue,),
                name=f"CameraCmdDispatcher-{self._reader.index}",
                daemon=True,
            )
            cmd_thread.start()

            self._preview_queue = preview_queue
            self._preview_cmd_queue = cmd_queue
            self._preview_stop_event = stop_event
            self._preview_process = proc
            self._cmd_thread = cmd_thread
            self._feeder_queue = feeder_queue
            self._feeder_thread = feeder_thread
            self._previewing = True
            logger.info("Preview window open. Press q to close.")

        if block:
            # Block the REPL until the preview window is closed (q pressed)
            proc.join()

    def stop(self) -> None:
        """Stops the preview subprocess and waits for it to exit."""
        with self._state_lock:
            if not self._previewing:
                return

            logger.info("Stopping preview...")
            self._previewing = False

            if self._preview_stop_event is not None:
                try:
                    self._preview_stop_event.set()
                except Exception as e:
                    logger.warning(f"Error signalling preview stop event: {e}")

            # Send None sentinels so threads and subprocess wake up immediately
            if self._preview_queue is not None:
                try:
                    self._preview_queue.put_nowait(None)
                except Exception:
                    pass

            if self._preview_cmd_queue is not None:
                try:
                    self._preview_cmd_queue.put_nowait(None)
                except Exception:
                    pass

            if self._feeder_queue is not None:
                self._reader.unsubscribe(self._feeder_queue)
                try:
                    self._feeder_queue.put_nowait(None)
                except Exception:
                    pass

            proc = self._preview_process
            cmd_thread = self._cmd_thread
            feeder_thread = self._feeder_thread
            self._preview_process = None
            self._preview_queue = None
            self._preview_stop_event = None
            self._preview_cmd_queue = None
            self._cmd_thread = None
            self._feeder_queue = None
            self._feeder_thread = None

        if proc is not None and proc.is_alive():
            proc.join(timeout=2.0)
            if proc.is_alive():
                logger.warning("Preview process did not exit in time. Terminating.")
                try:
                    proc.terminate()
                    proc.join(timeout=1.0)
                except Exception as e:
                    logger.error(f"Error terminating preview process: {e}")

        # Skip joining a thread if we are currently running inside it
        current = threading.current_thread()
        if cmd_thread is not None and cmd_thread.is_alive() and current is not cmd_thread:
            cmd_thread.join(timeout=1.0)

        if feeder_thread is not None and feeder_thread.is_alive() and current is not feeder_thread:
            feeder_thread.join(timeout=1.0)

        logger.info("Preview stopped.")

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _feeder_loop(
        self,
        feeder_queue: queue.Queue,
        preview_queue,
        stop_event,
    ) -> None:
        """
        Reads frames from the FrameReader subscriber queue, attaches the latest
        detections, and forwards ``(frame, detections)`` to the subprocess queue.
        Keeps only the latest frame in the single-slot subprocess queue.
        """
        logger.info("Preview feeder thread active.")
        try:
            while not stop_event.is_set():
                try:
                    frame = feeder_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if frame is None:
                    break

                detections = self._detector.get_detections()
                payload = (frame, detections)

                # Discard stale payload so the subprocess always gets the latest
                try:
                    preview_queue.get_nowait()
                except Exception:
                    pass
                try:
                    preview_queue.put_nowait(payload)
                except Exception:
                    pass

        except Exception:
            logger.exception("Unexpected exception in preview feeder thread.")
        finally:
            logger.info("Preview feeder thread stopped.")

    def _cmd_dispatcher_loop(self, cmd_queue) -> None:
        """
        Reads command strings sent by the preview subprocess and executes the
        corresponding action in the main process.

        Commands
        --------
        "record"          recorder.start(<timestamp>.mp4)
        "stop_recording"  recorder.stop()
        "capture"         reader.get_frame() + cv2.imwrite
        "toggle_detect"   detector.start() or detector.stop()
        None              stop sentinel — exits the loop
        """
        logger.info("Command dispatcher thread active.")
        while True:
            try:
                cmd = cmd_queue.get(timeout=0.2)
            except Exception:
                # queue.Empty or pipe closed — check whether preview is still running
                if not self._previewing:
                    break
                continue

            if cmd is None:
                logger.info("Command dispatcher received stop sentinel.")
                break

            logger.info(f"Dispatching preview command: '{cmd}'")
            try:
                if cmd.startswith("click:"):
                    try:
                        _, coords = cmd.split(":", 1)
                        x_str, y_str = coords.split(",", 1)
                        x, y = int(x_str), int(y_str)
                        if hasattr(self._detector, "set_click_target"):
                            self._detector.set_click_target(x, y)
                    except Exception as e:
                        logger.error(f"Failed to parse click command '{cmd}': {e}")

                elif cmd == "clear_tracked":
                    if hasattr(self._detector, "clear_tracked_objects"):
                        self._detector.clear_tracked_objects()

                elif cmd == "record":
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    path = f"recording_{ts}.mp4"
                    self._recorder.start(path)
                    logger.info(f"Recording started → {path}")

                elif cmd == "stop_recording":
                    self._recorder.stop()

                elif cmd == "capture":
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    path = f"capture_{ts}.jpg"
                    frame = self._reader.get_frame()
                    ok = cv2.imwrite(path, frame)
                    if not ok:
                        raise RuntimeError(f"cv2.imwrite returned False for path: '{path}'")
                    logger.info(f"Frame captured → {path}")

                elif cmd == "toggle_detect":
                    if self._detector.is_running:
                        self._detector.stop()
                        logger.info("Detection stopped.")
                    else:
                        self._detector.start(self._detector.model_path, confidence=self._detector.confidence, classes=self._detector.classes, imgsz=self._detector.imgsz)
                        logger.info("Detection started.")

                elif cmd == "cycle_detector":
                    if self._detector.is_running:
                        MODELS = [
                            "yolov8n.pt",
                            "yolov8n-seg.pt",
                            "yolov8x.pt",
                            "yolov8x-seg.pt",
                            "mobile_sam.pt",
                            "yolov8n.pt+mobile_sam.pt"
                        ]
                        current_path = self._detector.model_path or "yolov8n.pt"
                        
                        try:
                            idx = MODELS.index(current_path)
                        except ValueError:
                            idx = 0
                            
                        next_idx = (idx + 1) % len(MODELS)
                        new_model = MODELS[next_idx]
                        logger.info(f"Cycling detector to: {new_model}")
                        
                        # Preserve other options
                        conf = self._detector.confidence
                        cls = self._detector.classes
                        imgsz = self._detector.imgsz
                        
                        self._detector.stop()
                        self._detector.start(new_model, confidence=conf, classes=cls, imgsz=imgsz)
                    else:
                        logger.info("Ignoring detector cycle: detection is not running (press 'd' to enable).")

                elif cmd == "quit":
                    logger.info("Preview subprocess signaled exit. Stopping preview subsystems.")
                    threading.Thread(target=self.stop, name="PreviewStopThread", daemon=True).start()

                else:
                    logger.warning(f"Unknown command from preview process: '{cmd}'")

            except Exception as e:
                logger.error(f"Error executing preview command '{cmd}': {e}")

        logger.info("Command dispatcher thread stopped.")

        logger.info("Command dispatcher thread stopped.")
