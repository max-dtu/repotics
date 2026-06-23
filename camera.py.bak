import cv2
import threading
import multiprocessing
import queue
import time
import logging
import sys

logger = logging.getLogger(__name__)



def _run_preview_process(frame_queue, cmd_queue, exit_event):

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
    }

    def _handle_key(key):
        """Dispatch a keypress. Returns True if the preview should exit."""
        if key == ord("q"):
            log.info("'q' pressed. Closing preview.")
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

    window_name = "Camera Preview [r=record  s=stop  c=capture  q=quit]"
    error_count = 0
    max_errors = 10

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        while not exit_event.is_set():
            try:
                frame = frame_queue.get(timeout=0.1)
            except queue.Empty:
                # No new frame — still pump the GUI event loop
                key = cv2.waitKey(1) & 0xFF
                if _handle_key(key):
                    break
                continue
            except (EOFError, ConnectionError, KeyboardInterrupt):
                log.info("IPC channel closed. Exiting.")
                break

            if frame is None:
                log.info("Received stop sentinel. Exiting.")
                break

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


class Camera:
    def __init__(self, index=0):
        self.index = index
        self.cap = None

        # Serializes infrequent lifecycle transitions (open, close, preview, record)
        self._state_lock = threading.Lock()
        # Protects hot-path access to the latest frame and its dimensions
        self._frame_lock = threading.Lock()
        # Signaled whenever a new frame is stored
        self._frame_ready_event = threading.Event()

        # Acquisition state
        self._running = False
        self._reader_thread = None
        self._latest_frame = None
        self._width = 0
        self._height = 0

        # Recording state
        self._recording = False
        self._writer = None
        self._recording_queue = None
        self._recording_thread = None
        self._recording_stop_event = None

        # Preview state
        self._previewing = False
        self._preview_process = None
        self._preview_queue = None
        self._preview_stop_event = None
        self._preview_cmd_queue = None
        self._preview_cmd_thread = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def open(self):
        """
        Opens the camera device and starts the frame acquisition thread.
        Idempotent — safe to call multiple times.
        """
        with self._state_lock:
            if self._running and self.cap is not None:
                return

            logger.info(f"Opening camera device at index {self.index}...")
            try:
                cap = cv2.VideoCapture(self.index)
                if not cap.isOpened():
                    cap.release()
                    raise RuntimeError(f"Failed to open camera at index {self.index}.")
            except Exception as e:
                raise RuntimeError(f"Error accessing camera device: {e}") from e

            self.cap = cap

            try:
                self._width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            except Exception as e:
                logger.warning(f"Could not read camera dimensions (will resolve from first frame): {e}")
                self._width = 0
                self._height = 0

            self._running = True
            self._reader_thread = threading.Thread(
                target=self._reader_loop,
                name=f"CameraReaderThread-{self.index}",
                daemon=True,
            )
            self._reader_thread.start()
            logger.info("Camera reader thread started.")

    def get_dimensions(self):
        """Returns the current (width, height) of captured frames, thread-safely."""
        with self._frame_lock:
            return self._width, self._height

    def capture(self, path=None, timeout=5.0):
        """
        Returns the latest camera frame as a NumPy array (BGR).
        Blocks until the first frame arrives or `timeout` seconds elapse.
        Optionally saves the frame to `path`.
        """
        self.open()
        frame = self._wait_for_frame(timeout)

        frame_copy = frame.copy()

        if path:
            logger.info(f"Saving frame to: {path}")
            try:
                ok = cv2.imwrite(path, frame_copy)
                if not ok:
                    raise RuntimeError(f"cv2.imwrite returned False for path: '{path}'")
            except Exception as e:
                raise RuntimeError(f"Failed to save image to '{path}': {e}") from e
            logger.info(f"Frame saved to {path}.")

        return frame_copy

    def preview(self):
        """
        Opens a live preview window and BLOCKS until the window is closed.

        The REPL is intentionally paused while the window is open — all camera
        control happens via keyboard shortcuts in the window itself:
            r — start recording  (auto-generates a timestamped .mp4 filename)
            s — stop recording
            c — capture frame    (auto-generates a timestamped .jpg filename)
            q — close preview and shut down the camera

        Returns once the camera has been fully shut down.
        """
        self.open()

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
                name=f"CameraPreviewProcess-{self.index}",
                daemon=True,
            )
            try:
                proc.start()
            except Exception as e:
                raise RuntimeError(f"Failed to spawn preview process: {e}") from e

            cmd_thread = threading.Thread(
                target=self._cmd_dispatcher_loop,
                args=(cmd_queue,),
                name=f"CameraCmdDispatcher-{self.index}",
                daemon=True,
            )
            cmd_thread.start()

            self._preview_queue = preview_queue
            self._preview_cmd_queue = cmd_queue
            self._preview_stop_event = stop_event
            self._preview_process = proc
            self._preview_cmd_thread = cmd_thread
            self._previewing = True
            logger.info("Preview window open. Press q to close.")

        # Block the REPL until the preview window is closed (q pressed)
        proc.join()

        # The preview process has exited — perform full camera shutdown
        self.close()

    def stop_preview(self):
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

            # Send a None sentinel to each queue so the preview process and the
            # cmd-dispatcher thread wake up immediately without waiting for the
            # next timeout, complementing the exit_event signal above.
            if self._preview_queue is not None:
                try:
                    self._preview_queue.put_nowait(None)  # frame-queue stop sentinel
                except Exception:
                    pass

            if self._preview_cmd_queue is not None:
                try:
                    self._preview_cmd_queue.put_nowait(None)  # cmd-queue stop sentinel
                except Exception:
                    pass

            proc = self._preview_process
            cmd_thread = self._preview_cmd_thread
            self._preview_process = None
            self._preview_queue = None
            self._preview_stop_event = None
            self._preview_cmd_queue = None
            self._preview_cmd_thread = None

        # Join outside the lock so other operations are not blocked
        if proc is not None and proc.is_alive():
            proc.join(timeout=2.0)
            if proc.is_alive():
                logger.warning("Preview process did not exit in time. Terminating.")
                try:
                    proc.terminate()
                    proc.join(timeout=1.0)
                except Exception as e:
                    logger.error(f"Error terminating preview process: {e}")

        # Skip joining the cmd_thread if this call originates from within it
        # (e.g. the dispatcher calling close() after its loop exits).
        if cmd_thread is not None and cmd_thread.is_alive():
            if threading.current_thread() is not cmd_thread:
                cmd_thread.join(timeout=1.0)

        logger.info("Preview stopped.")

    def record(self, path, fps=30):
        """
        Starts recording camera frames to a video file asynchronously.
        Raises RuntimeError if recording is already active.
        """
        self.open()

        width, height = self._wait_for_dimensions(timeout=5.0)

        with self._state_lock:
            if self._recording:
                raise RuntimeError("Recording is already active.")

            writer = self._open_video_writer(path, fps, width, height)

            stop_event = threading.Event()
            rec_queue = queue.Queue(maxsize=128)

            self._writer = writer
            self._recording_queue = rec_queue
            self._recording_stop_event = stop_event
            self._recording = True

            self._recording_thread = threading.Thread(
                target=self._recording_loop,
                args=(writer, rec_queue, stop_event),
                name=f"CameraRecordingThread-{self.index}",
                daemon=True,
            )
            self._recording_thread.start()
            logger.info("Recording thread started.")

    def stop_recording(self):
        """Stops recording, drains remaining frames, and finalizes the video file."""
        with self._state_lock:
            if not self._recording:
                return

            logger.info("Stopping recording...")
            self._recording = False

            if self._recording_stop_event is not None:
                self._recording_stop_event.set()

            if self._recording_queue is not None:
                try:
                    self._recording_queue.put_nowait(None)  # Stop sentinel
                except Exception:
                    pass

            rec_thread = self._recording_thread
            self._recording_thread = None
            self._recording_queue = None
            self._recording_stop_event = None
            self._writer = None

        # Join outside the lock so other operations are not blocked
        if rec_thread is not None and rec_thread.is_alive():
            logger.info("Waiting for recording thread to drain...")
            rec_thread.join(timeout=5.0)
            if rec_thread.is_alive():
                logger.warning("Recording thread did not stop within timeout.")

        logger.info("Recording stopped.")

    def close(self):
        """
        Shuts down in a safe, deterministic order:
        1. Stop frame acquisition (reader thread).
        2. Stop and drain the recording pipeline.
        3. Stop the preview subprocess.
        4. Release the camera hardware.
        """
        logger.info("Shutting down Camera...")

        # 1. Stop acquisition
        with self._state_lock:
            self._running = False
            reader_thread = self._reader_thread
            self._reader_thread = None

        if reader_thread is not None and reader_thread.is_alive():
            reader_thread.join(timeout=2.0)
            if reader_thread.is_alive():
                logger.warning("Camera reader thread did not exit within timeout.")

        # 2. Stop recording (drains remaining frames before releasing writer)
        self.stop_recording()

        # 3. Stop preview
        self.stop_preview()

        # 4. Release hardware (safe now that the reader thread has exited)
        with self._state_lock:
            if self.cap is not None:
                logger.info("Releasing camera hardware...")
                try:
                    self.cap.release()
                except Exception as e:
                    logger.error(f"Error releasing camera hardware: {e}")
                self.cap = None

            with self._frame_lock:
                self._latest_frame = None
                self._width = 0
                self._height = 0

        logger.info("Camera shutdown complete.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _cmd_dispatcher_loop(self, cmd_queue):
        """
        Runs in a daemon thread in the main process.
        Reads command strings from the preview process and executes the
        corresponding Camera action with auto-generated filenames.

        Commands:
            "record"        — camera.record("<timestamp>.mp4")
            "stop_recording"— camera.stop_recording()
            "capture"       — camera.capture("<timestamp>.jpg")
            None            — stop sentinel; exits the loop
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
                if cmd == "record":
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    path = f"recording_{ts}.mp4"
                    self.record(path)
                    logger.info(f"Recording started → {path}")
                elif cmd == "stop_recording":
                    self.stop_recording()
                elif cmd == "capture":
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    path = f"capture_{ts}.jpg"
                    self.capture(path)
                    logger.info(f"Frame captured → {path}")
                else:
                    logger.warning(f"Unknown command from preview process: '{cmd}'")
            except Exception as e:
                logger.error(f"Error executing preview command '{cmd}': {e}")

        logger.info("Command dispatcher thread stopped.")

    def _wait_for_frame(self, timeout):
        """
        Blocks until a frame is available or timeout elapses.
        Polls _frame_ready_event in short slices to avoid busy-waiting while
        remaining responsive to the deadline.
        """
        deadline = time.monotonic() + timeout
        while True:
            with self._frame_lock:
                frame = self._latest_frame
            if frame is not None:
                return frame
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("Timeout waiting for first camera frame.")
            self._frame_ready_event.wait(timeout=min(remaining, 0.05))
            self._frame_ready_event.clear()

    def _wait_for_dimensions(self, timeout):
        """
        Blocks until frame dimensions are known (> 0) or timeout elapses.
        Dimensions may be populated from cap.get() before the first frame, or
        from the first frame itself; either way _frame_ready_event wakes us.
        """
        deadline = time.monotonic() + timeout
        while True:
            width, height = self.get_dimensions()
            if width > 0 and height > 0:
                return width, height
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("Timeout waiting for camera frame dimensions.")
            self._frame_ready_event.wait(timeout=min(remaining, 0.05))
            self._frame_ready_event.clear()

    def _open_video_writer(self, path, fps, width, height):
        """
        Tries a sequence of codecs and returns the first VideoWriter that opens.
        Raises RuntimeError if none succeed.
        """
        codecs = ["mp4v", "avc1", "MJPG", "XVID"]
        for codec in codecs:
            logger.info(f"Trying VideoWriter codec '{codec}' ({width}x{height} @ {fps} fps)...")
            try:
                writer = cv2.VideoWriter(
                    path,
                    cv2.VideoWriter_fourcc(*codec),
                    fps,
                    (width, height),
                )
                if writer.isOpened():
                    logger.info(f"VideoWriter opened with codec '{codec}'.")
                    return writer
                writer.release()
            except Exception as e:
                logger.warning(f"Codec '{codec}' failed: {e}")

        raise RuntimeError(
            f"Could not open VideoWriter for '{path}' ({width}x{height}) "
            f"with any of the codecs: {codecs}. "
            "Verify the output directory exists and is writable."
        )

    def _reader_loop(self):
        """
        Single-producer frame acquisition loop.
        Reads frames from the webcam and distributes them to queues.
        Never holds _state_lock during frame retrieval.
        """
        logger.info("Camera reader thread active.")
        consecutive_failures = 0
        max_failures = 30

        while self._running:
            cap = self.cap
            if cap is None:
                time.sleep(0.01)
                continue

            try:
                ok, frame = cap.read()
            except Exception as e:
                logger.error(f"Exception reading camera frame: {e}")
                ok, frame = False, None

            if not ok or frame is None:
                consecutive_failures += 1
                logger.warning(f"Frame read failure ({consecutive_failures}/{max_failures}).")
                if consecutive_failures >= max_failures:
                    logger.critical("Max consecutive read failures reached. Stopping reader.")
                    self._running = False
                    break
                time.sleep(0.03)
                continue

            consecutive_failures = 0

            with self._frame_lock:
                self._latest_frame = frame
                if self._width == 0 or self._height == 0:
                    self._height, self._width = frame.shape[:2]

            self._frame_ready_event.set()

            # Fan-out to recording queue (non-blocking; drop if full)
            rec_queue = self._recording_queue
            if rec_queue is not None:
                try:
                    rec_queue.put_nowait(frame)
                except queue.Full:
                    logger.warning("Recording queue full — frame dropped.")
                except Exception as e:
                    logger.error(f"Error queuing frame for recording: {e}")

            # Fan-out to preview queue (non-blocking; keep only the latest frame)
            prev_queue = self._preview_queue
            if prev_queue is not None:
                # Discard any stale frame already in the single-slot queue
                try:
                    prev_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    prev_queue.put_nowait(frame)
                except queue.Full:
                    pass  # Preview process is busy; skip this frame
                except Exception as e:
                    logger.error(f"Error queuing frame for preview: {e}")

        logger.info("Camera reader thread stopped.")

    def _recording_loop(self, writer, q, stop_event):
        """
        Background recording pipeline.
        Drains frames from the queue and writes them to the VideoWriter.
        No locks are held during I/O.
        """
        logger.info("Recording thread active.")
        try:
            while not stop_event.is_set() or not q.empty():
                try:
                    frame = q.get(timeout=0.1)
                except queue.Empty:
                    continue

                if frame is None:
                    logger.info("Received recording stop sentinel.")
                    break

                try:
                    writer.write(frame)
                except Exception as e:
                    logger.error(f"Error writing frame to video: {e}")
                    break
        except Exception:
            logger.exception("Unexpected exception in recording thread.")
        finally:
            logger.info("Releasing VideoWriter...")
            try:
                writer.release()
                logger.info("VideoWriter released.")
            except Exception as e:
                logger.error(f"Error releasing VideoWriter: {e}")