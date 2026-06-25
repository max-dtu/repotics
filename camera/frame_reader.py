import cv2
import threading
import queue
import time
import logging

logger = logging.getLogger(__name__)


class FrameReader:
    """
    Owns the camera hardware and the raw frame acquisition loop.

    Subsystems register interest by subscribing a queue.  The reader pushes
    every new frame to all subscribed queues.  Two fan-out modes are supported:

    * drop_stale=True  — stale frame is discarded before the new one is enqueued,
                         so the consumer always sees the latest (preview-style).
    * drop_stale=False — frames are appended; dropped only when the queue is full
                         (recording-style, maxsize=128).
    """

    def __init__(self, index: int = 0) -> None:
        self.index = index
        self.cap = None

        # Serializes lifecycle transitions (open / close)
        self._state_lock = threading.Lock()
        # Protects hot-path frame data
        self._frame_lock = threading.Lock()
        # Signaled whenever a new frame is stored
        self._frame_ready_event = threading.Event()

        self._running = False
        self._reader_thread = None
        self._latest_frame = None
        self._width = 0
        self._height = 0

        # Subscriber list: list of (queue.Queue, drop_stale: bool)
        self._subscribers_lock = threading.Lock()
        self._subscribers: list[tuple[queue.Queue, bool]] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Opens the camera device and starts the frame acquisition thread.  Idempotent."""
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

    def close(self) -> None:
        """Stops frame acquisition and releases the camera hardware."""
        logger.info("Shutting down FrameReader...")

        with self._state_lock:
            self._running = False
            reader_thread = self._reader_thread
            self._reader_thread = None

        if reader_thread is not None and reader_thread.is_alive():
            reader_thread.join(timeout=2.0)
            if reader_thread.is_alive():
                logger.warning("Camera reader thread did not exit within timeout.")

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

        logger.info("FrameReader shut down.")

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def get_frame(self, timeout: float = 5.0):
        """
        Returns a copy of the latest frame, blocking until one arrives.
        Raises RuntimeError on timeout.
        """
        deadline = time.monotonic() + timeout
        while True:
            with self._frame_lock:
                frame = self._latest_frame
            if frame is not None:
                return frame.copy()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("Timeout waiting for first camera frame.")
            self._frame_ready_event.wait(timeout=min(remaining, 0.05))
            self._frame_ready_event.clear()

    def get_dimensions(self, timeout: float = 5.0) -> tuple:
        """
        Returns (width, height), blocking until dimensions are known.
        Raises RuntimeError on timeout.
        """
        deadline = time.monotonic() + timeout
        while True:
            with self._frame_lock:
                w, h = self._width, self._height
            if w > 0 and h > 0:
                return w, h
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("Timeout waiting for camera frame dimensions.")
            self._frame_ready_event.wait(timeout=min(remaining, 0.05))
            self._frame_ready_event.clear()

    # ------------------------------------------------------------------
    # Subscriber fan-out
    # ------------------------------------------------------------------

    def subscribe(self, q: queue.Queue, *, drop_stale: bool = False) -> None:
        """Register *q* to receive every new frame."""
        with self._subscribers_lock:
            self._subscribers.append((q, drop_stale))

    def unsubscribe(self, q: queue.Queue) -> None:
        """Remove *q* from the subscriber list."""
        with self._subscribers_lock:
            self._subscribers = [(sq, ds) for sq, ds in self._subscribers if sq is not q]

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _reader_loop(self) -> None:
        """
        Single-producer frame acquisition loop.
        Reads frames from the webcam and fans them out to all subscribers.
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
                
                # Reopen camera if we have several consecutive failures to attempt hardware recovery
                if consecutive_failures % 10 == 0 and consecutive_failures < max_failures:
                    logger.info("Attempting to recover camera by reopening device...")
                    try:
                        cap.release()
                        new_cap = cv2.VideoCapture(self.index)
                        if new_cap.isOpened():
                            self.cap = new_cap
                            cap = new_cap
                            logger.info("Camera recovered and reopened successfully.")
                        else:
                            new_cap.release()
                            logger.warning("Failed to reopen camera during recovery.")
                    except Exception as exc:
                        logger.error(f"Error reopening camera during recovery: {exc}")

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

            # Fan-out to all subscribers
            with self._subscribers_lock:
                snapshot = list(self._subscribers)

            for q, drop_stale in snapshot:
                if drop_stale:
                    # Discard any stale frame so the consumer always gets the latest
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        pass
                try:
                    q.put_nowait(frame)
                except queue.Full:
                    if not drop_stale:
                        logger.warning("Subscriber queue full — frame dropped.")
                except Exception as e:
                    logger.error(f"Error queuing frame for subscriber: {e}")

        logger.info("Camera reader thread stopped.")
