import cv2
import threading
import queue
import logging

from .frame_reader import FrameReader

logger = logging.getLogger(__name__)


class VideoRecorder:
    """
    Owns the recording pipeline: subscribes to FrameReader, drains frames
    through a bounded queue, and writes them to a VideoWriter in a daemon thread.
    """

    def __init__(self, reader: FrameReader) -> None:
        self._reader = reader
        self._state_lock = threading.Lock()

        self._recording = False
        self._writer = None
        self._recording_queue: queue.Queue | None = None
        self._recording_thread: threading.Thread | None = None
        self._recording_stop_event: threading.Event | None = None

    @property
    def is_recording(self) -> bool:
        return self._recording

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, path: str, fps: int = 30) -> None:
        """
        Starts recording to *path*.
        Blocks briefly to obtain frame dimensions, then returns immediately.
        Raises RuntimeError if recording is already active.
        """
        width, height = self._reader.get_dimensions(timeout=5.0)

        with self._state_lock:
            if self._recording:
                raise RuntimeError("Recording is already active.")

            writer = self._open_video_writer(path, fps, width, height)
            stop_event = threading.Event()
            rec_queue: queue.Queue = queue.Queue(maxsize=128)

            self._writer = writer
            self._recording_queue = rec_queue
            self._recording_stop_event = stop_event
            self._recording = True

            # Register with the reader *before* starting the drain thread so no
            # frames are missed between the two operations.
            self._reader.subscribe(rec_queue, drop_stale=False)

            self._recording_thread = threading.Thread(
                target=self._recording_loop,
                args=(writer, rec_queue, stop_event),
                name="CameraRecordingThread",
                daemon=True,
            )
            self._recording_thread.start()
            logger.info("Recording thread started.")

    def stop(self) -> None:
        """Stops recording, drains remaining frames, and finalizes the video file."""
        with self._state_lock:
            if not self._recording:
                return

            logger.info("Stopping recording...")
            self._recording = False

            if self._recording_stop_event is not None:
                self._recording_stop_event.set()

            rec_queue = self._recording_queue
            if rec_queue is not None:
                # Unsubscribe first so no new frames arrive after the sentinel
                self._reader.unsubscribe(rec_queue)
                try:
                    rec_queue.put_nowait(None)  # stop sentinel
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

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _open_video_writer(self, path: str, fps: int, width: int, height: int):
        """Tries a sequence of codecs and returns the first VideoWriter that opens."""
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

    def _recording_loop(
        self,
        writer,
        q: queue.Queue,
        stop_event: threading.Event,
    ) -> None:
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
