"""
Commander — translates Command values into physical robot actions.

The backend is a plain callable with the signature::

    backend(command: Command) -> None

Swapping the transport layer (serial, ROS, TCP, …) requires only passing a
different backend to the constructor; nothing else in the package needs to change.

Built-in backends
-----------------
* ``None``  (default) — stub that logs and does nothing else.

Adding a real backend
---------------------
::

    import serial

    port = serial.Serial("/dev/ttyUSB0", 115200, timeout=1)

    def serial_backend(command):
        port.write((command.value + "\\n").encode())

    commander = Commander(backend=serial_backend)
"""

import logging
from .commands import Command

logger = logging.getLogger(__name__)


class Commander:
    """
    Sends ``Command`` values to the robot via a pluggable backend callable.

    Parameters
    ----------
    backend:
        Callable ``(command: Command) -> None`` that delivers the command to
        the physical robot.  Defaults to a stub that only logs.
    """

    def __init__(self, backend=None) -> None:
        self._backend = backend or self._stub_backend

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send(self, command: Command) -> bool:
        """
        Send *command* to the robot.

        Logs the dispatch at INFO level regardless of backend, then delegates
        to the backend callable.  Any exception raised by the backend is
        propagated to the caller so the Agent can handle it.

        Returns
        -------
        bool: True if the send succeeded, False otherwise.
        """
        logger.info(f"Sending command: {command.value!r}")
        res = self._backend(command)
        if res is False:
            return False
        return True

    # ------------------------------------------------------------------
    # Built-in backends
    # ------------------------------------------------------------------

    @staticmethod
    def _stub_backend(command: Command) -> None:
        """
        No-op backend.  Replace with a real transport when hardware is available.
        """
        logger.debug(f"[STUB] command '{command.value}' not sent to any hardware.")
