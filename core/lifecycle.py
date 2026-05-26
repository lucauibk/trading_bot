"""
Singleton-Lock and signal handling – extracted from main.py.
"""

import logging
import os
import signal
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_PIDFILE = Path(".bot.pid")


def acquire_singleton() -> None:
    """Ensure only one bot instance runs. Exit if another is already running."""
    if _PIDFILE.exists():
        try:
            existing = int(_PIDFILE.read_text().strip())
            if existing != os.getpid():
                os.kill(existing, 0)  # raises if process doesn't exist
                logger.error("Bot already running (PID %d) – exiting.", existing)
                sys.exit(0)
        except (ProcessLookupError, ValueError):
            pass
    _PIDFILE.write_text(str(os.getpid()))


def release_singleton() -> None:
    try:
        _PIDFILE.unlink(missing_ok=True)
    except Exception:
        pass


class ShutdownFlag:
    """Thread-safe flag that becomes True when SIGTERM/SIGINT is received."""

    def __init__(self):
        self._running = True
        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)

    def _handler(self, sig, frame):
        logger.info("Shutdown signal received (%s)", sig)
        self._running = False

    def is_running(self) -> bool:
        return self._running

    def stop(self):
        self._running = False
