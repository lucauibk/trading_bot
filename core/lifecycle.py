"""
Singleton-Lock and signal handling – extracted from main.py.

Uses fcntl.flock for a race-free OS-level exclusive lock so two concurrent
start paths (start.sh --bot and Dashboard subprocess) can never both succeed.
The PID file is still written (for stop.sh and external tooling) but is no
longer the primary guard — the flock is.
"""

import fcntl
import logging
import os
import signal
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_PIDFILE  = Path(".bot.pid")
_LOCKFILE = Path(".bot.lock")

# Module-level handle: the OS lock is held as long as this fd is open.
# Never close it deliberately — the lock is released automatically on process exit.
_lock_fh = None


def acquire_singleton() -> None:
    """Ensure only one bot instance runs. Exits immediately if another is running."""
    global _lock_fh

    # --- Primary guard: fcntl exclusive lock (race-free) ---
    _LOCKFILE.touch(exist_ok=True)
    fh = _LOCKFILE.open("r")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.close()
        # Another process holds the lock; try to find its PID for a helpful log
        try:
            pid = int(_PIDFILE.read_text().strip())
            logger.error("Bot already running (PID %d, lock held) – exiting.", pid)
        except Exception:
            logger.error("Bot already running (lock held by unknown PID) – exiting.")
        sys.exit(0)

    # Lock acquired – keep the file handle alive for the life of the process
    _lock_fh = fh

    # --- Secondary: write PID file (for stop.sh / external tooling) ---
    _PIDFILE.write_text(str(os.getpid()))
    logger.debug("Singleton acquired (PID %d)", os.getpid())


def release_singleton() -> None:
    global _lock_fh
    try:
        _PIDFILE.unlink(missing_ok=True)
    except Exception:
        pass
    if _lock_fh is not None:
        try:
            fcntl.flock(_lock_fh, fcntl.LOCK_UN)
            _lock_fh.close()
        except Exception:
            pass
        _lock_fh = None


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
