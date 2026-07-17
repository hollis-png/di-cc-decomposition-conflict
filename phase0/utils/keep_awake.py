"""Prevent Windows from sleeping while a long-running GPU script is active.

Root cause of repeated multi-hour job interruptions (2026-07-14): the
machine uses Modern Standby, which sleeps on its own idle timer regardless
of the classic `powercfg` AC standby-timeout setting (that was already 0 /
never, yet the machine still slept every 1-2 hours per the Kernel-Power
event log). Rather than changing the machine's global power policy (which
would affect normal use), each long-running script requests sleep
prevention for just its own lifetime via the Windows execution-state API --
the same mechanism video players / download managers use.

No-op on non-Windows platforms.
"""
import atexit
import ctypes
import sys

_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


def prevent_sleep() -> None:
    if sys.platform != "win32":
        return
    ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED)
    atexit.register(allow_sleep)


def allow_sleep() -> None:
    if sys.platform != "win32":
        return
    ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
