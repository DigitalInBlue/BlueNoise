from __future__ import annotations
import faulthandler
import os
import sys
import threading
import traceback
from datetime import datetime
from typing import Callable, Optional
from PySide6 import QtCore
import logging

logger = logging.getLogger("Errors")

_CRASH_DIR = os.path.abspath("logs")
_CRASH_DUMP = os.path.join(_CRASH_DIR, "crash.dump")

# Keep strong refs so they aren't GC'd
_faulthandler_file: Optional[object] = None
_qt_handler_ref: Optional[Callable] = None

def _ensure_dirs():
    os.makedirs(_CRASH_DIR, exist_ok=True)

def _write_dump(prefix: str, exc_text: str) -> None:
    _ensure_dirs()
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(_CRASH_DIR, f"{prefix}_{ts}.dump")
    with open(path, "w", encoding="utf-8") as f:
        f.write(exc_text)
    logger.error("Wrote exception dump: %s", path)

def install_global_exception_hooks(install_qt_handler: bool = True) -> None:
    """
    Capture: sys.excepthook, threading.excepthook, sys.unraisablehook,
    Qt messages (optional), and native crashes via faulthandler.
    """
    global _faulthandler_file, _qt_handler_ref
    _ensure_dirs()

    # 1) Python uncaught exceptions
    def excepthook(exc_type, exc, tb):
        buf = "".join(traceback.format_exception(exc_type, exc, tb))
        logger.critical("Uncaught exception:\n%s", buf)
        _write_dump("uncaught", buf)
        try:
            sys.__excepthook__(exc_type, exc, tb)
        except Exception:
            pass

    sys.excepthook = excepthook

    # 2) Threading exceptions (3.8+)
    def threading_hook(args: threading.ExceptHookArgs):
        buf = "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
        logger.critical("Thread exception in %s:\n%s", getattr(args.thread, "name", "<unknown>"), buf)
        _write_dump("thread", buf)
    try:
        threading.excepthook = threading_hook  # type: ignore[attr-defined]
    except Exception:
        pass

    # 3) Unraisable exceptions
    def unraisable_hook(unraisable):
        buf = "".join(traceback.format_exception(unraisable.exc_type, unraisable.exc_value, unraisable.exc_traceback))
        where = getattr(unraisable, "object", None)
        logger.error("Unraisable exception in %r:\n%s", where, buf)
        _write_dump("unraisable", buf)
    try:
        sys.unraisablehook = unraisable_hook  # type: ignore[attr-defined]
    except Exception:
        pass

    # 4) Qt messages → logger (guarded, stored in _qt_handler_ref to prevent GC)
    if install_qt_handler:
        def qt_message_handler(mode, context, message):
            try:
                level = {
                    QtCore.QtMsgType.QtDebugMsg: logging.DEBUG,
                    QtCore.QtMsgType.QtInfoMsg: logging.INFO,
                    QtCore.QtMsgType.QtWarningMsg: logging.WARNING,
                    QtCore.QtMsgType.QtCriticalMsg: logging.ERROR,
                    QtCore.QtMsgType.QtFatalMsg: logging.CRITICAL,
                }.get(mode, logging.INFO)
                file = getattr(context, "file", None) or "<qt>"
                line = getattr(context, "line", None)
                function = getattr(context, "function", None) or "?"
                text = f"{file}:{line} {function}: {message}"
                logger.log(level, "QT: %s", text)
                if mode == QtCore.QtMsgType.QtFatalMsg:
                    _write_dump("qt_fatal", text)
            except Exception:
                # Never let the handler crash the process
                logger.exception("Exception inside qt_message_handler")
        _qt_handler_ref = qt_message_handler
        try:
            QtCore.qInstallMessageHandler(_qt_handler_ref)
        except Exception:
            logger.exception("Failed to install Qt message handler")

    # 5) Faulthandler for native crashes: requires a *binary* file kept alive
    try:
        # Binary, unbuffered (buffering=0) for safety on Windows
        _faulthandler_file = open(_CRASH_DUMP, "ab", buffering=0)
        faulthandler.enable(file=_faulthandler_file, all_threads=True)
        logger.info("Faulthandler enabled: %s", _CRASH_DUMP)
    except Exception as e:
        logger.warning("Failed to enable faulthandler: %s", e)

def safe_slot(fn: Callable) -> Callable:
    """Decorator for Qt slots/callbacks: logs exceptions instead of letting them bubble into Qt."""
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            buf = traceback.format_exc()
            logger.error("Exception in slot %s:\n%s", getattr(fn, "__name__", str(fn)), buf)
            _write_dump("slot", buf)
    return wrapper
