import logging
import os
from logging.handlers import RotatingFileHandler
import sys

_LOG_DIR = os.path.abspath("logs")
_LOG_FILE = os.path.join(_LOG_DIR, "app.log")

def setup_logging(level=logging.DEBUG) -> None:
    os.makedirs(_LOG_DIR, exist_ok=True)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(fmt)
    ch.setLevel(level)
    root.addHandler(ch)

    fh = RotatingFileHandler(_LOG_FILE, maxBytes=5_000_000, backupCount=3)
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)
    root.addHandler(fh)

    # Tame noisy libs
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("watchdog").setLevel(logging.INFO)

def log_path() -> str:
    return _LOG_FILE
