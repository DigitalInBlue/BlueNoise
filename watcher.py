from __future__ import annotations
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
import time
import os
from typing import Callable, List, Optional


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif", ".webp"}


class _EnqueueHandler(FileSystemEventHandler):
    def __init__(self, callback: Callable[[List[str]], None]):
        self.cb = callback

    def on_created(self, event):
        if event.is_directory:
            return
        _, ext = os.path.splitext(event.src_path.lower())
        if ext in IMAGE_EXTS:
            self.cb([event.src_path])


class FolderWatcher:
    def __init__(self, callback: Callable[[List[str]], None]):
        self._obs: Optional[Observer] = None
        self._path: Optional[str] = None
        self._cb = callback

    def start(self, path: str):
        self.stop()
        self._path = path
        handler = _EnqueueHandler(self._cb)
        self._obs = Observer()
        self._obs.schedule(handler, path, recursive=False)
        self._obs.start()

    def stop(self):
        if self._obs:
            self._obs.stop()
            self._obs.join(timeout=2.0)
            self._obs = None
            self._path = None

    def is_running(self) -> bool:
        return self._obs is not None

    def path(self) -> Optional[str]:
        return self._path
