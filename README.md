# Image Variants GUI (Windows, PySide6)

A Windows-only GUI to preview and batch-apply image variants. Defaults to PNG output (alpha preserved). Animated variants export MP4 (H.264) **and** a first-frame PNG.

## Features
- Drag-and-drop files into a queue; optional folder watch (enqueue-only).
- Preview the **active file × active variant** (downsampled to max 1024px for speed).
- Dynamic **variant registry** with a decorator and an **option schema** (Bool/Int/Float/Enum/Color). New variants are auto-discovered on start.
- PNG output by default; animated variants → MP4 + first-frame PNG.
- Overwrite policy: **single consolidated warning**, then overwrite (also logs a WARN).
- `_1920` preference toggle; non-deterministic effects allowed by design.
- Windows-only; run from source (no packaging).

## Quick start
```bash
python -m pip install Pillow PySide6 watchdog imageio imageio-ffmpeg
python app_qt.py
```
