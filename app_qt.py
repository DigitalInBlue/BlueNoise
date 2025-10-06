from __future__ import annotations
import sys
import os
import logging
from typing import Dict, Any, List, Optional, Tuple
from PySide6 import QtCore, QtGui, QtWidgets
from PIL import ImageQt, Image

from logconf import setup_logging
from io_utils import load_image, downscale_for_preview, save_png, make_output_path
from variants_core import discover_variants, VariantMeta, Bool, Int, Float, Enum, Color
from encode_mp4 import encode_frames_to_mp4
from presets import Preset
from watcher import FolderWatcher
from color_field import ColorField  # color picker + hex/RGB input


APP_TITLE = "Image Variants GUI"
PREVIEW_DEBOUNCE_MS = 150
PREVIEW_TIMEOUT_MS = 3000

def pil_to_qpixmap(img: Image.Image) -> QtGui.QPixmap:
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA")
    buf = img.tobytes("raw", img.mode)
    w, h = img.size
    if img.mode == "RGBA":
        qimg = QtGui.QImage(buf, w, h, QtGui.QImage.Format_RGBA8888)
    else:
        qimg = QtGui.QImage(buf, w, h, 3 * w, QtGui.QImage.Format_RGB888)
    return QtGui.QPixmap.fromImage(qimg.copy())


class FileListWidget(QtWidgets.QListWidget):
    fileDropped = QtCore.Signal(list)
    def __init__(self):
        super().__init__()
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setAcceptDrops(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DropOnly)
    def dragEnterEvent(self, e: QtGui.QDragEnterEvent) -> None:
        e.acceptProposedAction() if e.mimeData().hasUrls() else e.ignore()
    def dragMoveEvent(self, e: QtGui.QDragMoveEvent) -> None:
        e.acceptProposedAction() if e.mimeData().hasUrls() else e.ignore()
    def dropEvent(self, e: QtGui.QDropEvent) -> None:
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        paths = [p for p in paths if p and os.path.isfile(p)]
        if paths: self.fileDropped.emit(paths)


class VariantOptionsPanel(QtWidgets.QWidget):
    optionsChanged = QtCore.Signal(dict)
    def __init__(self):
        super().__init__()
        self._layout = QtWidgets.QFormLayout(self)
        self._controls: Dict[str, QtWidgets.QWidget] = {}
    def build_for(self, meta):
        while self._layout.count():
            itm = self._layout.takeAt(0)
            w = itm.widget()
            if w: w.deleteLater()
        self._controls.clear()
        for key, opt in meta.options.items():
            if isinstance(opt, Bool):
                w = QtWidgets.QCheckBox(); w.setChecked(bool(opt.default)); w.stateChanged.connect(lambda _=None: self.emit())
            elif isinstance(opt, Int):
                w = QtWidgets.QSpinBox(); w.setRange(opt.min, opt.max); w.setSingleStep(opt.step); w.setValue(int(opt.default)); w.valueChanged.connect(lambda _=None: self.emit())
            elif isinstance(opt, Float):
                w = QtWidgets.QDoubleSpinBox(); w.setRange(opt.min, opt.max); w.setSingleStep(opt.step); w.setDecimals(3); w.setValue(float(opt.default)); w.valueChanged.connect(lambda _=None: self.emit())
            elif isinstance(opt, Enum):
                w = QtWidgets.QComboBox(); w.addItems(opt.choices); w.setCurrentText(opt.default); w.currentTextChanged.connect(lambda _=None: self.emit())
            elif isinstance(opt, Color):
                w = ColorField(str(opt.default)); w.changed.connect(lambda _=None: self.emit())
            else:
                w = QtWidgets.QLabel("Unsupported option type")
            self._controls[key] = w
            self._layout.addRow(key, w)
    def values(self) -> Dict[str, Any]:
        vals: Dict[str, Any] = {}
        for key, w in self._controls.items():
            if isinstance(w, QtWidgets.QCheckBox): vals[key] = w.isChecked()
            elif isinstance(w, QtWidgets.QSpinBox): vals[key] = w.value()
            elif isinstance(w, QtWidgets.QDoubleSpinBox): vals[key] = w.value()
            elif isinstance(w, QtWidgets.QComboBox): vals[key] = w.currentText()
            elif isinstance(w, ColorField): vals[key] = w.value()
        return vals
    def set_values(self, values: Dict[str, Any]):
        # populate controls from saved values (if keys match)
        for key, w in self._controls.items():
            if key not in values: continue
            v = values[key]
            if isinstance(w, QtWidgets.QCheckBox): w.setChecked(bool(v))
            elif isinstance(w, QtWidgets.QSpinBox): w.setValue(int(v))
            elif isinstance(w, QtWidgets.QDoubleSpinBox): w.setValue(float(v))
            elif isinstance(w, QtWidgets.QComboBox):
                idx = w.findText(str(v))
                w.setCurrentIndex(idx if idx >= 0 else 0)
            elif isinstance(w, ColorField): w._edit.setText(str(v)); w._on_edit_finished()
        self.emit()
    def emit(self): self.optionsChanged.emit(self.values())


class ABPreviewWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(400, 300)
        self._pix_orig: Optional[QtGui.QPixmap] = None
        self._pix_proc: Optional[QtGui.QPixmap] = None
        self._split = 0.5
        self._busy = False
        self.setMouseTracking(True)
        self._logger = logging.getLogger("Preview")
        self._checker_brush = self._make_checker_brush()

    def _make_checker_brush(self) -> QtGui.QBrush:
        """Create a 16x16 checkerboard texture brush (two grays)."""
        size = 16
        pm = QtGui.QPixmap(size, size)
        pm.fill(QtCore.Qt.transparent)
        p = QtGui.QPainter(pm)
        c1 = QtGui.QColor(48, 48, 48)
        c2 = QtGui.QColor(64, 64, 64)
        p.fillRect(0, 0, size//2, size//2, c1)
        p.fillRect(size//2, 0, size//2, size//2, c2)
        p.fillRect(0, size//2, size//2, size//2, c2)
        p.fillRect(size//2, size//2, size//2, size//2, c1)
        p.end()
        return QtGui.QBrush(pm)

    def set_images(self, img_orig: Optional[Image.Image], img_proc: Optional[Image.Image]):
        self._logger.debug("set_images(): orig=%s proc=%s", bool(img_orig), bool(img_proc))
        self._pix_orig = pil_to_qpixmap(img_orig) if img_orig is not None else None
        self._pix_proc = pil_to_qpixmap(img_proc) if img_proc is not None else None
        self._busy = False; self.update()
    def set_busy(self, busy: bool):
        self._busy = busy; self._logger.debug("set_busy(%s)", busy); self.update()
    def mousePressEvent(self, e: QtGui.QMouseEvent): self._update_split(e.position().x())
    def mouseMoveEvent(self, e: QtGui.QMouseEvent):
        if e.buttons() & QtCore.Qt.LeftButton: self._update_split(e.position().x())
    def _update_split(self, x: float):
        r = self.rect()
        if r.width() > 0: self._split = max(0.0, min(1.0, (x - r.x()) / r.width())); self.update()

    def paintEvent(self, _):
        p = QtGui.QPainter(self)

        # Fill overall widget background (outside the image area)
        p.fillRect(self.rect(), QtGui.QColor("#202020"))

        # If nothing to show, still draw checker in a centered empty box (optional)
        # but early-out to skip splits
        base_pix = self._pix_proc or self._pix_orig
        if not base_pix:
            return

        # Compute destination rect to draw (fit-to-window)
        r = self.rect()
        pw, ph = base_pix.width(), base_pix.height()
        scale = min(r.width() / pw, r.height() / ph)
        w = int(pw * scale)
        h = int(ph * scale)
        x = r.x() + (r.width() - w) // 2
        y = r.y() + (r.height() - h) // 2
        dst = QtCore.QRect(x, y, w, h)

        # 1) Draw checkerboard under the image area
        p.save()
        p.setBrush(self._checker_brush)
        p.setPen(QtCore.Qt.NoPen)
        p.drawRect(dst)
        p.restore()

        # 2) Decide split geometry
        split_px = int(w * max(0.0, min(1.0, self._split)))
        left_rect  = QtCore.QRect(x, y, split_px, h)
        right_rect = QtCore.QRect(x + split_px, y, w - split_px, h)

        # 3) Draw sides separately to avoid “bleeding through” alpha
        if self._pix_orig and self._pix_proc:
            # Left = original (only)
            if left_rect.width() > 0:
                p.save()
                p.setClipRect(left_rect)
                p.drawPixmap(dst, self._pix_orig)
                p.restore()

            # Right = processed (only)
            if right_rect.width() > 0:
                p.save()
                p.setClipRect(right_rect)
                p.drawPixmap(dst, self._pix_proc)
                p.restore()

            # Handle line
            p.setPen(QtGui.QPen(QtGui.QColor("#FFFFFF"), 2))
            p.drawLine(x + split_px, y, x + split_px, y + h)

        else:
            # Only one pixmap: draw over checker (no clipping needed)
            pix = self._pix_proc or self._pix_orig
            p.drawPixmap(dst, pix)

        # 4) Busy overlay on top (unchanged)
        if self._busy:
            p.fillRect(self.rect(), QtGui.QColor(0, 0, 0, 100))
            p.setPen(QtGui.QPen(QtGui.QColor("#FFFFFF")))
            font = p.font()
            font.setPointSize(font.pointSize() + 2)
            p.setFont(font)
            p.drawText(self.rect(), QtCore.Qt.AlignCenter, "Rendering preview…")


class Worker(QtCore.QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__(); self.fn = fn; self.args = args; self.kwargs = kwargs
        self.signals = WorkerSignals(); self._logger = logging.getLogger("Worker")
    @QtCore.Slot()
    def run(self):
        try:
            self._logger.debug("Run start")
            result = self.fn(*self.args, **self.kwargs)
        except Exception as e:
            logging.exception("Worker error")
            self.signals.error.emit(str(e))
        else:
            self.signals.result.emit(result)
        finally:
            self._logger.debug("Run finished")
            self.signals.finished.emit()


class WorkerSignals(QtCore.QObject):
    finished = QtCore.Signal()
    error = QtCore.Signal(str)
    result = QtCore.Signal(object)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        setup_logging()
        self.setWindowTitle(APP_TITLE)
        self.resize(1400, 880)
        self.registry = discover_variants()
        self.logger = logging.getLogger("GUI"); self.logger.setLevel(logging.DEBUG)

        # --- state ---
        self.out_dir = os.path.abspath("_variants")
        self.overwrite_warned = False
        self.pref_use_1920 = True
        self.active_variant: Optional[VariantMeta] = None
        self.active_file: Optional[str] = None

        # Per-variant saved options
        self.variant_options: Dict[str, Dict[str, Any]] = {}

        # current UI options for the active variant
        self.current_options: Dict[str, Any] = {}
        self.animated_fps_default = 24; self.animated_crf_default = 20

        # Preview infra
        self._preview_base_cache: Dict[str, Image.Image] = {}
        self._seen: Dict[str, float] = {}
        self._preview_token = 0
        self._debounce_timer = QtCore.QTimer(self); self._debounce_timer.setSingleShot(True)
        self._debounce_timer.timeout.connect(self._kick_preview_worker)
        self._timeout_timer = QtCore.QTimer(self); self._timeout_timer.setSingleShot(True)
        self._timeout_timer.timeout.connect(self._on_preview_timeout)

        # Watcher
        self.watcher = FolderWatcher(self._watch_enqueue)

        # --- UI ---
        self.files = FileListWidget()
        self.files.fileDropped.connect(self.add_files)
        self.files.itemSelectionChanged.connect(self.on_file_selection)

        # Variant list with checkboxes (check → included in processing; select → active for preview)
        self.variant_list = QtWidgets.QListWidget()
        self.variant_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        for name in sorted(self.registry.keys()):
            it = QtWidgets.QListWidgetItem(name)
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            it.setCheckState(QtCore.Qt.Unchecked)
            self.variant_list.addItem(it)
        self.variant_list.currentTextChanged.connect(self.on_variant_changed)

        # Quick actions for checkboxes
        self.btn_check_all = QtWidgets.QPushButton("Check All")
        self.btn_uncheck_all = QtWidgets.QPushButton("Uncheck All")
        self.btn_check_all.clicked.connect(self._check_all_variants)
        self.btn_uncheck_all.clicked.connect(self._uncheck_all_variants)

        self.options_panel = VariantOptionsPanel()
        self.options_panel.optionsChanged.connect(self.on_options_changed)

        self.preview_title = QtWidgets.QLabel("—")
        self.preview_title.setStyleSheet("color:#ddd; font-weight:bold; padding:4px;")
        self.preview = ABPreviewWidget()

        self.btn_process_selected = QtWidgets.QPushButton("Process Selected")
        self.btn_process_all = QtWidgets.QPushButton("Process All Queued")
        self.btn_process_selected.clicked.connect(self.process_selected)
        self.btn_process_all.clicked.connect(self.process_all)

        self.chk_1920 = QtWidgets.QCheckBox("Prefer *_1920 source if present"); self.chk_1920.setChecked(True)
        self.chk_1920.stateChanged.connect(lambda _=None: setattr(self, "pref_use_1920", self.chk_1920.isChecked()))

        self.slider_conc = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.slider_conc.setMinimum(1)
        self.slider_conc.setMaximum(max(4, QtCore.QThreadPool.globalInstance().maxThreadCount()))
        self.slider_conc.setValue(4); self.slider_conc.valueChanged.connect(self._apply_concurrency)

        self.spin_fps = QtWidgets.QSpinBox(); self.spin_fps.setRange(12, 60); self.spin_fps.setValue(self.animated_fps_default)
        self.spin_crf = QtWidgets.QSpinBox(); self.spin_crf.setRange(10, 40); self.spin_crf.setValue(self.animated_crf_default)

        self.edit_watch = QtWidgets.QLineEdit()
        self.btn_browse_watch = QtWidgets.QPushButton("Browse…")
        self.btn_toggle_watch = QtWidgets.QPushButton("Watch: OFF")
        self.btn_browse_watch.clicked.connect(self._browse_watch)
        self.btn_toggle_watch.clicked.connect(self._toggle_watch)

        # Layout
        left = QtWidgets.QWidget(); llay = QtWidgets.QVBoxLayout(left)
        llay.addWidget(QtWidgets.QLabel("Files (drag & drop):")); llay.addWidget(self.files, 1)
        lwatch = QtWidgets.QHBoxLayout(); lwatch.addWidget(self.edit_watch, 1); lwatch.addWidget(self.btn_browse_watch); lwatch.addWidget(self.btn_toggle_watch)
        llay.addLayout(lwatch)

        right = QtWidgets.QWidget(); rlay = QtWidgets.QVBoxLayout(right)
        rlay.addWidget(QtWidgets.QLabel("Variants:"))
        rlay.addWidget(self.variant_list, 1)
        cbox = QtWidgets.QHBoxLayout()
        cbox.addWidget(self.btn_check_all)
        cbox.addWidget(self.btn_uncheck_all)
        cbox.addStretch(1)
        rlay.addLayout(cbox)
        rlay.addWidget(QtWidgets.QLabel("Options:"))
        rlay.addWidget(self.options_panel, 2)

        out_group = QtWidgets.QGroupBox("Output & Preferences"); out_l = QtWidgets.QFormLayout(out_group)
        out_l.addRow("Prefer *_1920:", self.chk_1920); out_l.addRow("Animated FPS:", self.spin_fps)
        out_l.addRow("Animated CRF:", self.spin_crf); out_l.addRow("Concurrency:", self.slider_conc)
        rlay.addWidget(out_group)
        rlay.addWidget(self.btn_process_selected)
        rlay.addWidget(self.btn_process_all)

        center = QtWidgets.QWidget(); cv = QtWidgets.QVBoxLayout(center)
        cv.addWidget(self.preview_title); cv.addWidget(self.preview, 1)

        central = QtWidgets.QWidget(); hlay = QtWidgets.QHBoxLayout(central)
        hlay.addWidget(left, 2); hlay.addWidget(center, 5); hlay.addWidget(right, 3)
        self.setCentralWidget(central)

        # Menu: Presets (NOTE: these now store global settings + current active variant’s options.
        # We could extend to store all per-variant options if you want—easy follow-up.)
        mb = self.menuBar(); m_file = mb.addMenu("&File"); m_presets = m_file.addMenu("Presets")
        act_save = QtGui.QAction("Save…", self); act_load = QtGui.QAction("Load…", self)
        act_save.triggered.connect(self._save_preset); act_load.triggered.connect(self._load_preset)
        m_presets.addAction(act_save); m_presets.addAction(act_load)

        if self.variant_list.count() > 0: self.variant_list.setCurrentRow(0)
        self.threadpool = QtCore.QThreadPool.globalInstance()
        self._apply_concurrency(self.slider_conc.value())
        self.statusBar().showMessage("Ready")
        self.logger.debug("MainWindow initialized; threadpool max=%d", self.threadpool.maxThreadCount())

    # ---------- helpers ----------
    def _find_item_by_name(self, name: str) -> Optional[QtWidgets.QListWidgetItem]:
        for i in range(self.variant_list.count()):
            it = self.variant_list.item(i)
            if it.text() == name:
                return it
        return None

    def _checked_variant_names(self) -> List[str]:
        names: List[str] = []
        for i in range(self.variant_list.count()):
            it = self.variant_list.item(i)
            if it.checkState() == QtCore.Qt.Checked:
                names.append(it.text())
        return names

    def _check_all_variants(self):
        for i in range(self.variant_list.count()):
            self.variant_list.item(i).setCheckState(QtCore.Qt.Checked)

    def _uncheck_all_variants(self):
        for i in range(self.variant_list.count()):
            self.variant_list.item(i).setCheckState(QtCore.Qt.Unchecked)

    # ---------- watch ----------

    def _browse_watch(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose watch folder", os.getcwd())
        if d: self.edit_watch.setText(d)


    def _toggle_watch(self):
        if self.watcher.is_running():
            self.watcher.stop(); self.btn_toggle_watch.setText("Watch: OFF")
            self.statusBar().showMessage("Watch stopped", 3000); self.logger.info("Watch stopped")
        else:
            path = self.edit_watch.text().strip()
            if path and os.path.isdir(path):
                self.watcher.start(path); self.btn_toggle_watch.setText("Watch: ON")
                self.statusBar().showMessage(f"Watching: {path}", 3000); self.logger.info("Watch started: %s", path)

    def _watch_enqueue(self, paths: List[str]):
        QtCore.QMetaObject.invokeMethod(self, "_enqueue_files", QtCore.Qt.QueuedConnection, QtCore.Q_ARG(list, paths))

    @QtCore.Slot(list)

    def _enqueue_files(self, paths: List[str]):
        self.add_files(paths); self.statusBar().showMessage(f"Enqueued {len(paths)} new file(s)", 2000)

    # ---- Concurrency ----
    def _apply_concurrency(self, val: int):
        self.threadpool.setMaxThreadCount(max(1, int(val)))
        self.statusBar().showMessage(f"Concurrency: {val}", 1500)
    # ---------- files & variants ----------

    def add_files(self, paths: List[str]):
        added = 0
        for p in paths:
            try:
                mt = os.path.getmtime(p)
            except OSError:
                continue
            if p in self._seen and self._seen[p] == mt:
                continue
            self._seen[p] = mt
            if not self._list_contains(self.files, p):
                self.files.addItem(QtWidgets.QListWidgetItem(p)); added += 1
            self._preview_base_cache.pop(p, None)
        if added: self.logger.info("Enqueued %d files", added)

    def _list_contains(self, listw: QtWidgets.QListWidget, text: str) -> bool:
        for i in range(listw.count()):
            if listw.item(i).text() == text: return True
        return False


    def on_file_selection(self):
        items = self.files.selectedItems()
        self.active_file = items[0].text() if items else None
        self._schedule_preview()


    def on_variant_changed(self, name: str):
        meta = self.registry.get(name)
        self.active_variant = meta
        if meta:
            # Build panel and load saved options (or defaults)
            self.options_panel.build_for(meta)
            defaults = {k: opt.default for k, opt in meta.options.items()}
            saved = self.variant_options.get(meta.name, defaults)
            self.current_options = dict(saved)
            self.options_panel.set_values(self.current_options)
        self._schedule_preview()


    def on_options_changed(self, values: Dict[str, Any]):
        self.current_options = values
        # persist for this variant
        if self.active_variant:
            self.variant_options[self.active_variant.name] = dict(values)
        self._schedule_preview()

    def choose_source_for(self, path: str) -> str:
        if not self.pref_use_1920: return path
        base, ext = os.path.splitext(path); candidate = f"{base}_1920{ext}"
        return candidate if os.path.exists(candidate) else path

    # ---------- preview ----------
    def _set_preview_title(self):
        if not self.active_file or not self.active_variant:
            self.preview_title.setText("—"); return
        self.preview_title.setText(f"{os.path.basename(self.active_file)}  —  {self.active_variant.name}")

    def _get_preview_base(self, path: str) -> Optional[Image.Image]:
        cached = self._preview_base_cache.get(path)
        if cached is not None:
            return cached
        try: img = load_image(path)
        except Exception as e:
            self.logger.error("Failed to load %s: %s", path, e); return None
        base_small = downscale_for_preview(img); self._preview_base_cache[path] = base_small
        return base_small

    def _render_preview_pair_small(self, path: str, meta: VariantMeta, opts: Dict[str, Any]) -> Tuple[Image.Image, Image.Image]:
        base = self._get_preview_base(path)
        if base is None: raise RuntimeError("Preview base not available")
        orig_small = base.copy()
        if meta.animated:
            frames = meta.func(base, **opts)
            proc = frames[0]["image"] if frames else base
        else:
            try:
                proc = meta.func(base, **opts)
            except Exception as e:
                self.logger.error("Failed to process %s: %s", path, e)
                proc = base
        return (orig_small, proc)


    def _schedule_preview(self):
        self._set_preview_title()
        if not self.active_file or not self.active_variant:
            self.preview.set_images(None, None); return
        self._preview_token += 1
        self.preview.set_busy(True)
        self._debounce_timer.start(PREVIEW_DEBOUNCE_MS)
        self._timeout_timer.start(PREVIEW_TIMEOUT_MS)


    def _on_preview_timeout(self):
        self.preview.set_busy(False)
        if self.active_file and self.active_variant:
            self._debounce_timer.stop(); self._debounce_timer.start(PREVIEW_DEBOUNCE_MS)


    def _kick_preview_worker(self):
        if not self.active_file or not self.active_variant:
            self.preview.set_images(None, None); self._timeout_timer.stop(); return
        token = self._preview_token
        path = self.choose_source_for(self.active_file)
        meta = self.active_variant
        opts = dict(self.current_options)

        def render(): return self._render_preview_pair_small(path, meta, opts)

    
        def on_result(pair, tok=token):
            if tok != self._preview_token: return
            self.preview.set_images(*pair); self._timeout_timer.stop()

    
        def on_error(msg: str, tok=token):
            if tok != self._preview_token: return
            self.preview.set_busy(False); self._timeout_timer.stop()
            self.statusBar().showMessage(f"Preview error: {msg}", 5000)

    
        def on_finished(tok=token):
            if tok != self._preview_token: return
            if self._timeout_timer.isActive(): self._timeout_timer.stop()
            self.preview.set_busy(False)

        worker = Worker(render)
        worker.signals.result.connect(on_result)
        worker.signals.error.connect(on_error)
        worker.signals.finished.connect(on_finished)
        self.threadpool.start(worker)

    # ---------- processing ----------
    def _targets_for(self, files: List[str], variant_names: List[str]) -> List[str]:
        targets: List[str] = []
        for in_path in files:
            for vname in variant_names:
                meta = self.registry.get(vname)
                if not meta: continue
                if meta.animated:
                    targets.append(make_output_path(self.out_dir, in_path, vname, "mp4"))
                    targets.append(make_output_path(self.out_dir, in_path, vname + "_first", "png"))
                else:
                    targets.append(make_output_path(self.out_dir, in_path, vname, "png"))
        return targets

    def ensure_overwrite_warning(self, targets: List[str]):
        if self.overwrite_warned: return True
        existing = [t for t in targets if os.path.exists(t)]
        if not existing: return True
        msg = QtWidgets.QMessageBox(self)
        msg.setIcon(QtWidgets.QMessageBox.Warning)
        msg.setWindowTitle("Overwrite warning")
        msg.setText(f"{len(existing)} output files already exist and will be overwritten.")
        msg.setInformativeText("Proceed?")
        msg.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if msg.exec() == QtWidgets.QMessageBox.Yes:
            self.overwrite_warned = True; return True
        return False


    def process_selected(self):
        self._process(items=self.files.selectedItems())


    def process_all(self):
        self._process(items=[self.files.item(i) for i in range(self.files.count())])

    def _process(self, items: List[QtWidgets.QListWidgetItem]):
        if not items:
            return
        chosen = self._checked_variant_names()
        if not chosen:
            QtWidgets.QMessageBox.information(self, "No variants", "Check one or more variants to process.")
            return

        file_paths = [it.text() for it in items]
        os.makedirs(self.out_dir, exist_ok=True)
        targets = self._targets_for(file_paths, chosen)
        if not self.ensure_overwrite_warning(targets):
            return

        fps = int(self.spin_fps.value()); crf = int(self.spin_crf.value())

        def job():
            for in_path in file_paths:
                img = load_image(self.choose_source_for(in_path))
                for vname in chosen:
                    meta = self.registry.get(vname)
                    if not meta: continue
                    # load saved options or defaults for each variant
                    opts = self.variant_options.get(vname, {k: o.default for k, o in meta.options.items()})
                    if meta.animated:
                        frames = meta.func(img.copy(), **opts)  # type: ignore
                        if frames:
                            save_png(frames[0]["image"], make_output_path(self.out_dir, in_path, vname + "_first", "png"), overwrite=True)
                        encode_frames_to_mp4(frames, make_output_path(self.out_dir, in_path, vname, "mp4"), fps=fps, crf=crf)
                    else:
                        out_img = meta.func(img.copy(), **opts)
                        save_png(out_img, make_output_path(self.out_dir, in_path, vname, "png"), overwrite=True)
            return True

        worker = Worker(job)
        worker.signals.result.connect(lambda _: self.statusBar().showMessage("Done", 3000))
        self.threadpool.start(worker)

    # ---------- presets (scope: active variant’s options + globals) ----------
    def _save_preset(self):
        if not self.active_variant:
            QtWidgets.QMessageBox.information(self, "Presets", "Select a variant first."); return
        p = Preset(
            version=1,
            selected_variant=self.active_variant.name,
            variant_options=dict(self.current_options),
            prefer_1920=self.pref_use_1920,
            animated_fps=int(self.spin_fps.value()),
            animated_crf=int(self.spin_crf.value()),
            concurrency=int(self.slider_conc.value()),
        )
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Preset", os.getcwd(), "Preset (*.json)")
        if not fn: return
        with open(fn, "w", encoding="utf-8") as f: f.write(p.to_json())
        self.statusBar().showMessage(f"Saved preset: {fn}", 4000)

    def _load_preset(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Preset", os.getcwd(), "Preset (*.json)")
        if not fn: return
        with open(fn, "r", encoding="utf-8") as f:
            p = Preset.from_json(f.read())
        row = self._find_item_by_name(p.selected_variant)
        if row:
            self.variant_list.setCurrentItem(row)
        self.pref_use_1920 = bool(p.prefer_1920); self.chk_1920.setChecked(self.pref_use_1920)
        self.spin_fps.setValue(int(p.animated_fps)); self.spin_crf.setValue(int(p.animated_crf))
        self.slider_conc.setValue(int(p.concurrency))
        # apply to active variant; store
        if self.active_variant:
            self.variant_options[self.active_variant.name] = dict(p.variant_options)
            self.options_panel.set_values(p.variant_options)
        self._schedule_preview()
        self.statusBar().showMessage(f"Loaded preset: {fn}", 4000)


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(); w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
