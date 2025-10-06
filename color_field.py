from __future__ import annotations
from typing import Optional, Tuple
from PySide6 import QtWidgets, QtGui, QtCore
import re

_HEX_RE = re.compile(r"^#?[0-9a-fA-F]{6}$")
_RGB_RE = re.compile(r"^\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*$")
_RGB_FN_RE = re.compile(r"^\s*rgb\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)\s*$")

def _clamp(v: int) -> int:
    return max(0, min(255, v))

def _parse_color(text: str) -> Optional[QtGui.QColor]:
    t = text.strip()
    m = _HEX_RE.match(t)
    if m:
        if not t.startswith("#"):
            t = "#" + t
        c = QtGui.QColor(t)
        return c if c.isValid() else None
    m = _RGB_RE.match(t) or _RGB_FN_RE.match(t)
    if m:
        r, g, b = ( _clamp(int(m.group(i))) for i in (1,2,3) )
        return QtGui.QColor(r, g, b)
    # Try named colors (e.g., "red")
    c = QtGui.QColor(t)
    return c if c.isValid() else None

def _to_hex(c: QtGui.QColor) -> str:
    return c.name(QtGui.QColor.NameFormat.HexRgb)

class ColorField(QtWidgets.QWidget):
    """Color input with:
       - standard QColorDialog picker
       - line edit accepting #RRGGBB / R,G,B / rgb(r,g,b) / named
       - swatch button reflecting current color
    """
    changed = QtCore.Signal(str)  # emits hex "#RRGGBB"

    def __init__(self, initial: str = "#00ffd5", parent=None):
        super().__init__(parent)
        self._color = _parse_color(initial) or QtGui.QColor("#00ffd5")

        self._layout = QtWidgets.QHBoxLayout(self)
        self._layout.setContentsMargins(0,0,0,0)

        self._edit = QtWidgets.QLineEdit(_to_hex(self._color))
        self._edit.setPlaceholderText("#RRGGBB or r,g,b")
        self._edit.editingFinished.connect(self._on_edit_finished)

        self._btn = QtWidgets.QPushButton()
        self._btn.setFixedWidth(28)
        self._btn.clicked.connect(self._pick)
        self._update_swatch()

        self._layout.addWidget(self._edit, 1)
        self._layout.addWidget(self._btn)

    def _update_swatch(self):
        c = self._color
        pix = QtGui.QPixmap(18, 18)
        pix.fill(c)
        self._btn.setIcon(QtGui.QIcon(pix))
        self._btn.setIconSize(QtCore.QSize(18,18))

    def _on_edit_finished(self):
        text = self._edit.text()
        c = _parse_color(text)
        if not c:
            # revert to current color on invalid input
            self._edit.setText(_to_hex(self._color))
            return
        self._color = c
        self._edit.setText(_to_hex(self._color))
        self._update_swatch()
        self.changed.emit(self.value())

    def _pick(self):
        c = QtWidgets.QColorDialog.getColor(self._color, self, "Pick Color")
        if c.isValid():
            self._color = c
            self._edit.setText(_to_hex(self._color))
            self._update_swatch()
            self.changed.emit(self.value())

    def value(self) -> str:
        return _to_hex(self._color)
