from __future__ import annotations
import importlib
import importlib.util
import pkgutil
import logging
from typing import Dict, Any, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---------------- registry & types (unchanged) ----------------
_registry: Dict[str, "VariantMeta"] = {}

@dataclass
class Option: default: Any
class Bool(Option): pass
class Int(Option):
    def __init__(self, default: int, min_value: int, max_value: int, step: int = 1):
        super().__init__(default); self.min=min_value; self.max=max_value; self.step=step
class Float(Option):
    def __init__(self, default: float, min_value: float, max_value: float, step: float = 0.01):
        super().__init__(default); self.min=min_value; self.max=max_value; self.step=step
class Enum(Option):
    def __init__(self, default: str, choices):
        super().__init__(default); self.choices=list(choices)
class Color(Option): pass

@dataclass
class VariantMeta:
    name: str
    func: Callable[..., Any]
    options: Dict[str, Option]
    animated: bool
    preview_safe: bool

def variant(name: str, *, options: Dict[str, Option], animated: bool=False, preview_safe: bool=True):
    def deco(fn: Callable[..., Any]):
        _registry[name] = VariantMeta(name=name, func=fn, options=options, animated=animated, preview_safe=preview_safe)
        return fn
    return deco

# ---------------- robust discovery ----------------
def _import_variants_package(package: str = "variants") -> None:
    """Import the variants package and its submodules; surface exceptions."""
    pkg = importlib.import_module(package)
    # Import autoload first (if present), so it can register variants.py
    try:
        importlib.import_module(f"{package}.autoload")
    except ModuleNotFoundError:
        pass
    # Import all other modules in the package (classic, experimental, etc.)
    if hasattr(pkg, "__path__"):
        for m in pkgutil.iter_modules(pkg.__path__):
            modname = f"{package}.{m.name}"
            if modname.endswith(".autoload"):
                continue
            importlib.import_module(modname)

def _fallback_load_variants_py() -> None:
    """If nothing registered, try loading project-root variants.py directly."""
    import os
    root = os.path.abspath(os.path.dirname(__file__))          # .../variants_core.py
    root = os.path.abspath(os.path.join(root, os.pardir))      # project root
    path = os.path.join(root, "variants.py")
    if not os.path.exists(path):
        return
    spec = importlib.util.spec_from_file_location("user_variants", path)
    if spec is None or spec.loader is None:
        return
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    VARIANTS = getattr(mod, "VARIANTS", {})
    if not isinstance(VARIANTS, dict):
        return
    from PIL import Image
    def wrap(fn):
        def f(img: Image.Image, **kwargs):  # pass kwargs if the function accepts them
            try:
                return fn(img, **kwargs)
            except TypeError:
                return fn(img)
        return f
    for name, fn in VARIANTS.items():
        if callable(fn):
            # best-effort: no options inference here; static registration
            variant(name=name, options={}, animated=(name=="scan_jitter_gif"), preview_safe=(name!="scan_jitter_gif"))(wrap(fn))

def discover_variants(package: str = "variants") -> Dict[str, VariantMeta]:
    """Load variants from the package; fall back to root variants.py if needed."""
    # Clear previous (in case of hot-reload)
    # (If you want to keep prior, drop this)
    # _registry.clear()  # optional; keep commented if you load once

    try:
        _import_variants_package(package)
    except Exception as e:
        logger.error("Variant discovery error importing package '%s': %s", package, e, exc_info=True)

    if not _registry:
        try:
            _fallback_load_variants_py()
        except Exception as e:
            logger.error("Fallback load of variants.py failed: %s", e, exc_info=True)

    logger.info("Variants registered: %d -> %s", len(_registry), sorted(_registry.keys()))
    return dict(_registry)
