"""Image Vector Converter - low quality bitmap to Illustrator-ready EPS."""

from .document import VectorDocument, VectorLayer, save_eps, save_svg, to_eps, to_svg
from .engine import ConverterSettings, convert, load_image, render
from .enhance import EnhanceSettings, enhance
from .quantize import quantize
from .trace import TraceSettings, trace_mask

__version__ = "1.0.0"

__all__ = [
    "ConverterSettings",
    "EnhanceSettings",
    "TraceSettings",
    "VectorDocument",
    "VectorLayer",
    "convert",
    "enhance",
    "load_image",
    "quantize",
    "render",
    "save_eps",
    "save_svg",
    "to_eps",
    "to_svg",
    "trace_mask",
    "__version__",
]
