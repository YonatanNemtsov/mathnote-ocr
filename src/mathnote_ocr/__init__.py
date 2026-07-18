"""MathNote OCR — stroke-based handwritten math to LaTeX."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mathnote-ocr")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "0.0.0.dev0"

from mathnote_ocr.api import MathOCR, Session
from mathnote_ocr.expression import DetectedSymbol, Expression, empty_expression
from mathnote_ocr.pin import PinEdge, PinnedTree, PinSymbol
from mathnote_ocr.tree_parser.tree_v2 import Edge, Tree

__all__ = [
    "MathOCR",
    "Session",
    "Expression",
    "DetectedSymbol",
    "empty_expression",
    "Tree",
    "Edge",
    "PinnedTree",
    "PinSymbol",
    "PinEdge",
]
