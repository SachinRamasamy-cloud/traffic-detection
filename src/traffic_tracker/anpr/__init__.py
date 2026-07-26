"""License-plate localization and OCR utilities."""

from .detector import PlateDetector
from .memory import PlateMemory
from .ocr_engine import OCRRead, PlateOCREngine
from .temporal_consensus import PlateTextConsensus, PlateTextResult

__all__ = [
    "OCRRead",
    "PlateDetector",
    "PlateMemory",
    "PlateOCREngine",
    "PlateTextConsensus",
    "PlateTextResult",
]
