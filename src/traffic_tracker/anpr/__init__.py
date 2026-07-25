"""License-plate localization utilities.

This package implements plate detection only. OCR is intentionally separate so
plate localization can be validated before text recognition is introduced.
"""

from .detector import PlateDetector
from .memory import PlateMemory

__all__ = ["PlateDetector", "PlateMemory"]
