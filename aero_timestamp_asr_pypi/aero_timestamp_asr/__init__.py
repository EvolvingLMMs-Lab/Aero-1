"""Aero Timestamp ASR package."""

__version__ = "0.0.1"

from .loader import load_aero_model
from .transcribe import transcribe

__all__ = ["load_aero_model", "transcribe"]
