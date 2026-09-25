"""Paradee: an 8M-parameter English text-to-speech model distilled from Kokoro-82M (voice af_heart)."""
from .tts import Paradee, SAMPLE_RATE

__all__ = ["Paradee", "SAMPLE_RATE"]
__version__ = "1.0.0"
