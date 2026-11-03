"""Paradee: an 8M-parameter English text-to-speech model distilled from Kokoro-82M, with four of its voices."""
from .tts import Paradee, SAMPLE_RATE, VOICES

__all__ = ["Paradee", "SAMPLE_RATE", "VOICES"]
__version__ = "1.2.1"
