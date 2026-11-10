"""Paradee: an 8M-parameter text-to-speech model distilled from Kokoro-82M, with eight of its voices in English,
Hindi, French and Spanish."""
from .tts import Paradee, SAMPLE_RATE, VOICES, LANGUAGES

__all__ = ["Paradee", "SAMPLE_RATE", "VOICES", "LANGUAGES"]
__version__ = "1.3.1"
