from importlib.metadata import version

from pitchly.config import PitchlyConfig
from pitchly.detector import PitchlyDetector
from pitchly.types import (
    AudioDetection,
    EnvelopeFeatures,
    FeatureFlags,
    HarmonicFeatures,
    SpectralFeatures,
    TimbreFeatures,
)


__version__: str = version("pitchly")
__all__: list[str] = [
    "PitchlyDetector",
    "PitchlyConfig",
    "FeatureFlags",
    "AudioDetection",
    "SpectralFeatures",
    "HarmonicFeatures",
    "EnvelopeFeatures",
    "TimbreFeatures",
]
