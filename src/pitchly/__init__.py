# Standard modules
from importlib.metadata import version

# Local modules
from .core import AudioDetection, EnvelopeFeatures, HarmonicFeatures, PitchlyDetector, SpectralFeatures, TimbreFeatures


__version__: str = version("pitchly")
__all__ = ["AudioDetection", "EnvelopeFeatures", "HarmonicFeatures", "PitchlyDetector", "SpectralFeatures", "TimbreFeatures"]
