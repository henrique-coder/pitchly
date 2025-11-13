"""Type definitions and feature configuration for Pitchly"""

from __future__ import annotations

from enum import Flag, auto
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field


if TYPE_CHECKING:
    pass


class FeatureFlags(Flag):
    """Feature extraction flags - control what gets computed"""

    # Core features (always enabled)
    FREQUENCY = auto()  # Fundamental frequency detection
    CONFIDENCE = auto()  # Pitch confidence
    ENERGY = auto()  # RMS and peak energy

    # Optional feature groups
    SPECTRAL = auto()  # Spectral analysis (centroid, spread, etc)
    HARMONIC = auto()  # Harmonic structure analysis
    ENVELOPE = auto()  # Amplitude envelope analysis
    TIMBRE = auto()  # Timbral characteristics (MFCC, chroma)
    RHYTHM = auto()  # Onset and tempo detection

    # Convenience combinations
    BASIC = FREQUENCY | CONFIDENCE | ENERGY
    QUALITY = BASIC | SPECTRAL | HARMONIC
    FULL = BASIC | SPECTRAL | HARMONIC | ENVELOPE | TIMBRE | RHYTHM


class SpectralFeatures(BaseModel):
    """Spectral analysis features"""

    model_config = ConfigDict(frozen=True, validate_assignment=False)

    centroid: float = Field(ge=0.0, description="Spectral centroid in Hz")
    spread: float = Field(ge=0.0, description="Spectral spread")
    rolloff_85: float = Field(ge=0.0, description="85% energy rolloff frequency")
    rolloff_95: float = Field(ge=0.0, description="95% energy rolloff frequency")
    flatness: float = Field(ge=0.0, le=1.0, description="Spectral flatness")
    crest: float = Field(ge=0.0, description="Spectral crest factor")
    entropy: float = Field(ge=0.0, description="Spectral entropy")
    slope: float = Field(description="Spectral slope")
    brightness: float = Field(ge=0.0, le=1.0, description="High frequency energy ratio")
    warmth: float = Field(ge=0.0, le=1.0, description="Low-mid frequency energy ratio")
    sharpness: float = Field(ge=0.0, description="High frequency emphasis")


class HarmonicFeatures(BaseModel):
    """Harmonic structure analysis"""

    model_config = ConfigDict(frozen=True, validate_assignment=False)

    fundamental_strength: float = Field(ge=0.0, description="Fundamental frequency amplitude")
    weights: tuple[float, ...] = Field(description="Harmonic amplitude weights")
    frequencies: tuple[float, ...] = Field(description="Detected harmonic frequencies in Hz")
    deviation: float = Field(ge=0.0, description="Average harmonic frequency deviation")
    inharmonicity: float = Field(ge=0.0, description="Inharmonicity coefficient")
    noise_ratio: float = Field(description="Harmonic to noise ratio")


class EnvelopeFeatures(BaseModel):
    """Amplitude envelope analysis"""

    model_config = ConfigDict(frozen=True, validate_assignment=False)

    attack_time: float = Field(ge=0.0, description="Attack time in seconds")
    decay_slope: float = Field(description="Decay slope coefficient")
    stability: float = Field(ge=0.0, le=1.0, description="Amplitude envelope stability")
    modulation_rate: float = Field(ge=0.0, description="Amplitude modulation frequency")


class TimbreFeatures(BaseModel):
    """Timbral characteristics"""

    model_config = ConfigDict(frozen=True, validate_assignment=False)

    roughness: float = Field(ge=0.0, description="Sensory dissonance measure")
    mfcc: tuple[float, ...] = Field(description="Mel-frequency cepstral coefficients")
    chroma: tuple[float, ...] = Field(description="Chromagram features")
    spectral_contrast: tuple[float, ...] = Field(description="Spectral contrast")


class AudioDetection(BaseModel):
    """Complete audio detection result"""

    model_config = ConfigDict(frozen=True, validate_assignment=False)

    # Core measurements (always present)
    timestamp: float = Field(description="Detection timestamp in seconds")
    frequency: float = Field(ge=0.0, description="Fundamental frequency in Hz")
    confidence: float = Field(ge=0.0, le=1.0, description="Pitch detection confidence")
    voiced: bool = Field(description="Voice/sound activity detection")

    # Energy measurements (always present)
    rms_energy: float = Field(ge=0.0, description="RMS energy level")
    peak_amplitude: float = Field(ge=0.0, description="Peak amplitude")
    dynamic_range: float = Field(ge=0.0, description="Peak to RMS ratio")
    snr_db: float = Field(description="Signal to noise ratio in dB")

    # Basic audio characteristics (always present)
    zero_crossing_rate: float = Field(ge=0.0, le=1.0, description="Zero crossing rate")
    autocorr_peak: float = Field(ge=0.0, le=1.0, description="Autocorrelation peak strength")
    periodicity: float = Field(ge=0.0, le=1.0, description="Signal periodicity measure")

    # Optional feature groups (None if disabled)
    spectral: SpectralFeatures | None = Field(None, description="Spectral analysis features")
    harmonic: HarmonicFeatures | None = Field(None, description="Harmonic structure analysis")
    envelope: EnvelopeFeatures | None = Field(None, description="Amplitude envelope analysis")
    timbre: TimbreFeatures | None = Field(None, description="Timbral characteristics")

    # Rhythm features (None if disabled)
    onset_strength: float | None = Field(None, ge=0.0, description="Onset detection strength")
    tempo_bpm: float | None = Field(None, ge=0.0, description="Estimated tempo in BPM")

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return self.model_dump()

    def to_json(self) -> str:
        """Convert to JSON string"""
        return self.model_dump_json()
