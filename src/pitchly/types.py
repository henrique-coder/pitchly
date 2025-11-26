"""Type definitions for Pitchly"""

from __future__ import annotations

from enum import Flag, auto

from pydantic import BaseModel, ConfigDict, Field


class FeatureFlags(Flag):
    """
    Feature extraction flags - control what gets computed.

    Use bitwise OR (|) to combine flags for custom feature sets.

    Example:
        >>> flags = FeatureFlags.FREQUENCY | FeatureFlags.SPECTRAL
    """

    FREQUENCY = auto()
    """Fundamental frequency detection using YIN algorithm"""

    CONFIDENCE = auto()
    """Pitch detection confidence score (0.0 to 1.0)"""

    ENERGY = auto()
    """RMS energy, peak amplitude, and dynamic range"""

    SPECTRAL = auto()
    """Spectral analysis: centroid, spread, rolloff, flatness, brightness"""

    HARMONIC = auto()
    """Harmonic structure: fundamental strength, harmonic weights, inharmonicity"""

    ENVELOPE = auto()
    """Amplitude envelope: attack time, decay slope, stability"""

    TIMBRE = auto()
    """Timbral features: MFCC, chroma, spectral contrast, roughness"""

    RHYTHM = auto()
    """Rhythm detection: onset strength, tempo BPM"""

    BASIC = FREQUENCY | CONFIDENCE | ENERGY
    """Basic features: frequency, confidence, and energy metrics (fastest)"""

    QUALITY = BASIC | SPECTRAL | HARMONIC
    """Quality features: basic + spectral + harmonic analysis"""

    FULL = BASIC | SPECTRAL | HARMONIC | ENVELOPE | TIMBRE | RHYTHM
    """All features enabled (most comprehensive, slower)"""


class SpectralFeatures(BaseModel):
    """
    Spectral analysis features extracted from the frequency spectrum.

    Contains metrics describing the frequency distribution of the audio signal.

    Attributes:
        centroid: Center of mass of the spectrum in Hz (brightness indicator)
        spread: Standard deviation around the centroid (spectral width)
        rolloff_85: Frequency below which 85% of energy is contained
        rolloff_95: Frequency below which 95% of energy is contained
        flatness: Ratio of geometric to arithmetic mean (0=tonal, 1=noisy)
        crest: Peak to average ratio (dynamic range indicator)
        entropy: Spectral entropy (randomness/complexity measure)
        slope: Linear regression slope of the spectrum
        brightness: Energy ratio above 1500Hz
        warmth: Energy ratio in 200-800Hz range
        sharpness: Energy ratio above 2000Hz
    """

    model_config = ConfigDict(frozen=True)

    centroid: float = Field(description="Spectral centroid in Hz")
    spread: float = Field(description="Spectral spread")
    rolloff_85: float = Field(description="85% energy rolloff frequency")
    rolloff_95: float = Field(description="95% energy rolloff frequency")
    flatness: float = Field(description="Spectral flatness (0=tonal, 1=noisy)")
    crest: float = Field(description="Spectral crest factor")
    entropy: float = Field(description="Spectral entropy")
    slope: float = Field(description="Spectral slope")
    brightness: float = Field(description="High frequency energy ratio (>1500Hz)")
    warmth: float = Field(description="Low-mid frequency energy ratio (200-800Hz)")
    sharpness: float = Field(description="High frequency emphasis (>2000Hz)")


class HarmonicFeatures(BaseModel):
    """
    Harmonic structure analysis of the audio signal.

    Analyzes the relationship between the fundamental frequency and its harmonics.

    Attributes:
        fundamental_strength: Amplitude of the fundamental frequency
        weights: Relative amplitudes of each harmonic (normalized to fundamental)
        frequencies: Detected frequencies of each harmonic in Hz
        deviation: Average deviation of harmonics from ideal positions
        inharmonicity: Weighted sum of harmonic deviations (piano strings have high values)
        noise_ratio: Ratio of harmonic energy to noise energy
    """

    model_config = ConfigDict(frozen=True)

    fundamental_strength: float = Field(description="Fundamental frequency amplitude")
    weights: tuple[float, ...] = Field(description="Harmonic amplitude weights (relative to fundamental)")
    frequencies: tuple[float, ...] = Field(description="Detected harmonic frequencies in Hz")
    deviation: float = Field(description="Average harmonic frequency deviation")
    inharmonicity: float = Field(description="Inharmonicity coefficient")
    noise_ratio: float = Field(description="Harmonic to noise ratio")


class EnvelopeFeatures(BaseModel):
    """
    Amplitude envelope analysis of the audio signal.

    Describes how the volume of the sound changes over time.

    Attributes:
        attack_time: Time to reach peak amplitude (0.0 to 1.0, relative)
        decay_slope: Rate of amplitude decrease after peak
        stability: Consistency of amplitude over time (0=unstable, 1=stable)
        modulation_rate: Frequency of amplitude modulation (vibrato, tremolo)
    """

    model_config = ConfigDict(frozen=True)

    attack_time: float = Field(description="Attack time (relative)")
    decay_slope: float = Field(description="Decay slope coefficient")
    stability: float = Field(description="Amplitude envelope stability (0-1)")
    modulation_rate: float = Field(description="Amplitude modulation frequency in Hz")


class TimbreFeatures(BaseModel):
    """
    Timbral characteristics of the audio signal.

    Features that describe the "color" or "quality" of the sound.

    Attributes:
        roughness: Sensory dissonance from beating frequencies
        mfcc: 13 Mel-frequency cepstral coefficients (voice/instrument fingerprint)
        chroma: 12 pitch class energies (C, C#, D, ... B)
        spectral_contrast: 7 bands of peak-valley difference
    """

    model_config = ConfigDict(frozen=True)

    roughness: float = Field(description="Sensory dissonance measure")
    mfcc: tuple[float, ...] = Field(description="13 Mel-frequency cepstral coefficients")
    chroma: tuple[float, ...] = Field(description="12 pitch class energies (C to B)")
    spectral_contrast: tuple[float, ...] = Field(description="7 spectral contrast bands")


class AudioDetection(BaseModel):
    """
    Complete audio detection result from a single analysis frame.

    Contains all extracted features from the current audio moment.
    Core features are always present, optional features depend on FeatureFlags.

    Attributes:
        timestamp: Monotonic timestamp of the detection
        frequency: Detected fundamental frequency in Hz (None if no valid pitch detected)
        confidence: Pitch detection confidence (0.0 to 1.0)
        rms_energy: Root mean square energy level
        peak_amplitude: Maximum absolute amplitude
        dynamic_range: Ratio of peak to RMS (crest factor)
        snr_db: Signal to noise ratio in decibels
        zero_crossing_rate: Rate of sign changes (noise indicator)
        autocorr_peak: Autocorrelation peak strength (periodicity)
        periodicity: Signal periodicity measure (0=aperiodic, 1=periodic)
        spectral: Spectral features (None if SPECTRAL flag disabled)
        harmonic: Harmonic features (None if HARMONIC flag disabled)
        envelope: Envelope features (None if ENVELOPE flag disabled)
        timbre: Timbre features (None if TIMBRE flag disabled)
        onset_strength: Onset detection strength (None if RHYTHM flag disabled)
        tempo_bpm: Estimated tempo in BPM (None if RHYTHM flag disabled)
    """

    model_config = ConfigDict(frozen=True)

    timestamp: float = Field(description="Detection timestamp (monotonic)")
    frequency: float | None = Field(description="Detected frequency in Hz (None if invalid)")
    confidence: float = Field(description="Detection confidence (0-1)")

    rms_energy: float = Field(description="RMS energy level")
    peak_amplitude: float = Field(description="Peak amplitude")
    dynamic_range: float = Field(description="Peak to RMS ratio")
    snr_db: float = Field(description="Signal to noise ratio in dB")

    zero_crossing_rate: float = Field(description="Zero crossing rate")
    autocorr_peak: float = Field(description="Autocorrelation peak strength")
    periodicity: float = Field(description="Signal periodicity measure")

    spectral: SpectralFeatures | None = Field(None, description="Spectral features")
    harmonic: HarmonicFeatures | None = Field(None, description="Harmonic features")
    envelope: EnvelopeFeatures | None = Field(None, description="Envelope features")
    timbre: TimbreFeatures | None = Field(None, description="Timbre features")

    onset_strength: float | None = Field(None, description="Onset detection strength")
    tempo_bpm: float | None = Field(None, description="Estimated tempo in BPM")

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return self.model_dump()

    def to_json(self) -> str:
        """Convert to JSON string"""
        return self.model_dump_json()
