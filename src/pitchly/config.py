"""Configuration for Pitchly detector"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from pitchly.types import FeatureFlags


class PitchlyConfig(BaseModel):
    """
    Configuration for PitchlyDetector.

    Default values are optimized for maximum quality frequency detection.
    No filtering, no limits - captures exactly what the microphone hears.

    Attributes:
        sample_rate: Audio sample rate in Hz (default: 48000)
        buffer_duration: Audio buffer duration in seconds (default: 0.2)
        hop_length: Samples between analysis frames (default: 512)
        fft_size: FFT size for spectral analysis (default: 8192)
        pitch_algorithm: Aubio pitch algorithm - "yin", "yinfast", or "yinfft"
        pitch_tolerance: YIN algorithm tolerance (default: 0.1)
        features: FeatureFlags controlling which features to extract
        num_harmonics: Number of harmonics to analyze (default: 16)
        envelope_history_size: Envelope history buffer size (default: 200)
        tuning_reference: A4 tuning reference in Hz (default: 440.0)

    Example:
        >>> config = PitchlyConfig(
        ...     sample_rate=48000,
        ...     features=FeatureFlags.BASIC,
        ... )
        >>> detector = PitchlyDetector(config)
    """

    model_config = ConfigDict(frozen=True, validate_assignment=True)

    sample_rate: int = Field(
        default=48000,
        ge=8000,
        le=192000,
        description="Audio sample rate in Hz",
    )
    """Audio sample rate in Hz. 48000 is professional standard"""

    buffer_duration: float = Field(
        default=0.2,
        gt=0.0,
        le=2.0,
        description="Buffer duration in seconds",
    )
    """Buffer duration in seconds. Larger = more stable low frequency detection"""

    hop_length: int = Field(
        default=512,
        ge=64,
        le=4096,
        description="Samples between analysis frames",
    )
    """Samples between analysis frames. 512 = ~10.7ms updates at 48kHz"""

    fft_size: int = Field(
        default=8192,
        ge=256,
        le=16384,
        description="FFT size for spectral analysis",
    )
    """FFT size. 8192 = ~5.9Hz resolution at 48kHz (excellent for all frequencies)"""

    pitch_algorithm: str = Field(
        default="yin",
        pattern="^(yin|yinfast|yinfft)$",
        description="Aubio pitch detection algorithm",
    )
    """Pitch algorithm: 'yin' is most accurate for monophonic pitch detection"""

    pitch_tolerance: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="YIN tolerance",
    )
    """YIN tolerance. Lower = stricter fundamental detection"""

    features: FeatureFlags = Field(
        default=FeatureFlags.FULL,
        description="Features to extract",
    )
    """FeatureFlags controlling which features to compute"""

    num_harmonics: int = Field(
        default=16,
        ge=1,
        le=32,
        description="Number of harmonics to analyze",
    )
    """Number of harmonics to analyze for harmonic features"""

    envelope_history_size: int = Field(
        default=200,
        ge=10,
        le=1000,
        description="Envelope history buffer size",
    )
    """Size of envelope history buffer for stability calculations"""

    tuning_reference: float = Field(
        default=440.0,
        ge=400.0,
        le=480.0,
        description="A4 tuning reference in Hz",
    )
    """A4 tuning reference frequency (440Hz standard, 442Hz Berlin, 443Hz Vienna)"""
