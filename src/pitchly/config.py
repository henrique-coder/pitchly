"""Configuration for Pitchly detector with full Pydantic validation"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from pitchly.types import FeatureFlags


class PitchlyConfig(BaseModel):
    """
    Configuration for PitchlyDetector with full Pydantic validation

    Optimized defaults for universal frequency detection across all instruments and audio sources:
    - sample_rate: 48000 Hz (professional audio standard, supports frequencies up to 24kHz Nyquist)
    - buffer_duration: 0.093 seconds (~4480 samples at 48kHz, provides stable low-frequency detection)
    - hop_length: 1024 samples (~21ms hop at 48kHz, balanced update rate)
    - fft_size: 2048 (provides ~23Hz frequency resolution at 48kHz)
    - pitch_algorithm: yin (most accurate pitch detection algorithm)
    - pitch_tolerance: 0.8 (permissive, captures all pitched sounds)
    - pitch_min_confidence: 0.1 (low threshold to detect all sounds including quiet ones)
    - freq_min: 16.0 Hz (supports deepest instruments: pipe organ, contrabass, tuba pedal notes)
    - freq_max: 20000.0 Hz (full human audible range, supports piccolo, violin harmonics, ultrasonic)

    Universal Instrument Coverage:
    - Deep bass: Pipe organ 16Hz (C0), Contrabass 41Hz (E1), Tuba 29Hz (Bb0)
    - Bass: Bass guitar 41Hz-392Hz, Cello 65Hz-1047Hz, Bassoon 58Hz-587Hz
    - Mid: Guitar 82Hz-1175Hz, Clarinet 146Hz-1568Hz, French horn 87Hz-880Hz
    - High: Violin 196Hz-3136Hz, Flute 262Hz-2349Hz, Oboe 233Hz-1568Hz
    - Extreme high: Piccolo 587Hz-4186Hz (D5-C8), Violin harmonics up to 12kHz+
    - Harmonics: All instruments produce harmonics extending into ultrasonic range (>20kHz)

    Library Limits:
    - Aubio YIN: Tested and reliable from 20Hz to Nyquist frequency (sample_rate / 2)
    - Librosa pYIN: Supports fmin=20Hz to fmax=Nyquist (up to 96kHz sample rate)
    - SoundDevice: Supports sample rates from 8kHz to 192kHz (hardware dependent)
    - FFT analysis: Limited by Nyquist frequency (must be < sample_rate / 2)

    These settings provide:
    - Universal frequency detection (16Hz-20kHz range covers all instruments + harmonics)
    - Excellent pitch accuracy across all frequency ranges
    - Good temporal resolution (~21ms between updates)
    - Stable detection for low frequencies (large buffer crucial for bass)
    - Fast processing (~5-15ms analysis time)
    """

    model_config = ConfigDict(
        frozen=True,
        validate_assignment=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    # Audio capture settings (optimized for universal frequency detection)
    sample_rate: int = Field(
        default=48000,
        ge=8000,
        le=192000,
        description="Audio sample rate in Hz - 48000 is professional standard, supports up to 24kHz Nyquist frequency",
    )

    buffer_duration: float = Field(
        default=0.093,
        gt=0.0,
        le=1.0,
        description="Buffer duration in seconds - larger buffers improve low-frequency detection stability (critical for bass instruments like tuba, organ)",
    )

    hop_length: int = Field(
        default=1024,
        ge=64,
        le=4096,
        description="Samples between analysis frames - higher = more stable detection, lower = faster updates (trade-off between latency and stability)",
    )

    # Processing settings
    fft_size: int = Field(
        default=2048,
        ge=256,
        le=8192,
        description="FFT size for spectral analysis - larger = better frequency resolution (critical for distinguishing low frequencies, e.g., 16Hz organ from 20Hz)",
    )

    # Pitch detection settings
    pitch_algorithm: str = Field(
        default="yin",
        pattern="^(yin|yinfast|yinfft)$",
        description="Aubio pitch detection algorithm: yin (most accurate, best for all instruments), yinfast (faster, good for most cases), yinfft (frequency domain, experimental)",
    )

    pitch_tolerance: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="YIN tolerance (0.0-1.0) - controls pitch detection strictness. 0.8 is permissive (detects all pitched sounds), 0.2 is strict (only clear pitches)",
    )

    pitch_min_confidence: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold - 0.1 is low (detects quiet/weak pitches, good for tuning), 0.5+ is strict (only strong clear pitches)",
    )

    use_pyin_fallback: bool = Field(
        default=False,
        description="Enable pYIN fallback for difficult pitch detection (adds ~50-100ms latency but improves accuracy for complex/noisy sounds)",
    )

    # Frequency validation (supports full audible range + all instruments)
    freq_min: float = Field(
        default=16.0,
        ge=10.0,
        description="Minimum valid frequency in Hz - 16Hz supports deepest organ pipes (C0), 20Hz is human hearing limit, 29Hz covers tuba pedal notes",
    )

    freq_max: float = Field(
        default=20000.0,
        le=24000.0,
        description="Maximum valid frequency in Hz - 20kHz is human hearing limit, covers all instrument fundamentals + harmonics (piccolo C8 = 4186Hz, harmonics extend to 20kHz+)",
    )

    # Tuning reference (for musical applications)
    tuning_reference: float = Field(
        default=440.0,
        ge=400.0,
        le=480.0,
        description="A4 tuning reference frequency in Hz - standard pitch is 440Hz (orchestras may use 442Hz Berlin, 443Hz Vienna, 415Hz baroque)",
    )

    # Feature extraction control
    features: FeatureFlags = Field(
        default=FeatureFlags.FULL,
        description="Feature flags controlling what gets extracted - BASIC (frequency only), QUALITY (frequency+spectral+harmonic), FULL (everything including timbre)",
    )

    # Performance tuning
    num_harmonics: int = Field(
        default=8,
        ge=1,
        le=16,
        description="Number of harmonics to analyze - 8 is good balance, 16 for detailed timbre analysis (instruments typically have 10-20 significant harmonics)",
    )

    envelope_history_size: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Number of envelope samples to keep - 100 is good balance (~2 seconds at 48kHz with 1024 hop), larger for better envelope tracking",
    )

    @field_validator("freq_max")
    @classmethod
    def validate_freq_max(cls, v: float, info) -> float:
        """
        Validate that freq_max is greater than freq_min and within Nyquist limit

        The Nyquist frequency is half the sample rate and represents the maximum
        frequency that can be accurately represented in digital audio.
        """
        freq_min = info.data.get("freq_min", 16.0)
        sample_rate = info.data.get("sample_rate", 48000)

        if v <= freq_min:
            raise ValueError(f"freq_max ({v}Hz) must be greater than freq_min ({freq_min}Hz)")

        nyquist = sample_rate / 2
        if v > nyquist:
            raise ValueError(
                f"freq_max ({v}Hz) cannot exceed Nyquist frequency ({nyquist}Hz = sample_rate/2). "
                f"Increase sample_rate to {int(v * 2)} or lower freq_max to {nyquist}"
            )

        return v

    @field_validator("buffer_duration")
    @classmethod
    def validate_buffer_duration(cls, v: float, info) -> float:
        """
        Validate that buffer duration results in reasonable sample count

        Minimum 512 samples required for meaningful frequency analysis.
        Larger buffers improve low-frequency detection accuracy.
        """
        sample_rate = info.data.get("sample_rate", 48000)
        buffer_samples = int(v * sample_rate)

        if buffer_samples < 512:
            raise ValueError(
                f"buffer_duration ({v}s) too small: results in {buffer_samples} samples (need >= 512). "
                f"Increase to at least {512 / sample_rate:.4f} seconds"
            )

        return v

    @field_validator("hop_length")
    @classmethod
    def validate_hop_length(cls, v: int, info) -> int:
        """
        Validate that hop_length is reasonable for the buffer size

        hop_length should not exceed buffer size to ensure overlap between frames.
        """
        sample_rate = info.data.get("sample_rate", 48000)
        buffer_duration = info.data.get("buffer_duration", 0.093)
        buffer_samples = int(buffer_duration * sample_rate)

        if v > buffer_samples:
            raise ValueError(
                f"hop_length ({v} samples) cannot be larger than buffer size ({buffer_samples} samples). "
                f"Decrease hop_length or increase buffer_duration"
            )

        return v
