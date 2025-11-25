"""Configuration for Pitchly detector with full Pydantic validation"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from pitchly.types import FeatureFlags


class PitchlyConfig(BaseModel):
    """
    Configuration for PitchlyDetector with full Pydantic validation

    Optimized defaults for universal frequency detection with minimal latency:
    - sample_rate: 48000 Hz (professional audio standard, supports frequencies up to 24kHz Nyquist)
    - buffer_duration: 0.128 seconds (~6144 samples = 14 cycles at 440Hz, stable detection with low latency)
    - hop_length: 512 samples (~10.7ms updates at 48kHz, fast response without compromising stability)
    - fft_size: 4096 (provides ~11.7Hz frequency resolution at 48kHz, excellent bass discrimination)
    - pitch_algorithm: yin (most accurate pitch detection algorithm)
    - pitch_tolerance: 0.15 (optimal for fundamental detection, rejects harmonics)
    - pitch_min_confidence: 0.2 (balanced threshold for clean pitch detection)
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
    - Fast temporal resolution (~10.7ms between updates, 2x faster than 1024 hop)
    - Stable detection optimized for 82Hz-4kHz range (guitar to flute)
    - Low total latency: ~139ms (128ms buffer + 10.7ms hop + ~3ms processing)
    - Efficient CPU usage: ~5-10ms processing time per frame
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
        default=0.128,
        gt=0.0,
        le=1.0,
        description="Buffer duration in seconds (default 128ms = ~6144 samples at 48kHz). Minimum for stable detection: 50ms for bass (82Hz), 15ms for mid (440Hz), 5ms for treble (2kHz+). Larger = more stable but higher latency.",
    )

    hop_length: int = Field(
        default=512,
        ge=64,
        le=4096,
        description="Samples between analysis frames (default 512 = 10.7ms at 48kHz). Lower = faster response but more CPU (256=5.3ms, 512=10.7ms, 1024=21.3ms). Must be <= buffer size.",
    )

    # Processing settings
    fft_size: int = Field(
        default=4096,
        ge=256,
        le=8192,
        description="FFT size for spectral analysis (default 4096 = 11.7Hz resolution at 48kHz). Larger = better bass resolution (2048=23Hz, 4096=11.7Hz, 8192=5.9Hz). Minimal CPU impact.",
    )

    # Pitch detection settings
    pitch_algorithm: str = Field(
        default="yin",
        pattern="^(yin|yinfast|yinfft)$",
        description="Aubio pitch detection algorithm: yin (most accurate, best for all instruments), yinfast (faster, good for most cases), yinfft (frequency domain, experimental)",
    )

    pitch_tolerance: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="YIN tolerance (0.0-1.0) - controls pitch detection strictness. 0.15 is optimal (detects fundamental accurately), 0.8 is too permissive (catches harmonics), 0.05 is too strict (misses weak signals)",
    )

    pitch_min_confidence: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold - 0.2 is balanced (detects clean pitches, rejects noise), 0.1 is too low (accepts harmonics/noise), 0.5+ is strict (misses weak but valid signals)",
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
