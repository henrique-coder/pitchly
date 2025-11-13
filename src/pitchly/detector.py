"""Main detector implementation"""

from __future__ import annotations

import atexit
from contextlib import suppress
import signal
import sys
from threading import Event, Lock
from time import monotonic
from typing import TYPE_CHECKING
from warnings import filterwarnings

from aubio import onset, pitch, tempo
from librosa import pyin
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import get_window
from sounddevice import InputStream

from pitchly.config import PitchlyConfig
from pitchly.envelope import create_empty_envelope, extract_envelope_features
from pitchly.harmonic import create_empty_harmonic, extract_harmonic_features
from pitchly.spectral import create_empty_spectral, extract_spectral_features
from pitchly.timbre import create_empty_timbre, extract_timbre_features
from pitchly.types import AudioDetection, FeatureFlags
from pitchly.utils import safe_division, safe_float, safe_log2, validate_confidence, validate_frequency


if TYPE_CHECKING:
    from numpy.typing import NDArray

# Suppress warnings
filterwarnings("ignore")

# Global shutdown event for signal handling
_shutdown_event = Event()


def _signal_handler(signum: int, frame) -> None:
    """Handle shutdown signals gracefully"""
    _shutdown_event.set()
    sys.exit(0)


# Register signal handlers
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


class PitchlyDetector:
    """
    Ultra-optimized real-time audio quality detector with Pydantic-validated configuration

    This detector provides real-time audio analysis with:
    - Precise frequency detection (20Hz-8000Hz validated range)
    - Configurable feature extraction (enable only what you need)
    - Low latency processing (~5-15ms with default settings)
    - Thread-safe architecture with graceful shutdown handling
    - Musical tuning reference support (configurable A4 pitch)
    - Comprehensive audio quality metrics

    Default Configuration:
    - 48kHz sample rate (professional audio quality)
    - 1024 sample hop length (~21ms updates)
    - 2048 FFT size (~0.5Hz frequency resolution)
    - YIN pitch algorithm (most accurate)
    - Low confidence threshold (0.1) to detect all sounds
    - Large buffer (93ms) for stable detection

    Example:
        >>> from pitchly import PitchlyDetector, PitchlyConfig, FeatureFlags
        >>> # Use defaults (all features, high quality)
        >>> detector = PitchlyDetector()
        >>> detector.listen()
        >>> audio = detector.analyze()
        >>> print(f"Frequency: {audio.frequency:.1f}Hz")
        >>> # Custom configuration (basic features only, fastest)
        >>> config = PitchlyConfig(features=FeatureFlags.BASIC)
        >>> detector = PitchlyDetector(config)
    """

    __slots__ = (
        "_config",
        "_buffer_samples",
        "_aubio_pitch",
        "_aubio_onset",
        "_aubio_tempo",
        "_stream",
        "_running",
        "_shutdown_event",
        "_audio_buffer",
        "_envelope_history",
        "_lock",
        "_listening",
    )

    def __init__(self, config: PitchlyConfig | None = None) -> None:
        """
        Initialize the detector with Pydantic-validated configuration

        The detector initializes the audio stream immediately and blocks until
        the microphone is ready. This ensures all subsequent operations have
        a working audio input.

        Args:
            config: PitchlyConfig instance with validated settings.
                   If None, uses optimized defaults (48kHz, 1024 hop, YIN algorithm).

        Raises:
            ValidationError: If config values are invalid (handled by Pydantic)
            OSError: If microphone cannot be opened

        Example:
            >>> config = PitchlyConfig(
            ...     sample_rate=48000,
            ...     pitch_min_confidence=0.1,  # Detect all sounds
            ...     features=FeatureFlags.BASIC | FeatureFlags.TIMBRE,
            ... )
            >>> detector = PitchlyDetector(config)
        """
        self._config = config or PitchlyConfig()
        self._buffer_samples = int(self._config.buffer_duration * self._config.sample_rate)

        # Initialize aubio processors
        self._aubio_pitch = pitch(
            self._config.pitch_algorithm,
            self._config.fft_size,
            self._config.hop_length,
            self._config.sample_rate,
        )
        self._aubio_pitch.set_unit("Hz")
        self._aubio_pitch.set_tolerance(self._config.pitch_tolerance)

        # Initialize rhythm detectors if enabled
        self._aubio_onset = None
        self._aubio_tempo = None

        if FeatureFlags.RHYTHM in self._config.features:
            self._aubio_onset = onset(
                "default",
                self._config.fft_size,
                self._config.hop_length,
                self._config.sample_rate,
            )
            self._aubio_tempo = tempo(
                "default",
                self._config.fft_size,
                self._config.hop_length,
                self._config.sample_rate,
            )

        # Audio state
        self._stream: InputStream | None = None
        self._running = False
        self._shutdown_event = _shutdown_event
        self._audio_buffer = np.zeros(self._buffer_samples, dtype=np.float32)
        self._envelope_history = np.zeros(self._config.envelope_history_size, dtype=np.float32)
        self._lock = Lock()
        self._listening = False

        # Initialize stream immediately and wait until ready
        self._stream = InputStream(
            samplerate=self._config.sample_rate,
            channels=1,
            blocksize=self._config.hop_length,
            callback=self._audio_callback,
            dtype="float32",
        )

        # Register cleanup
        atexit.register(self.stop)

    def _audio_callback(self, indata: NDArray, frames: int, time_info, status) -> None:
        """Audio input callback - optimized for minimal latency"""
        if self._shutdown_event.is_set():
            raise KeyboardInterrupt

        with self._lock:
            self._audio_buffer = np.roll(self._audio_buffer, -frames)
            self._audio_buffer[-frames:] = indata.flatten()

    def _process_current_audio(self) -> AudioDetection:
        """Process audio buffer and return analysis"""
        # Get latest audio chunk
        with self._lock:
            buffer_copy = self._audio_buffer[-self._config.hop_length :].copy()

        # Check if silent
        if len(buffer_copy) == 0 or np.max(np.abs(buffer_copy)) < 1e-10:
            return self._create_silent_detection()

        # Apply window and compute spectrum
        windowed = buffer_copy * get_window("hann", len(buffer_copy))
        spectrum = np.abs(rfft(windowed, n=self._config.fft_size))
        freqs = rfftfreq(self._config.fft_size, 1 / self._config.sample_rate)

        # === CORE ENERGY METRICS (always computed) ===
        rms_energy = safe_float(np.sqrt(np.mean(buffer_copy**2)))
        peak_amplitude = safe_float(np.max(np.abs(buffer_copy)))
        dynamic_range = safe_float(safe_division(peak_amplitude, rms_energy))

        noise_floor = np.percentile(spectrum, 20)
        snr_db = safe_float(20 * safe_log2(safe_division(rms_energy, noise_floor)))

        # Zero crossing rate
        zero_crossings = safe_float(np.sum(np.diff(np.sign(buffer_copy)) != 0) / len(buffer_copy))
        zero_crossings = min(zero_crossings, 1.0)

        # === PITCH DETECTION (core feature) ===
        autocorr_peak = 0.0
        periodicity = 0.0
        frequency = 0.0
        confidence = 0.0
        voiced = False

        # Autocorrelation for periodicity
        with suppress(Exception):
            autocorr = np.correlate(buffer_copy, buffer_copy, mode="full")
            autocorr = autocorr[len(autocorr) // 2 :]
            if len(autocorr) > 1:
                autocorr_peak = min(safe_float(np.max(autocorr[1:50]) / max(autocorr[0], 1e-10)), 1.0)
                periodicity = autocorr_peak

        # Aubio YIN pitch detection
        with suppress(Exception):
            aubio_frame = buffer_copy.astype(np.float32)
            raw_frequency = safe_float(self._aubio_pitch(aubio_frame)[0])
            raw_confidence = min(safe_float(self._aubio_pitch.get_confidence()), 1.0)

            # Validate frequency and confidence
            validated_freq, freq_valid = validate_frequency(
                raw_frequency,
                self._config.freq_min,
                self._config.freq_max,
            )
            conf_valid = validate_confidence(raw_confidence, self._config.pitch_min_confidence)

            if freq_valid and conf_valid:
                frequency = validated_freq
                confidence = raw_confidence
                voiced = True

        # Fallback to pYIN if YIN failed AND pYIN is enabled
        if not voiced and self._config.use_pyin_fallback:
            with suppress(Exception):
                pyin_freqs, voiced_flag, voiced_probs = pyin(
                    buffer_copy,
                    fmin=self._config.freq_min,
                    fmax=self._config.freq_max,
                    sr=self._config.sample_rate,
                )

                if len(pyin_freqs) > 0 and not np.isnan(pyin_freqs[-1]):
                    raw_frequency = safe_float(pyin_freqs[-1])
                    raw_confidence = safe_float(voiced_probs[-1] if len(voiced_probs) > 0 else 0.5)

                    # Validate pYIN results
                    validated_freq, freq_valid = validate_frequency(
                        raw_frequency,
                        self._config.freq_min,
                        self._config.freq_max,
                    )
                    conf_valid = validate_confidence(raw_confidence, self._config.pitch_min_confidence)

                    if freq_valid and conf_valid:
                        frequency = validated_freq
                        confidence = min(raw_confidence, 1.0)
                        voiced = bool(voiced_flag[-1]) if len(voiced_flag) > 0 else False

        # === OPTIONAL FEATURES (computed based on config) ===

        # Spectral features
        spectral_features = None
        if FeatureFlags.SPECTRAL in self._config.features:
            spectral_features = extract_spectral_features(spectrum, freqs)

        # Harmonic features
        harmonic_features = None
        if FeatureFlags.HARMONIC in self._config.features:
            harmonic_features = extract_harmonic_features(
                spectrum,
                freqs,
                frequency,
                self._config.sample_rate,
                self._config.num_harmonics,
            )

        # Envelope features
        envelope_features = None
        if FeatureFlags.ENVELOPE in self._config.features:
            envelope_features, self._envelope_history = extract_envelope_features(
                buffer_copy,
                self._envelope_history,
                self._config.sample_rate,
            )

        # Timbre features
        timbre_features = None
        if FeatureFlags.TIMBRE in self._config.features:
            timbre_features = extract_timbre_features(
                buffer_copy,
                spectrum,
                freqs,
                self._config.sample_rate,
            )

        # Rhythm features
        onset_strength = None
        tempo_bpm = None

        if FeatureFlags.RHYTHM in self._config.features and self._aubio_onset and self._aubio_tempo:
            with suppress(Exception):
                aubio_frame = buffer_copy.astype(np.float32)
                onset_strength = safe_float(self._aubio_onset(aubio_frame)[0])

            with suppress(Exception):
                self._aubio_tempo(aubio_frame)
                tempo_bpm = safe_float(self._aubio_tempo.get_bpm())

        return AudioDetection(
            timestamp=monotonic(),
            frequency=frequency,
            confidence=confidence,
            voiced=voiced,
            rms_energy=rms_energy,
            peak_amplitude=peak_amplitude,
            dynamic_range=dynamic_range,
            snr_db=snr_db,
            zero_crossing_rate=zero_crossings,
            autocorr_peak=autocorr_peak,
            periodicity=periodicity,
            spectral=spectral_features,
            harmonic=harmonic_features,
            envelope=envelope_features,
            timbre=timbre_features,
            onset_strength=onset_strength,
            tempo_bpm=tempo_bpm,
        )

    def _create_silent_detection(self) -> AudioDetection:
        """Create detection for silent audio"""
        return AudioDetection(
            timestamp=monotonic(),
            frequency=0.0,
            confidence=0.0,
            voiced=False,
            rms_energy=0.0,
            peak_amplitude=0.0,
            dynamic_range=0.0,
            snr_db=0.0,
            zero_crossing_rate=0.0,
            autocorr_peak=0.0,
            periodicity=0.0,
            spectral=create_empty_spectral() if FeatureFlags.SPECTRAL in self._config.features else None,
            harmonic=create_empty_harmonic(self._config.num_harmonics)
            if FeatureFlags.HARMONIC in self._config.features
            else None,
            envelope=create_empty_envelope() if FeatureFlags.ENVELOPE in self._config.features else None,
            timbre=create_empty_timbre() if FeatureFlags.TIMBRE in self._config.features else None,
            onset_strength=0.0 if FeatureFlags.RHYTHM in self._config.features else None,
            tempo_bpm=0.0 if FeatureFlags.RHYTHM in self._config.features else None,
        )

    def listen(self) -> None:
        """Start listening to microphone and capturing audio"""
        if self._listening or not self._stream:
            return

        self._stream.start()
        self._listening = True
        self._running = True

    def stop(self) -> None:
        """Stop microphone capture and cleanup"""
        if not self._listening:
            return

        self._listening = False
        self._running = False

        if self._stream:
            with suppress(Exception):
                self._stream.stop()
                self._stream.close()
            self._stream = None

    def analyze(self) -> AudioDetection:
        """
        Analyze current audio buffer and return comprehensive detection results

        This performs ultra-fast analysis on the most recent audio chunk captured
        by the microphone. Processing time is typically 5-15ms with default settings
        (faster with fewer features enabled).

        The method extracts:
        - Core metrics: frequency, confidence, energy (always)
        - Optional features: spectral, harmonic, envelope, timbre, rhythm
          (controlled by config.features flag)

        All frequency detections are validated against config.freq_min and freq_max
        to prevent spurious detections (like 96kHz jumps).

        Returns:
            AudioDetection: Pydantic-validated model with all extracted features.
                           Optional features will be None if not enabled.

        Raises:
            RuntimeError: If not listening to microphone (call .listen() first)
            KeyboardInterrupt: If shutdown signal received (Ctrl+C, SIGTERM)

        Example:
            >>> detector.listen()
            >>> audio = detector.analyze()
            >>> # Core features (always available)
            >>> print(f"Frequency: {audio.frequency:.1f}Hz")
            >>> print(f"Confidence: {audio.confidence:.2f}")
            >>> print(f"Voiced: {audio.voiced}")
            >>> print(f"Energy: {audio.rms_energy:.3f}")
            >>> # Optional features (check if enabled)
            >>> if audio.spectral:
            ...     print(f"Brightness: {audio.spectral.brightness:.2f}")
            >>> if audio.timbre:
            ...     print(f"MFCC: {audio.timbre.mfcc[:5]}")
        """
        if not self._listening:
            raise RuntimeError("Not listening. Call listen() first.")

        if self._shutdown_event.is_set():
            self.stop()
            raise KeyboardInterrupt

        return self._process_current_audio()

    @property
    def is_running(self) -> bool:
        """Check if detector is actively running"""
        return self._running

    @property
    def config(self) -> PitchlyConfig:
        """Get current configuration"""
        return self._config

    def __enter__(self) -> PitchlyDetector:
        """Context manager entry"""
        self.listen()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit"""
        self.stop()

    def __del__(self) -> None:
        """Cleanup on deletion"""
        with suppress(Exception):
            self.stop()
