"""Main detector implementation - captures raw audio without any filters"""

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
from pitchly.utils import safe_division, safe_float, safe_log2


if TYPE_CHECKING:
    from numpy.typing import NDArray

filterwarnings("ignore")

_shutdown_event = Event()


def _signal_handler(signum: int, frame) -> None:
    """Handle shutdown signals"""
    _shutdown_event.set()
    sys.exit(0)


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


class PitchlyDetector:
    """
    Real-time audio detector - captures exactly what the microphone hears.

    No filtering, no limits, no noise reduction - pure raw audio analysis.
    Returns the exact frequency detected at the moment of analysis.

    The detector uses the YIN algorithm for pitch detection, optimized for
    maximum quality. All filtering/limiting should be done by the end user.

    Attributes:
        is_running: True if the detector is actively listening
        config: Current PitchlyConfig configuration

    Example:
        >>> from pitchly import PitchlyDetector
        >>>
        >>> detector = PitchlyDetector()
        >>> detector.listen()
        >>> audio = detector.analyze()
        >>> if audio.frequency is not None:
        ...     print(f"Frequency: {audio.frequency:.1f}Hz")
        >>> detector.stop()
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
        Initialize detector with optional configuration.

        Args:
            config: PitchlyConfig instance. If None, uses optimized defaults.
        """
        self._config = config or PitchlyConfig()
        self._buffer_samples = int(self._config.buffer_duration * self._config.sample_rate)

        # Initialize aubio pitch detector with maximum quality settings
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

        # Initialize stream - raw audio capture
        self._stream = InputStream(
            samplerate=self._config.sample_rate,
            channels=1,
            blocksize=self._config.hop_length,
            callback=self._audio_callback,
            dtype="float32",
        )

        atexit.register(self.stop)

    def _audio_callback(self, indata: NDArray, frames: int, time_info, status) -> None:
        """Audio callback - stores raw audio without any processing"""
        if self._shutdown_event.is_set():
            raise KeyboardInterrupt

        with self._lock:
            self._audio_buffer = np.roll(self._audio_buffer, -frames)
            self._audio_buffer[-frames:] = indata.flatten()

    def _process_current_audio(self) -> AudioDetection:
        """Process current audio buffer and return raw analysis"""
        with self._lock:
            # Use full buffer for analysis
            buffer_copy = self._audio_buffer.copy()
            # Use hop_length chunk for pitch detection
            pitch_frame = self._audio_buffer[-self._config.hop_length :].copy()

        # Check for complete silence (no signal at all)
        max_amplitude = np.max(np.abs(buffer_copy))
        if len(buffer_copy) == 0 or max_amplitude < 1e-10:
            return self._create_silent_detection()

        # Compute spectrum for analysis using larger window
        analysis_size = min(len(buffer_copy), self._config.fft_size)
        analysis_window = buffer_copy[-analysis_size:]
        windowed = analysis_window * get_window("hann", len(analysis_window))
        spectrum = np.abs(rfft(windowed, n=self._config.fft_size))
        freqs = rfftfreq(self._config.fft_size, 1 / self._config.sample_rate)

        # === RAW ENERGY METRICS ===
        rms_energy = safe_float(np.sqrt(np.mean(buffer_copy**2)))
        peak_amplitude = safe_float(max_amplitude)
        dynamic_range = safe_float(safe_division(peak_amplitude, rms_energy))

        noise_floor = np.percentile(spectrum, 20)
        snr_db = safe_float(20 * safe_log2(safe_division(rms_energy, noise_floor)))

        # Zero crossing rate
        zero_crossings = safe_float(np.sum(np.diff(np.sign(buffer_copy)) != 0) / len(buffer_copy))

        # === RAW PITCH DETECTION ===
        autocorr_peak = 0.0
        periodicity = 0.0
        frequency: float | None = None
        confidence = 0.0

        # Autocorrelation for periodicity measure
        with suppress(Exception):
            autocorr = np.correlate(analysis_window, analysis_window, mode="full")
            autocorr = autocorr[len(autocorr) // 2 :]
            if len(autocorr) > 1 and autocorr[0] > 1e-10:
                autocorr_peak = safe_float(np.max(autocorr[1:]) / autocorr[0])
                periodicity = min(autocorr_peak, 1.0)

        # Aubio pitch detection - raw output
        with suppress(Exception):
            aubio_frame = pitch_frame.astype(np.float32)
            raw_frequency = safe_float(self._aubio_pitch(aubio_frame)[0])
            raw_confidence = safe_float(self._aubio_pitch.get_confidence())

            # Validate: aubio returns 0 or sample_rate when no pitch found
            # Only return frequency if it's a valid detection
            nyquist = self._config.sample_rate / 2
            is_valid = (
                raw_frequency > 0 and raw_frequency < nyquist and not np.isnan(raw_frequency) and not np.isinf(raw_frequency)
            )

            if is_valid:
                frequency = raw_frequency
                confidence = min(max(raw_confidence, 0.0), 1.0)
            else:
                # Invalid frequency - return None
                frequency = None
                confidence = 0.0

        # === OPTIONAL FEATURES ===

        # Spectral features
        spectral_features = None
        if FeatureFlags.SPECTRAL in self._config.features:
            spectral_features = extract_spectral_features(spectrum, freqs)

        # Harmonic features (only if frequency was detected)
        harmonic_features = None
        if FeatureFlags.HARMONIC in self._config.features:
            harmonic_features = extract_harmonic_features(
                spectrum,
                freqs,
                frequency if frequency is not None else 0.0,
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
                analysis_window,
                spectrum,
                freqs,
                self._config.sample_rate,
            )

        # Rhythm features
        onset_strength = None
        tempo_bpm = None

        if FeatureFlags.RHYTHM in self._config.features and self._aubio_onset and self._aubio_tempo:
            with suppress(Exception):
                aubio_frame = pitch_frame.astype(np.float32)
                onset_strength = safe_float(self._aubio_onset(aubio_frame)[0])

            with suppress(Exception):
                self._aubio_tempo(aubio_frame)
                tempo_bpm = safe_float(self._aubio_tempo.get_bpm())

        return AudioDetection(
            timestamp=monotonic(),
            frequency=frequency,
            confidence=confidence,
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
        """Create detection for complete silence (no signal)"""
        return AudioDetection(
            timestamp=monotonic(),
            frequency=None,
            confidence=0.0,
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
        """
        Start listening to microphone.

        Must be called before analyze(). Safe to call multiple times.
        """
        if self._listening or not self._stream:
            return

        self._stream.start()
        self._listening = True
        self._running = True

    def stop(self) -> None:
        """
        Stop microphone capture and cleanup resources.

        Safe to call multiple times.
        """
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
        Analyze current audio and return detection results.

        Returns exactly what the microphone captures - no filtering, no limits.
        frequency will be None if no valid pitch was detected.

        Returns:
            AudioDetection: Complete raw analysis of current audio moment

        Raises:
            RuntimeError: If not listening (call listen() first)
            KeyboardInterrupt: If shutdown signal received
        """
        if not self._listening:
            raise RuntimeError("Not listening. Call listen() first.")

        if self._shutdown_event.is_set():
            self.stop()
            raise KeyboardInterrupt

        return self._process_current_audio()

    @property
    def is_running(self) -> bool:
        """True if detector is actively listening to microphone"""
        return self._running

    @property
    def config(self) -> PitchlyConfig:
        """Current detector configuration"""
        return self._config

    def __enter__(self) -> PitchlyDetector:
        """Context manager entry - starts listening"""
        self.listen()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - stops and cleans up"""
        self.stop()

    def __del__(self) -> None:
        """Destructor - ensures cleanup"""
        with suppress(Exception):
            self.stop()
