# Standard modules
from contextlib import suppress
from threading import Lock, Thread
from time import time
from warnings import filterwarnings

# Third-party modules
from aubio import onset, pitch, tempo
from librosa import feature, power_to_db, pyin
from numpy import (
    argmax,
    argmin,
    correlate,
    cumsum,
    diff,
    exp,
    fft,
    float32,
    isinf,
    isnan,
    log,
    log10,
    mean,
    ndarray,
    percentile,
    polyfit,
    roll,
    sign,
    sqrt,
    std,
    where,
    zeros,
)
from pydantic import BaseModel, Field
from scipy.fft import rfft, rfftfreq
from scipy.signal import get_window
from scipy.stats import entropy
from sounddevice import InputStream


# Ignore warnings from underlying libraries
filterwarnings("ignore")


class SpectralFeatures(BaseModel):
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
    fundamental_strength: float = Field(ge=0.0, description="Fundamental frequency amplitude")
    weights: list[float] = Field(description="Harmonic amplitude weights (relative to fundamental)")
    frequencies: list[float] = Field(description="Detected harmonic frequencies in Hz")
    deviation: float = Field(ge=0.0, description="Average harmonic frequency deviation")
    inharmonicity: float = Field(ge=0.0, description="Inharmonicity coefficient")
    noise_ratio: float = Field(description="Harmonic to noise ratio")


class EnvelopeFeatures(BaseModel):
    attack_time: float = Field(ge=0.0, description="Attack time in seconds")
    decay_slope: float = Field(description="Decay slope coefficient")
    stability: float = Field(ge=0.0, le=1.0, description="Amplitude envelope stability")
    modulation_rate: float = Field(ge=0.0, description="Amplitude modulation frequency")


class TimbreFeatures(BaseModel):
    roughness: float = Field(ge=0.0, description="Sensory dissonance measure")
    mfcc: list[float] = Field(description="Mel-frequency cepstral coefficients")
    chroma: list[float] = Field(description="Chromagram features")
    spectral_contrast: list[float] = Field(description="Spectral contrast across frequency bands")


class AudioDetection(BaseModel):
    timestamp: float = Field(description="Detection timestamp in seconds")
    frequency: float = Field(ge=0.0, description="Fundamental frequency in Hz")
    confidence: float = Field(ge=0.0, le=1.0, description="Pitch detection confidence")
    voiced: bool = Field(description="Voice/sound activity detection")

    rms_energy: float = Field(ge=0.0, description="RMS energy level")
    peak_amplitude: float = Field(ge=0.0, description="Peak amplitude")
    dynamic_range: float = Field(ge=0.0, description="Peak to RMS ratio")
    snr_db: float = Field(description="Signal to noise ratio in dB")

    zero_crossing_rate: float = Field(ge=0.0, le=1.0, description="Zero crossing rate")
    autocorr_peak: float = Field(ge=0.0, le=1.0, description="Autocorrelation peak strength")
    periodicity: float = Field(ge=0.0, le=1.0, description="Signal periodicity measure")

    spectral: SpectralFeatures = Field(description="Spectral analysis features")
    harmonic: HarmonicFeatures = Field(description="Harmonic structure analysis")
    envelope: EnvelopeFeatures = Field(description="Amplitude envelope analysis")
    timbre: TimbreFeatures = Field(description="Timbral characteristics")

    onset_strength: float = Field(ge=0.0, description="Onset detection strength")
    tempo_bpm: float = Field(ge=0.0, description="Estimated tempo in BPM")

    def to_dict(self) -> dict:
        """Convert the AudioDetection instance to a dictionary"""

        return self.model_dump()

    def to_json(self) -> str:
        """Convert the AudioDetection instance to JSON string"""

        return self.model_dump_json()


def _safe_float(value, default: float = 0.0) -> float:
    """Convert value to float, handling NaN and inf"""

    if isnan(value) or isinf(value):
        return default

    return float(value)


def _safe_array(array, default_value: float = 0.0) -> list[float]:
    """Convert array to list, handling NaN and inf"""

    result = []

    for val in array:
        if isnan(val) or isinf(val):
            result.append(default_value)
        else:
            result.append(float(val))

    return result


class PitchlyDetector:
    def __init__(self, sample_rate: int = 48000, buffer_duration: float = 0.1, hop_length: int = 512) -> None:
        self._sample_rate = sample_rate
        self._hop_length = hop_length
        self._buffer_samples = int(buffer_duration * sample_rate)
        self._fft_size = 1024

        self._aubio_pitch = pitch("yin", self._fft_size, self._hop_length, self._sample_rate)
        self._aubio_pitch.set_unit("Hz")
        self._aubio_pitch.set_tolerance(0.8)

        self._aubio_onset = onset("default", self._fft_size, self._hop_length, self._sample_rate)
        self._aubio_tempo = tempo("default", self._fft_size, self._hop_length, self._sample_rate)

        self._stream: InputStream | None = None
        self._running = False
        self._thread: Thread | None = None
        self._audio_buffer = zeros(self._buffer_samples, dtype=float32)
        self._envelope_history = zeros(100)
        self._lock = Lock()

        self._started = False

    def _extract_spectral_features(self, spectrum: ndarray, freqs: ndarray) -> SpectralFeatures:
        spectrum_norm = spectrum / (sum(spectrum) + 1e-8)

        centroid = _safe_float(sum(freqs * spectrum_norm))
        spread = _safe_float(sqrt(sum(((freqs - centroid) ** 2) * spectrum_norm)))

        cumsum_data = cumsum(spectrum_norm)
        rolloff_85_idx = where(cumsum_data >= 0.85)[0]
        rolloff_95_idx = where(cumsum_data >= 0.95)[0]
        rolloff_85 = _safe_float(freqs[rolloff_85_idx[0]] if len(rolloff_85_idx) > 0 else freqs[-1])
        rolloff_95 = _safe_float(freqs[rolloff_95_idx[0]] if len(rolloff_95_idx) > 0 else freqs[-1])

        geometric_mean = exp(mean(log(spectrum + 1e-8)))
        arithmetic_mean = mean(spectrum)
        flatness = _safe_float(geometric_mean / (arithmetic_mean + 1e-8), 0.0)
        flatness = min(flatness, 1.0)

        crest = _safe_float(max(spectrum) / (arithmetic_mean + 1e-8))
        spec_entropy = _safe_float(entropy(spectrum_norm + 1e-8))
        slope = _safe_float(polyfit(freqs, spectrum, 1)[0])

        brightness = _safe_float(sum(spectrum[freqs > 1500]) / (sum(spectrum) + 1e-8))
        brightness = min(brightness, 1.0)
        warmth = _safe_float(sum(spectrum[(freqs >= 200) & (freqs <= 800)]) / (sum(spectrum) + 1e-8))
        warmth = min(warmth, 1.0)
        sharpness = _safe_float(sum(spectrum[freqs > 2000]) / (sum(spectrum) + 1e-8))

        return SpectralFeatures(
            centroid=centroid,
            spread=spread,
            rolloff_85=rolloff_85,
            rolloff_95=rolloff_95,
            flatness=flatness,
            crest=crest,
            entropy=spec_entropy,
            slope=slope,
            brightness=brightness,
            warmth=warmth,
            sharpness=sharpness,
        )

    def _extract_harmonic_features(self, spectrum: ndarray, freqs: ndarray, f0: float) -> HarmonicFeatures:
        if f0 <= 0 or isnan(f0) or isinf(f0):
            return HarmonicFeatures(
                fundamental_strength=0.0,
                weights=[0.0] * 8,
                frequencies=[0.0] * 8,
                deviation=0.0,
                inharmonicity=0.0,
                noise_ratio=0.0,
            )

        f0_idx = argmin(abs(freqs - f0))
        fundamental_strength = _safe_float(spectrum[f0_idx])

        harmonic_weights = []
        harmonic_freqs = []
        harmonic_deviations = []

        for h in range(1, 9):
            target_freq = f0 * h

            if target_freq >= self._sample_rate / 2:
                harmonic_weights.append(0.0)
                harmonic_freqs.append(0.0)
                harmonic_deviations.append(0.0)
                continue

            target_idx = argmin(abs(freqs - target_freq))
            search_range = max(3, int(0.02 * target_idx))

            start_idx = max(0, target_idx - search_range)
            end_idx = min(len(spectrum), target_idx + search_range + 1)

            if start_idx >= end_idx:
                harmonic_weights.append(0.0)
                harmonic_freqs.append(0.0)
                harmonic_deviations.append(0.0)
                continue

            local_spectrum = spectrum[start_idx:end_idx]
            local_freqs = freqs[start_idx:end_idx]

            peak_idx = argmax(local_spectrum)
            actual_freq = local_freqs[peak_idx]
            peak_amplitude = local_spectrum[peak_idx]

            weight = _safe_float(peak_amplitude / (fundamental_strength + 1e-8))
            deviation = _safe_float(abs(actual_freq - target_freq) / (target_freq + 1e-8))

            harmonic_weights.append(weight)
            harmonic_freqs.append(_safe_float(actual_freq))
            harmonic_deviations.append(deviation)

        avg_deviation = _safe_float(mean(harmonic_deviations))

        inharmonicity_sum = 0.0

        for dev, weight in zip(harmonic_deviations, harmonic_weights, strict=False):
            if not (isnan(dev) or isnan(weight)):
                inharmonicity_sum += dev * weight

        inharmonicity = _safe_float(inharmonicity_sum)

        harmonic_energy = sum(w * fundamental_strength for w in harmonic_weights if not isnan(w))
        total_energy = _safe_float(sum(spectrum))
        noise_energy = max(0, total_energy - harmonic_energy)
        noise_ratio = _safe_float(harmonic_energy / (noise_energy + 1e-8))

        return HarmonicFeatures(
            fundamental_strength=fundamental_strength,
            weights=harmonic_weights,
            frequencies=harmonic_freqs,
            deviation=avg_deviation,
            inharmonicity=inharmonicity,
            noise_ratio=noise_ratio,
        )

    def _extract_envelope_features(self, audio_frame: ndarray) -> EnvelopeFeatures:
        envelope = abs(audio_frame)

        self._envelope_history = roll(self._envelope_history, -1)
        self._envelope_history[-1] = _safe_float(mean(envelope))

        stability = _safe_float(1.0 / (1.0 + std(self._envelope_history) + 1e-8))
        stability = min(stability, 1.0)

        attack_time = 0.0
        decay_slope = 0.0
        modulation_rate = 0.0

        active_samples = len(where(self._envelope_history > 0.01)[0])

        if active_samples > 10:
            peaks = where(diff(self._envelope_history) > 0)[0]

            if len(peaks) > 0:
                attack_samples = peaks[-1]
                attack_time = _safe_float(attack_samples / 100.0)

            if len(self._envelope_history) > 20:
                recent_envelope = self._envelope_history[-20:]
                decay_slope = 0.0
                modulation_rate = 0.0

                with suppress(Exception):
                    decay_slope = _safe_float(polyfit(range(len(recent_envelope)), recent_envelope, 1)[0])

                with suppress(Exception):
                    envelope_fft = abs(fft.fft(self._envelope_history))
                    envelope_freqs = fft.fftfreq(len(self._envelope_history), d=1.0 / self._sample_rate)

                    if len(envelope_fft) > 10:
                        mod_peak_idx = argmax(envelope_fft[1:10]) + 1
                        modulation_rate = _safe_float(abs(envelope_freqs[mod_peak_idx]))

        return EnvelopeFeatures(
            attack_time=attack_time, decay_slope=decay_slope, stability=stability, modulation_rate=modulation_rate
        )

    def _extract_timbre_features(self, audio_frame: ndarray, spectrum: ndarray, freqs: ndarray) -> TimbreFeatures:
        roughness = 0.0

        with suppress(Exception):
            for i in range(len(spectrum) - 1):
                for j in range(i + 1, min(i + 20, len(spectrum))):
                    freq_diff = freqs[j] - freqs[i]

                    if 0 < freq_diff < 240:
                        dissonance = exp(-3.5 * freq_diff / 100) * spectrum[i] * spectrum[j]

                        if not isnan(dissonance):
                            roughness += dissonance

        roughness = _safe_float(roughness)
        mfcc = [0.0] * 13
        chroma = [0.0] * 12
        contrast = [0.0] * 7

        with suppress(Exception):
            mel_spectrum = feature.melspectrogram(y=audio_frame, sr=self._sample_rate, n_mels=13)
            mfcc = feature.mfcc(S=power_to_db(mel_spectrum), n_mfcc=13)[:, 0]
            mfcc = _safe_array(mfcc)

        with suppress(Exception):
            chroma = feature.chroma_stft(y=audio_frame, sr=self._sample_rate)[:, 0]
            chroma = _safe_array(chroma)

        with suppress(Exception):
            contrast = feature.spectral_contrast(y=audio_frame, sr=self._sample_rate)[:, 0]
            contrast = _safe_array(contrast)

        return TimbreFeatures(roughness=roughness, mfcc=mfcc, chroma=chroma, spectral_contrast=contrast)

    def _process_current_audio(self) -> AudioDetection:
        """Process the current audio buffer and return analysis"""

        with self._lock:
            buffer_copy = self._audio_buffer[-self._hop_length :].copy()

        if len(buffer_copy) == 0 or max(abs(buffer_copy)) < 1e-8:
            return self._create_silent_detection()

        windowed = buffer_copy * get_window("hann", len(buffer_copy))
        spectrum = abs(rfft(windowed, n=self._fft_size))
        freqs = rfftfreq(self._fft_size, 1 / self._sample_rate)

        rms_energy = _safe_float(sqrt(mean(buffer_copy**2)))
        peak_amplitude = _safe_float(max(abs(buffer_copy)))
        dynamic_range = _safe_float(peak_amplitude / (rms_energy + 1e-8))

        noise_floor = percentile(spectrum, 20)
        snr_db = _safe_float(20 * log10((rms_energy + 1e-8) / (noise_floor + 1e-8)))

        zero_crossings = _safe_float(sum(diff(sign(buffer_copy)) != 0) / len(buffer_copy))
        zero_crossings = min(zero_crossings, 1.0)

        autocorr_peak = 0.0
        periodicity = 0.0
        frequency = 0.0
        confidence = 0.0
        voiced = False

        with suppress(Exception):
            autocorr = correlate(buffer_copy, buffer_copy, mode="full")
            autocorr = autocorr[len(autocorr) // 2 :]
            autocorr_peak = _safe_float(max(autocorr[1 : min(100, len(autocorr))]) / (autocorr[0] + 1e-8))
            autocorr_peak = min(autocorr_peak, 1.0)
            periodicity = autocorr_peak

        with suppress(Exception):
            aubio_frame = buffer_copy.astype(float32)
            frequency = _safe_float(self._aubio_pitch(aubio_frame)[0])
            confidence = _safe_float(self._aubio_pitch.get_confidence())
            confidence = min(confidence, 1.0)
            voiced = bool(frequency > 0 and confidence > 0.5)

        if not voiced:
            with suppress(Exception):
                pyin_freqs, voiced_flag, voiced_probs = pyin(buffer_copy, fmin=20.0, fmax=8000.0, sr=self._sample_rate)

                if len(pyin_freqs) > 0 and not isnan(pyin_freqs[-1]):
                    frequency = _safe_float(pyin_freqs[-1])
                    confidence = _safe_float(voiced_probs[-1] if len(voiced_probs) > 0 else 0.5)
                    confidence = min(confidence, 1.0)
                    voiced = bool(voiced_flag[-1]) if len(voiced_flag) > 0 else False

        spectral_features = self._extract_spectral_features(spectrum, freqs)
        harmonic_features = self._extract_harmonic_features(spectrum, freqs, frequency)
        envelope_features = self._extract_envelope_features(buffer_copy)
        timbre_features = self._extract_timbre_features(buffer_copy, spectrum, freqs)

        onset_strength = 0.0
        tempo_bpm = 0.0

        with suppress(Exception):
            onset_strength = _safe_float(self._aubio_onset(aubio_frame)[0])

        with suppress(Exception):
            self._aubio_tempo(aubio_frame)
            tempo_bpm = _safe_float(self._aubio_tempo.get_bpm())

        return AudioDetection(
            timestamp=time(),
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
        """Create detection for silent/empty audio"""

        return AudioDetection(
            timestamp=time(),
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
            spectral=SpectralFeatures(
                centroid=0.0,
                spread=0.0,
                rolloff_85=0.0,
                rolloff_95=0.0,
                flatness=0.0,
                crest=0.0,
                entropy=0.0,
                slope=0.0,
                brightness=0.0,
                warmth=0.0,
                sharpness=0.0,
            ),
            harmonic=HarmonicFeatures(
                fundamental_strength=0.0,
                weights=[0.0] * 8,
                frequencies=[0.0] * 8,
                deviation=0.0,
                inharmonicity=0.0,
                noise_ratio=0.0,
            ),
            envelope=EnvelopeFeatures(attack_time=0.0, decay_slope=0.0, stability=0.0, modulation_rate=0.0),
            timbre=TimbreFeatures(roughness=0.0, mfcc=[0.0] * 13, chroma=[0.0] * 12, spectral_contrast=[0.0] * 7),
            onset_strength=0.0,
            tempo_bpm=0.0,
        )

    def _audio_callback(self, indata: ndarray, frames: int, time_info, status) -> None:
        with self._lock:
            self._audio_buffer = roll(self._audio_buffer, -frames)
            self._audio_buffer[-frames:] = indata.flatten()

    def start(self) -> None:
        """Start audio capture"""

        if self._started:
            return

        self._started = True

        self._stream = InputStream(
            samplerate=self._sample_rate, channels=1, blocksize=self._hop_length, callback=self._audio_callback, dtype="float32"
        )
        self._stream.start()

    def stop(self) -> None:
        """Stop audio capture"""

        if not self._started:
            return

        self._started = False

        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def analyze(self) -> AudioDetection:
        """
        Analyze current audio and return detection results
        Call this method whenever you want to get current audio analysis
        """

        if not self._started:
            raise RuntimeError("Detector not started. Call start() first.")

        return self._process_current_audio()

    @property
    def is_running(self) -> bool:
        return self._started

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
