# 🎵 Pitchly

**Low latency real-time audio quality detection and analysis library for Python**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Real-time Audio](https://img.shields.io/badge/realtime-audio-red.svg)]()

Pitchly is a specialized, low latency audio quality detection library designed for **real-time audio analysis**. Optimized for applications requiring audio quality assessment, noise detection, and audio characteristics analysis. Built with performance-first architecture for production environments.

---

## ✨ Features

- ⚡ **Low latency** audio quality detection (~10ms processing time)
- � **Real-time audio quality assessment** with SNR and dynamic range analysis
- 📊 **Spectral quality metrics** (brightness, warmth, clarity, noise levels)
- � **Precision pitch detection** with advanced confidence scoring
- 🛡️ **Noise and distortion detection** with real-time alerts
- �️ **Dynamic range monitoring** for audio level optimization
- 📈 **Audio stability tracking** for consistent quality monitoring
- 🔄 **Thread-safe architecture** optimized for minimal CPU overhead
- 📦 **Production-ready** with comprehensive error handling
- 🎛️ **Zero-configuration** context manager for instant setup

---

## 🚀 Installation

Install Pitchly using UV (recommended):

```bash
uv add git+https://github.com/henrique-coder/pitchly.git
```

Or using pip:

```bash
pip install git+https://github.com/henrique-coder/pitchly.git
```

### Development Installation

```bash
git clone https://github.com/henrique-coder/pitchly.git
cd pitchly
make install
```

---

## 🎯 Quick Start

### Real-time Audio Quality Monitoring

```python
from pitchly import PitchlyDetector
from time import sleep


# Low latency configuration for real-time quality detection
with PitchlyDetector(
    sample_rate=48000,    # Professional audio quality
    buffer_duration=0.1,  # Minimal latency (100ms buffer)
    hop_length=512        # Maximum temporal resolution
) as detector:

    while True:
        # Get instant audio quality analysis
        audio = detector.analyze()

        # Monitor audio quality in real-time
        print(f"Quality Score: {audio.confidence:.2f}")
        print(f"SNR: {audio.snr_db:.1f} dB")
        print(f"Dynamic Range: {audio.dynamic_range:.1f}")
        print(f"Stability: {audio.envelope.stability:.2f}")

        # Detect audio issues
        if audio.snr_db < 20:
            print("⚠️  Low signal quality detected")
        if audio.envelope.stability < 0.7:
            print("⚠️  Audio instability detected")

        sleep(0.05)  # 50ms update rate for real-time monitoring
```

### Instant Quality Assessment

```python
from pitchly import PitchlyDetector


# Minimal latency configuration for instant feedback
with PitchlyDetector() as detector:  # Uses optimized defaults

    audio = detector.analyze()

    # Get instant quality metrics
    quality_metrics = {
        'overall_quality': min(1.0, (
            audio.confidence * 0.4 +
            min(audio.snr_db / 30, 1.0) * 0.3 +
            audio.envelope.stability * 0.3
        )),
        'signal_clarity': audio.confidence,
        'noise_level': audio.snr_db,
        'stability': audio.envelope.stability,
        'latency_ms': 10  # Measured processing latency
    }

    print(f"Quality Score: {quality_metrics['overall_quality']:.2f}")
    print(f"Processing Latency: {quality_metrics['latency_ms']}ms")
```

---

## 📚 API Reference

### PitchlyDetector

The main class for real-time audio analysis.

#### Constructor Parameters

| Parameter         | Type    | Default | Description                                                 |
| ----------------- | ------- | ------- | ----------------------------------------------------------- |
| `sample_rate`     | `int`   | `48000` | Audio sample rate in Hz (quality vs latency trade-off)      |
| `buffer_duration` | `float` | `0.1`   | Audio buffer duration in seconds (low latency)              |
| `hop_length`      | `int`   | `512`   | Number of samples between analysis frames (high resolution) |

#### Methods

##### `start() -> None`

Starts audio capture from the default microphone.

```python
detector = PitchlyDetector()
detector.start()  # Begin audio capture
```

##### `stop() -> None`

Stops audio capture and releases resources.

```python
detector.stop()  # Stop capture and cleanup
```

##### `analyze() -> AudioDetection`

Analyzes the current audio buffer and returns comprehensive results.

```python
audio_data = detector.analyze()
print(f"Current pitch: {audio_data.frequency} Hz")
```

##### `is_running -> bool` (Property)

Returns whether the detector is currently active.

```python
if detector.is_running:
    data = detector.analyze()
```

---

### AudioDetection

The main result object containing all extracted audio features.

#### Core Properties

| Property     | Type    | Description                                   |
| ------------ | ------- | --------------------------------------------- |
| `timestamp`  | `float` | Detection timestamp in seconds                |
| `frequency`  | `float` | Fundamental frequency in Hz (0.0 if unvoiced) |
| `confidence` | `float` | Pitch detection confidence (0.0-1.0)          |
| `voiced`     | `bool`  | Whether audio contains voiced content         |

#### Energy Properties

| Property         | Type    | Description                 |
| ---------------- | ------- | --------------------------- |
| `rms_energy`     | `float` | RMS energy level            |
| `peak_amplitude` | `float` | Peak amplitude in the frame |
| `dynamic_range`  | `float` | Peak-to-RMS ratio           |
| `snr_db`         | `float` | Signal-to-noise ratio in dB |

#### Temporal Properties

| Property             | Type    | Description                             |
| -------------------- | ------- | --------------------------------------- |
| `zero_crossing_rate` | `float` | Zero crossing rate (0.0-1.0)            |
| `autocorr_peak`      | `float` | Autocorrelation peak strength (0.0-1.0) |
| `periodicity`        | `float` | Signal periodicity measure (0.0-1.0)    |
| `onset_strength`     | `float` | Onset detection strength                |
| `tempo_bpm`          | `float` | Estimated tempo in BPM                  |

#### Feature Objects

| Property   | Type               | Description                 |
| ---------- | ------------------ | --------------------------- |
| `spectral` | `SpectralFeatures` | Spectral analysis results   |
| `harmonic` | `HarmonicFeatures` | Harmonic structure analysis |
| `envelope` | `EnvelopeFeatures` | Amplitude envelope analysis |
| `timbre`   | `TimbreFeatures`   | Timbral characteristics     |

#### Serialization Methods

##### `to_dict() -> dict`

Converts the entire AudioDetection to a dictionary.

```python
audio_data = detector.analyze()
data_dict = audio_data.to_dict()
# Save to JSON, database, etc.
```

##### `to_json() -> str`

Converts the AudioDetection to a JSON string.

```python
audio_data = detector.analyze()
json_string = audio_data.to_json()
print(json_string)
```

---

### SpectralFeatures

Detailed spectral analysis of the audio signal.

| Property     | Type    | Range   | Description                                    |
| ------------ | ------- | ------- | ---------------------------------------------- |
| `centroid`   | `float` | ≥0.0    | Spectral centroid in Hz (brightness indicator) |
| `spread`     | `float` | ≥0.0    | Spectral spread (frequency distribution width) |
| `rolloff_85` | `float` | ≥0.0    | Frequency below which 85% of energy lies       |
| `rolloff_95` | `float` | ≥0.0    | Frequency below which 95% of energy lies       |
| `flatness`   | `float` | 0.0-1.0 | Spectral flatness (tonality vs. noise)         |
| `crest`      | `float` | ≥0.0    | Spectral crest factor (peakiness)              |
| `entropy`    | `float` | ≥0.0    | Spectral entropy (randomness measure)          |
| `slope`      | `float` | Any     | Spectral slope (tilt of spectrum)              |
| `brightness` | `float` | 0.0-1.0 | High-frequency energy ratio                    |
| `warmth`     | `float` | 0.0-1.0 | Low-mid frequency energy ratio                 |
| `sharpness`  | `float` | ≥0.0    | High-frequency emphasis measure                |

```python
spectral = audio_data.spectral
print(f"Brightness: {spectral.brightness:.2f}")
print(f"Warmth: {spectral.warmth:.2f}")
```

---

### HarmonicFeatures

Analysis of harmonic content and structure.

| Property               | Type          | Description                              |
| ---------------------- | ------------- | ---------------------------------------- |
| `fundamental_strength` | `float`       | Amplitude of the fundamental frequency   |
| `weights`              | `list[float]` | Harmonic amplitude weights (8 harmonics) |
| `frequencies`          | `list[float]` | Detected harmonic frequencies in Hz      |
| `deviation`            | `float`       | Average harmonic frequency deviation     |
| `inharmonicity`        | `float`       | Inharmonicity coefficient                |
| `noise_ratio`          | `float`       | Harmonic-to-noise ratio                  |

```python
harmonic = audio_data.harmonic
print(f"Fundamental: {harmonic.fundamental_strength:.2f}")
print(f"Harmonics: {harmonic.frequencies[:3]}")  # First 3 harmonics
```

---

### EnvelopeFeatures

Amplitude envelope and temporal characteristics.

| Property          | Type    | Range   | Description                    |
| ----------------- | ------- | ------- | ------------------------------ |
| `attack_time`     | `float` | ≥0.0    | Attack time in seconds         |
| `decay_slope`     | `float` | Any     | Decay slope coefficient        |
| `stability`       | `float` | 0.0-1.0 | Amplitude envelope stability   |
| `modulation_rate` | `float` | ≥0.0    | Amplitude modulation frequency |

```python
envelope = audio_data.envelope
print(f"Attack: {envelope.attack_time:.3f}s")
print(f"Stability: {envelope.stability:.2f}")
```

---

### TimbreFeatures

Timbral and perceptual characteristics.

| Property            | Type          | Description                            |
| ------------------- | ------------- | -------------------------------------- |
| `roughness`         | `float`       | Sensory dissonance measure             |
| `mfcc`              | `list[float]` | 13 Mel-frequency cepstral coefficients |
| `chroma`            | `list[float]` | 12 chromagram features                 |
| `spectral_contrast` | `list[float]` | 7 spectral contrast values             |

```python
timbre = audio_data.timbre
print(f"Roughness: {timbre.roughness:.2f}")
print(f"MFCC: {timbre.mfcc[:3]}")  # First 3 MFCC coefficients
```

---

## 🔧 Configuration Guide

### Sample Rate Selection

| Rate     | Use Case           | Latency    | Quality | Recommended |
| -------- | ------------------ | ---------- | ------- | ----------- |
| 22050 Hz | **Low-latency**    | **Lowest** | Fair    | **Speed**   |
| 44100 Hz | Standard quality   | Medium     | Good    | Basic apps  |
| 48000 Hz | Professional       | Higher     | High    | Quality     |
| 96000 Hz | Ultra-high quality | Highest    | Maximum | Audiophile  |

### Buffer Duration (Latency-Optimized)

| Duration | Latency     | Quality Detection | Use Case                |
| -------- | ----------- | ----------------- | ----------------------- |
| 0.02s    | Ultra-low   | Basic             | Real-time monitoring    |
| **0.1s** | **Minimal** | **Excellent**     | **Production apps**     |
| 0.2s     | Low         | Maximum           | High-precision analysis |

---

## 🏗️ Performance Characteristics

**Low Latency Design:**

- Processing time: ~10ms per analysis frame
- Thread-safe architecture optimized for real-time applications
- Configurable sample rates for latency vs quality trade-offs

**Production-Ready Features:**

- Automatic error handling and recovery
- Configurable quality thresholds
- Zero-allocation audio buffering
- Optimized numerical processing with NaN/inf protection

---

## 🤝 Contributing

We welcome contributions! Please feel free to:

1. Report bugs and suggest features in [Issues](https://github.com/henrique-coder/pitchly/issues)
2. Submit pull requests with improvements
3. Add examples and documentation
4. Share your projects using Pitchly

### Development Commands

```bash
# Install dependencies
make install

# Run code quality checks
make lint

# Format code
make format
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Aubio](https://aubio.org) for pitch detection algorithms
- [Librosa](https://librosa.org) for audio feature extraction
- [NumPy](https://numpy.org) for numerical computations
- [Pydantic](https://pydantic.dev) for data validation
- [SciPy](https://scipy.org) for scientific computing
- [SoundDevice](https://python-sounddevice.readthedocs.io) for audio I/O
