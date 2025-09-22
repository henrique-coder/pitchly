# 🎵 Pitchly

**Real-time audio feature extraction and pitch detection library for Python**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Audio Processing](https://img.shields.io/badge/audio-processing-orange.svg)]()

Pitchly is a powerful, real-time audio analysis library that provides comprehensive feature extraction including pitch detection, spectral analysis, harmonic analysis, and timbral characteristics. Built with modern Python practices and designed for both researchers and developers working with audio applications.

---

## ✨ Features

- 🎯 **Real-time pitch detection** with confidence scoring
- 📊 **Comprehensive spectral analysis** (centroid, spread, rolloff, flatness, etc.)
- 🎼 **Harmonic structure analysis** with inharmonicity detection
- 🎚️ **Amplitude envelope analysis** (attack, decay, modulation)
- 🎨 **Timbral feature extraction** (MFCC, chroma, spectral contrast, roughness)
- ⚡ **High-performance** audio processing with configurable parameters
- 🔄 **Thread-safe** real-time audio capture and analysis
- 📦 **Type-safe** with Pydantic models and full type hints
- 🎛️ **Context manager** support for easy resource management

---

## 🚀 Installation

Install Pitchly using UV (recommended):

```bash
uv add git+https://github.com/henrique-coder/pitchly.git@main
```

Or using pip:

```bash
pip install git+https://github.com/henrique-coder/pitchly.git@main
```

### Development Installation

```bash
git clone https://github.com/henrique-coder/pitchly.git
cd pitchly
make install
```

---

## 🎯 Quick Start

### Basic Usage

```python
from pitchly import PitchlyDetector
from time import sleep

# Create and start the detector
detector = PitchlyDetector()
detector.start()

try:
    while True:
        # Analyze current audio
        audio_data = detector.analyze()

        # Access pitch information
        print(f"Frequency: {audio_data.frequency:.2f} Hz")
        print(f"Confidence: {audio_data.confidence:.2f}")
        print(f"Voiced: {audio_data.voiced}")

        sleep(0.1)
finally:
    detector.stop()
```

### Context Manager Usage

```python
from pitchly import PitchlyDetector

# Automatic resource management
with PitchlyDetector() as detector:
    while True:
        audio_data = detector.analyze()

        # Access spectral features
        spectral = audio_data.spectral
        print(f"Spectral Centroid: {spectral.centroid:.2f} Hz")
        print(f"Brightness: {spectral.brightness:.2f}")

        if audio_data.voiced:
            break
```

### Advanced Configuration

```python
from pitchly import PitchlyDetector

# Custom detector configuration
detector = PitchlyDetector(
    sample_rate=44100,    # Higher sample rate for better quality
    buffer_duration=1.0,  # Longer buffer duration for more context
    hop_length=1024       # Smaller hop length for finer temporal resolution
)
```

---

## 📚 API Reference

### PitchlyDetector

The main class for real-time audio analysis.

#### Constructor Parameters

| Parameter         | Type    | Default | Description                               |
| ----------------- | ------- | ------- | ----------------------------------------- |
| `sample_rate`     | `int`   | `44100` | Audio sample rate in Hz                   |
| `buffer_duration` | `float` | `1.0`   | Audio buffer duration in seconds          |
| `hop_length`      | `int`   | `1024`  | Number of samples between analysis frames |

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

## 🎼 Usage Examples

### Musical Note Detection

```python
from pitchly import PitchlyDetector
import time
import math

# Exact frequencies for equal temperament tuning (A4 = 440Hz)
# Formula: f(n) = 440 * 2^((n-69)/12) where n is MIDI note number
NOTES: dict[str, float] = {
    'C4': 261.6256,  'C#4': 277.1826, 'D4': 293.6648,  'D#4': 311.1270,
    'E4': 329.6276,  'F4': 349.2282,  'F#4': 369.9944, 'G4': 391.9954,
    'G#4': 415.3047, 'A4': 440.0000,  'A#4': 466.1638, 'B4': 493.8833,
    'C5': 523.2511,  'C#5': 554.3653, 'D5': 587.3295,  'D#5': 622.2540,
    'E5': 659.2551,  'F5': 698.4565,  'F#5': 739.9888, 'G5': 783.9909
}

def frequency_to_note(freq: float) -> tuple[str, float]:
    """Find the closest musical note to the given frequency (440Hz tuning)"""

    if freq <= 0:
        return "Silent", 0.0

    # Find the closest note by minimum frequency difference
    closest_note = min(NOTES.items(), key=lambda note: abs(note[1] - freq))
    note_name, note_freq = closest_note

    # Calculate how many cents off the detected frequency is
    cents_off = 1200 * math.log2(freq / note_freq) if note_freq > 0 else 0

    return note_name, cents_off

with PitchlyDetector() as detector:
    print("🎵 Musical Note Detection (440Hz tuning)")
    print("Listening for notes...")

    while True:
        audio_data = detector.analyze()

        if audio_data.voiced and audio_data.confidence > 0.7:
            note_name, cents_off = frequency_to_note(audio_data.frequency)

            # Format cents deviation
            cents_str = f"{cents_off:+.0f} cents" if abs(cents_off) >= 1 else "perfect"

            print(f"🎵 {note_name}: {audio_data.frequency:.2f} Hz ({cents_str})")
            print(f"   Confidence: {audio_data.confidence:.2f}")

        time.sleep(0.1)
```

```python
from pitchly import PitchlyDetector
import time

with PitchlyDetector() as detector:
    print("🎤 Voice Activity Detection")

    silence_count = 0
    speech_count = 0

    while True:
        audio_data = detector.analyze()

        # Voice activity detection
        if (audio_data.voiced and
            audio_data.confidence > 0.6 and
            audio_data.rms_energy > 0.01):

            speech_count += 1
            silence_count = 0
            print("🗣️  Speech detected")

        else:
            silence_count += 1
            speech_count = 0

            if silence_count > 10:  # 1 second of silence
                print("🤫 Silence...")

        time.sleep(0.1)
```

### Audio Feature Analysis

```python
from pitchly import PitchlyDetector
import json

def analyze_audio_features():
    with PitchlyDetector() as detector:
        print("🔍 Detailed Audio Analysis")

        while True:
            audio_data = detector.analyze()

            # Create feature summary
            features = {
                'pitch': {
                    'frequency': audio_data.frequency,
                    'confidence': audio_data.confidence,
                    'voiced': audio_data.voiced
                },
                'spectral': {
                    'brightness': audio_data.spectral.brightness,
                    'warmth': audio_data.spectral.warmth,
                    'centroid': audio_data.spectral.centroid
                },
                'energy': {
                    'rms': audio_data.rms_energy,
                    'dynamic_range': audio_data.dynamic_range,
                    'snr_db': audio_data.snr_db
                },
                'temporal': {
                    'periodicity': audio_data.periodicity,
                    'tempo': audio_data.tempo_bpm
                }
            }

            print(json.dumps(features, indent=2))
            break

analyze_audio_features()
```

---

## 🔧 Configuration Guide

### Sample Rate Selection

| Rate     | Use Case           | Pros                | Cons                    |
| -------- | ------------------ | ------------------- | ----------------------- |
| 22050 Hz | Speech analysis    | Lower CPU usage     | Limited frequency range |
| 44100 Hz | Music analysis     | Full audio spectrum | Higher CPU usage        |
| 48000 Hz | Professional audio | Studio quality      | Highest CPU usage       |

### Buffer Duration

| Duration | Use Case       | Latency  | Stability |
| -------- | -------------- | -------- | --------- |
| 0.2s     | Real-time apps | Very low | Lower     |
| 0.5s     | General use    | Low      | Good      |
| 1.0s     | Analysis apps  | Medium   | High      |

---

## 🏗️ Architecture

Pitchly is built with a modular architecture:

```
pitchly/
├── core.py           # Main detection and feature extraction
├── __init__.py       # Public API exports
└── cli/             # Command-line interface
    └── __main__.py
```

**Key Components:**

- **PitchlyDetector**: Thread-safe real-time audio capture and processing
- **Feature Classes**: Structured data models for different analysis types
- **AudioDetection**: Comprehensive result container with serialization
- **Helper Functions**: Safe numerical processing and error handling

---

## 🤝 Contributing

We welcome contributions! Please feel free to:

1. Report bugs and suggest features in [Issues](https://github.com/henrique-coder/pitchly/issues)
2. Submit pull requests with improvements
3. Add examples and documentation
4. Share your projects using Pitchly

### Running Tests

```bash
make tests
```

### Code Quality

```bash
make lint
make format
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Aubio](https://aubio.org/) for pitch detection algorithms
- [Librosa](https://librosa.org/) for audio feature extraction
- [Pydantic](https://pydantic.dev/) for data validation
- [SoundDevice](https://python-sounddevice.readthedocs.io/) for audio I/O

---

**Made with ❤️ for the audio processing community**
