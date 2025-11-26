# Pitchly

Low latency real-time audio quality detection with configurable features.

## Installation

```bash
uv pip install git+https://github.com/henrique-coder/pitchly.git@main
```

## Usage

```python
from pitchly import PitchlyDetector

detector = PitchlyDetector()
detector.listen()

while True:
    audio = detector.analyze()

    if audio.frequency is not None:
        print(f"Frequency: {audio.frequency:.1f}Hz")
    else:
        print("No pitch detected")
```

## Configuration

```python
from pitchly import PitchlyConfig, FeatureFlags

config = PitchlyConfig(
    sample_rate=48000,      # Sample rate (default: 48000)
    buffer_duration=0.2,    # Buffer duration in seconds (default: 0.2)
    hop_length=512,         # Samples between frames (default: 512)
    fft_size=8192,          # FFT size (default: 8192)
    pitch_tolerance=0.1,    # YIN tolerance (default: 0.1)
    features=FeatureFlags.BASIC,
)

detector = PitchlyDetector(config)
```

## Features

```python
# Combinations
FeatureFlags.BASIC    # Frequency + Energy (fastest)
FeatureFlags.QUALITY  # BASIC + Spectral + Harmonic
FeatureFlags.FULL     # All features

# Individual
FeatureFlags.FREQUENCY   # Frequency detection
FeatureFlags.SPECTRAL    # Spectral analysis
FeatureFlags.HARMONIC    # Harmonic analysis
FeatureFlags.ENVELOPE    # Envelope analysis
FeatureFlags.TIMBRE      # Timbre (MFCC, chroma)
FeatureFlags.RHYTHM      # Onset + tempo
```

## Result

```python
audio = detector.analyze()

# Core
audio.frequency       # float | None (None if no valid pitch)
audio.confidence      # 0.0 to 1.0
audio.rms_energy      # RMS energy
audio.peak_amplitude  # Peak amplitude

# Optional (depends on features)
audio.spectral        # SpectralFeatures | None
audio.harmonic        # HarmonicFeatures | None
audio.envelope        # EnvelopeFeatures | None
audio.timbre          # TimbreFeatures | None
audio.tempo_bpm       # float | None
```

## Example

```python
from pitchly import PitchlyDetector, PitchlyConfig, FeatureFlags
from time import sleep

config = PitchlyConfig(features=FeatureFlags.BASIC)

with PitchlyDetector(config) as detector:
    while True:
        audio = detector.analyze()

        if audio.frequency is not None:
            print(f"{audio.frequency:.1f}Hz (conf: {audio.confidence:.0%})")

        sleep(0.05)
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
