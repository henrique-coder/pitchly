"""Harmonic feature extraction"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pitchly.types import HarmonicFeatures
from pitchly.utils import safe_division, safe_float


if TYPE_CHECKING:
    from numpy.typing import NDArray


def extract_harmonic_features(
    spectrum: NDArray,
    freqs: NDArray,
    f0: float,
    sample_rate: int,
    num_harmonics: int = 8,
) -> HarmonicFeatures:
    """
    Extract harmonic structure features

    Args:
        spectrum: Magnitude spectrum
        freqs: Frequency bins
        f0: Fundamental frequency (Hz)
        sample_rate: Audio sample rate
        num_harmonics: Number of harmonics to analyze

    Returns:
        HarmonicFeatures: Harmonic structure characteristics
    """
    # Validate fundamental frequency
    if f0 <= 0 or np.isnan(f0) or np.isinf(f0):
        return create_empty_harmonic(num_harmonics)

    # Find fundamental frequency bin
    f0_idx = np.argmin(np.abs(freqs - f0))
    fundamental_strength = safe_float(spectrum[f0_idx])

    harmonic_weights = []
    harmonic_freqs = []
    harmonic_deviations = []

    # Analyze harmonics
    for h in range(1, num_harmonics + 1):
        target_freq = f0 * h

        # Check if harmonic exceeds Nyquist frequency
        if target_freq >= sample_rate / 2:
            # Fill remaining with zeros
            harmonic_weights.extend([0.0] * (num_harmonics + 1 - h))
            harmonic_freqs.extend([0.0] * (num_harmonics + 1 - h))
            break

        # Find peak near expected harmonic
        target_idx = np.argmin(np.abs(freqs - target_freq))
        search_range = max(3, int(0.02 * target_idx))

        start_idx = max(0, target_idx - search_range)
        end_idx = min(len(spectrum), target_idx + search_range + 1)

        if start_idx >= end_idx:
            harmonic_weights.append(0.0)
            harmonic_freqs.append(0.0)
            harmonic_deviations.append(0.0)
            continue

        # Extract local spectrum region
        local_spectrum = spectrum[start_idx:end_idx]
        local_freqs = freqs[start_idx:end_idx]

        # Find peak in local region
        peak_idx = np.argmax(local_spectrum)
        actual_freq = local_freqs[peak_idx]
        peak_amplitude = local_spectrum[peak_idx]

        # Calculate weight relative to fundamental
        weight = safe_float(safe_division(peak_amplitude, fundamental_strength))

        # Calculate frequency deviation from expected harmonic
        deviation = safe_float(np.abs(actual_freq - target_freq) / max(target_freq, 1e-10))

        harmonic_weights.append(weight)
        harmonic_freqs.append(safe_float(actual_freq))
        harmonic_deviations.append(deviation)

    # Calculate average deviation
    avg_deviation = safe_float(np.mean(harmonic_deviations) if harmonic_deviations else 0.0)

    # Calculate inharmonicity (weighted deviation)
    min_len = min(len(harmonic_deviations), len(harmonic_weights))
    inharmonicity_sum = sum(
        harmonic_deviations[i] * harmonic_weights[i]
        for i in range(min_len)
        if not (np.isnan(harmonic_deviations[i]) or np.isnan(harmonic_weights[i]))
    )
    inharmonicity = safe_float(inharmonicity_sum)

    # Harmonic to noise ratio
    harmonic_energy = sum(w * fundamental_strength for w in harmonic_weights if not np.isnan(w))
    total_energy = safe_float(np.sum(spectrum))
    noise_energy = max(0, total_energy - harmonic_energy)
    noise_ratio = safe_float(safe_division(harmonic_energy, noise_energy))

    return HarmonicFeatures(
        fundamental_strength=fundamental_strength,
        weights=tuple(harmonic_weights),
        frequencies=tuple(harmonic_freqs),
        deviation=avg_deviation,
        inharmonicity=inharmonicity,
        noise_ratio=noise_ratio,
    )


def create_empty_harmonic(num_harmonics: int = 8) -> HarmonicFeatures:
    """Create harmonic features with zero values"""
    return HarmonicFeatures(
        fundamental_strength=0.0,
        weights=tuple([0.0] * num_harmonics),
        frequencies=tuple([0.0] * num_harmonics),
        deviation=0.0,
        inharmonicity=0.0,
        noise_ratio=0.0,
    )
