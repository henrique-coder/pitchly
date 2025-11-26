"""Spectral feature extraction"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import entropy as scipy_entropy

from pitchly.types import SpectralFeatures
from pitchly.utils import safe_division, safe_float


if TYPE_CHECKING:
    from numpy.typing import NDArray


def extract_spectral_features(spectrum: NDArray, freqs: NDArray) -> SpectralFeatures:
    """Extract spectral features from frequency spectrum"""
    spectrum_sum = np.sum(spectrum)

    if spectrum_sum < 1e-10:
        return create_empty_spectral()

    spectrum_norm = spectrum / spectrum_sum

    # Centroid and spread
    centroid = safe_float(np.sum(freqs * spectrum_norm))
    spread = safe_float(np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum_norm)))

    # Rolloff frequencies
    cumsum_data = np.cumsum(spectrum_norm)
    rolloff_85_idx = np.searchsorted(cumsum_data, 0.85)
    rolloff_95_idx = np.searchsorted(cumsum_data, 0.95)
    rolloff_85 = safe_float(freqs[min(rolloff_85_idx, len(freqs) - 1)])
    rolloff_95 = safe_float(freqs[min(rolloff_95_idx, len(freqs) - 1)])

    # Flatness
    geometric_mean = np.exp(np.mean(np.log(spectrum + 1e-10)))
    arithmetic_mean = np.mean(spectrum)
    flatness = safe_float(safe_division(geometric_mean, arithmetic_mean))

    # Crest factor
    crest = safe_float(safe_division(np.max(spectrum), arithmetic_mean))

    # Entropy
    spec_entropy = safe_float(scipy_entropy(spectrum_norm + 1e-10))

    # Slope
    slope = safe_float(np.polyfit(freqs, spectrum, 1)[0])

    # Frequency-based metrics
    brightness = safe_float(np.sum(spectrum[freqs > 1500]) / spectrum_sum)
    warmth = safe_float(np.sum(spectrum[(freqs >= 200) & (freqs <= 800)]) / spectrum_sum)
    sharpness = safe_float(np.sum(spectrum[freqs > 2000]) / spectrum_sum)

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


def create_empty_spectral() -> SpectralFeatures:
    """Create spectral features with zero values"""
    return SpectralFeatures(
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
    )
