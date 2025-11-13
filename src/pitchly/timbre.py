"""Timbre feature extraction"""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING

from librosa import feature, power_to_db
import numpy as np

from pitchly.types import TimbreFeatures
from pitchly.utils import safe_array, safe_float


if TYPE_CHECKING:
    from numpy.typing import NDArray


def extract_timbre_features(
    audio_frame: NDArray,
    spectrum: NDArray,
    freqs: NDArray,
    sample_rate: int,
) -> TimbreFeatures:
    """
    Extract timbral characteristics

    Args:
        audio_frame: Audio samples
        spectrum: Magnitude spectrum
        freqs: Frequency bins
        sample_rate: Audio sample rate

    Returns:
        TimbreFeatures: Timbral characteristics
    """
    roughness = 0.0

    # Calculate roughness (sensory dissonance from close frequencies)
    with suppress(Exception):
        # Only check first 100 bins for performance
        for i in range(min(len(spectrum) - 1, 100)):
            for j in range(i + 1, min(i + 10, len(spectrum))):
                freq_diff = freqs[j] - freqs[i]

                # Critical band roughness (20-240 Hz difference)
                if 0 < freq_diff < 240:
                    dissonance = np.exp(-3.5 * freq_diff / 100) * spectrum[i] * spectrum[j]
                    if not np.isnan(dissonance):
                        roughness += dissonance

    roughness = safe_float(roughness)

    # MFCC (Mel-frequency cepstral coefficients)
    mfcc = [0.0] * 13
    with suppress(Exception):
        mel_spectrum = feature.melspectrogram(y=audio_frame, sr=sample_rate, n_mels=13)
        mfcc_result = feature.mfcc(S=power_to_db(mel_spectrum), n_mfcc=13)
        if mfcc_result.shape[1] > 0:
            mfcc = safe_array(mfcc_result[:, 0])

    # Chroma (pitch class distribution)
    chroma = [0.0] * 12
    with suppress(Exception):
        chroma_result = feature.chroma_stft(y=audio_frame, sr=sample_rate)
        if chroma_result.shape[1] > 0:
            chroma = safe_array(chroma_result[:, 0])

    # Spectral contrast (peak valley difference across frequency bands)
    contrast = [0.0] * 7
    with suppress(Exception):
        contrast_result = feature.spectral_contrast(y=audio_frame, sr=sample_rate)
        if contrast_result.shape[1] > 0:
            contrast = safe_array(contrast_result[:, 0])

    return TimbreFeatures(
        roughness=roughness,
        mfcc=tuple(mfcc),
        chroma=tuple(chroma),
        spectral_contrast=tuple(contrast),
    )


def create_empty_timbre() -> TimbreFeatures:
    """Create timbre features with zero values"""
    return TimbreFeatures(
        roughness=0.0,
        mfcc=tuple([0.0] * 13),
        chroma=tuple([0.0] * 12),
        spectral_contrast=tuple([0.0] * 7),
    )
