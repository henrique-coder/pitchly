"""Envelope feature extraction"""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING

import numpy as np

from pitchly.types import EnvelopeFeatures
from pitchly.utils import safe_float


if TYPE_CHECKING:
    from numpy.typing import NDArray


def extract_envelope_features(
    audio_frame: NDArray,
    envelope_history: NDArray,
    sample_rate: int,
) -> tuple[EnvelopeFeatures, NDArray]:
    """
    Extract amplitude envelope features

    Args:
        audio_frame: Audio samples
        envelope_history: Circular buffer of past envelope values
        sample_rate: Audio sample rate

    Returns:
        tuple: (EnvelopeFeatures, updated_envelope_history)
    """
    envelope = np.abs(audio_frame)

    # Update envelope history (circular buffer)
    updated_history = np.roll(envelope_history, -1)
    updated_history[-1] = safe_float(np.mean(envelope))

    # Calculate stability (inverse of variance)
    stability = min(safe_float(1.0 / (1.0 + np.std(updated_history) + 1e-10)), 1.0)

    attack_time = 0.0
    decay_slope = 0.0
    modulation_rate = 0.0

    # Only calculate if we have enough active samples
    active_samples = np.sum(updated_history > 0.01)

    if active_samples > 5:
        # Attack time (time to reach peak)
        peaks = np.where(np.diff(updated_history) > 0)[0]
        if len(peaks) > 0:
            attack_samples = peaks[-1]
            attack_time = safe_float(attack_samples / len(updated_history))

        # Decay slope and modulation (if enough history)
        if len(updated_history) > 10:
            recent_envelope = updated_history[-10:]

            # Decay slope via linear regression
            with suppress(Exception):
                decay_slope = safe_float(np.polyfit(range(len(recent_envelope)), recent_envelope, 1)[0])

            # Modulation rate via FFT
            with suppress(Exception):
                envelope_fft = np.abs(np.fft.fft(updated_history))
                envelope_freqs = np.fft.fftfreq(len(updated_history), d=1.0 / sample_rate)

                if len(envelope_fft) > 5:
                    # Look for modulation in low frequency range
                    mod_peak_idx = np.argmax(envelope_fft[1:5]) + 1
                    modulation_rate = safe_float(np.abs(envelope_freqs[mod_peak_idx]))

    features = EnvelopeFeatures(
        attack_time=attack_time,
        decay_slope=decay_slope,
        stability=stability,
        modulation_rate=modulation_rate,
    )

    return features, updated_history


def create_empty_envelope() -> EnvelopeFeatures:
    """Create envelope features with zero values"""
    return EnvelopeFeatures(
        attack_time=0.0,
        decay_slope=0.0,
        stability=0.0,
        modulation_rate=0.0,
    )
