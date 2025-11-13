"""Numerical utilities and validation functions"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit
import numpy as np


if TYPE_CHECKING:
    from numpy.typing import NDArray


@jit(nopython=True, cache=True, fastmath=True)
def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default value for zero denominator"""
    return numerator / denominator if denominator != 0 else default


@jit(nopython=True, cache=True, fastmath=True)
def safe_log2(value: float, default: float = 0.0) -> float:
    """Safe log2 with bounds checking for invalid values"""
    return np.log2(value) if value > 0 and not (np.isnan(value) or np.isinf(value)) else default


def safe_float(value: float | np.floating, default: float = 0.0) -> float:
    """Convert value to float, handling NaN and inf"""
    if np.isnan(value) or np.isinf(value):
        return default
    return float(value)


def safe_array(array: NDArray, default_value: float = 0.0) -> list[float]:
    """Convert array to list, replacing NaN and inf with default"""
    return [float(v) if not (np.isnan(v) or np.isinf(v)) else default_value for v in array]


def validate_frequency(frequency: float, freq_min: float, freq_max: float) -> tuple[float, bool]:
    """
    Validate frequency is within acceptable range

    Returns:
        tuple: (validated_frequency, is_valid)
            - If invalid (NaN, inf, or out of range), returns (0.0, False)
            - If valid, returns (frequency, True)
    """
    # Check for NaN or inf
    if np.isnan(frequency) or np.isinf(frequency):
        return 0.0, False

    # Check range
    if frequency < freq_min or frequency > freq_max:
        return 0.0, False

    return frequency, True


def validate_confidence(confidence: float, min_confidence: float) -> bool:
    """Check if confidence meets minimum threshold"""
    if np.isnan(confidence) or np.isinf(confidence):
        return False
    return confidence >= min_confidence
