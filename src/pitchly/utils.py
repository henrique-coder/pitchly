"""Numerical utilities"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit
import numpy as np


if TYPE_CHECKING:
    from numpy.typing import NDArray


@jit(nopython=True, cache=True, fastmath=True)
def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default for zero denominator"""
    return numerator / denominator if denominator != 0 else default


@jit(nopython=True, cache=True, fastmath=True)
def safe_log2(value: float, default: float = 0.0) -> float:
    """Safe log2 with bounds checking"""
    return np.log2(value) if value > 0 and not (np.isnan(value) or np.isinf(value)) else default


def safe_float(value: float | np.floating, default: float = 0.0) -> float:
    """Convert value to float, handling NaN and inf"""
    if np.isnan(value) or np.isinf(value):
        return default
    return float(value)


def safe_array(array: NDArray, default_value: float = 0.0) -> list[float]:
    """Convert array to list, replacing NaN and inf"""
    return [float(v) if not (np.isnan(v) or np.isinf(v)) else default_value for v in array]
