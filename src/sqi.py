"""Signal-quality gate for the BIDMC batch validator.

Rejects windows in the bottom 10th percentile of per-record band-passed variance.
The gate catches dead-sensor / flat-line windows; it is NOT a full SQI — production
code would add kurtosis, signal-to-noise, or template-correlation checks.

The validator reports both gated and ungated metrics so the gate is honest: it can't
hide poor algorithm performance, only filter obvious garbage windows.
"""
from __future__ import annotations
from typing import Sequence


def window_variance(y: Sequence[float]) -> float:
    """Variance of one window's filtered signal. Higher = more cardiac activity."""
    n = len(y)
    if n < 2:
        return 0.0
    mean = sum(y) / n
    return sum((v - mean) ** 2 for v in y) / (n - 1)


def variance_threshold(variances: Sequence[float], pct: float = 10.0) -> float:
    """Per-record pct-th percentile cutoff over window variances."""
    if not variances:
        return 0.0
    s = sorted(variances)
    k = int(len(s) * pct / 100.0)
    return s[min(k, len(s) - 1)]


def accept_mask(variances: Sequence[float], pct: float = 10.0) -> list[bool]:
    """Per-window mask: True = accept, False = reject as low-SQI."""
    threshold = variance_threshold(variances, pct)
    return [v >= threshold for v in variances]
