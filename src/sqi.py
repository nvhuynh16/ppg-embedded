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


def cnn_accept_mask(filtered_windows, weights_path=None, threshold=None):
    """1D-CNN SQI gate (numpy inference). Drop-in replacement for the variance
    gate on motion-corrupted wearable PPG (PPG-DaLiA-style data).

    `filtered_windows` : iterable of band-passed PPG windows. Each window
    should be the FIR output for an 8-s segment at the dataset's sample rate;
    if the window length isn't 512 (= 8 s @ 64 Hz, the training rate), it is
    resampled. Z-scoring is applied internally.

    Returns a list[bool] aligned with the input order.

    Imports numpy lazily so the variance-gate path stays stdlib-only.

    See `src/sqi_cnn.py` for the model definition. The weights live at
    `models/sqi_cnn_v1.npz` — not redistributed in the public repo (see the
    `models/` entry in `.gitignore`). Without weights, this raises FileNotFoundError.
    """
    import numpy as np
    from sqi_cnn import CNNSQIModel, WIN_LEN, DEFAULT_THRESHOLD, DEFAULT_WEIGHTS

    model = CNNSQIModel(weights_path or DEFAULT_WEIGHTS,
                        threshold or DEFAULT_THRESHOLD)
    out: list[bool] = []
    arr_list = []
    for y in filtered_windows:
        y_arr = np.asarray(y, dtype=np.float32)
        if len(y_arr) != WIN_LEN:
            # Resample to 512 samples (the CNN's training input length)
            from scipy.signal import resample
            y_arr = resample(y_arr, WIN_LEN).astype(np.float32)
        sd = float(y_arr.std()) or 1.0
        y_z = (y_arr - float(y_arr.mean())) / sd
        arr_list.append(y_z)
    if not arr_list:
        return out
    probs = model.forward(np.stack(arr_list))
    return [bool(p >= (threshold or DEFAULT_THRESHOLD)) for p in probs]
