"""1D-CNN SQI gate — numpy-only inference for the public artifact.

Replaces the variance-bottom-10% gate in src/sqi.py with a tiny learned
classifier that predicts whether the classical FFT-path HR estimator will
succeed on a given band-passed PPG window.

The training pipeline (PyTorch + GPU) is intentionally NOT part of this
repository. Public artifact = numpy forward pass + accept_mask integration +
the result numbers. Weights live in `models/sqi_cnn_v1.npz` (also gitignored —
not part of the public artifact).

Architecture (~1k parameters):

    1 ch input (512 samples, z-scored band-passed PPG)
      │
      ▼  Conv1d(1 → 8, kernel=15, padding=7)  +  ReLU
      ▼  MaxPool1d(4)                          →  (8, 128)
      ▼  Conv1d(8 → 16, kernel=7, padding=3)  +  ReLU
      ▼  MaxPool1d(4)                          →  (16, 32)
      ▼  AdaptiveAvgPool1d(1)                  →  (16,)
      ▼  Linear(16 → 1)                        →  scalar logit
      ▼  sigmoid                               →  acceptance probability

Default acceptance threshold = 0.45 (chosen on the val set to maintain
≥65 % acceptance rate; lower threshold → more permissive).
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = ROOT / "models" / "sqi_cnn_v1.npz"
DEFAULT_THRESHOLD = 0.45
WIN_LEN = 512               # 8 s @ 64 Hz
WIN_LEN_BIDMC = None        # see _resample_to_win_len()


def _conv1d_valid(x: np.ndarray, w: np.ndarray, b: np.ndarray, padding: int) -> np.ndarray:
    """1-D convolution with explicit padding. Mirrors PyTorch's Conv1d.

    x : (N, C_in, L_in)         — input
    w : (C_out, C_in, K)        — weight
    b : (C_out,)                — bias
    returns : (N, C_out, L_in + 2*padding - K + 1)
    """
    if padding:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding)), mode="constant")
    n, c_in, L = x.shape
    c_out, c_in_w, K = w.shape
    assert c_in == c_in_w, (c_in, c_in_w)
    out_L = L - K + 1
    # Cross-correlation (PyTorch convention), not flipped convolution.
    out = np.zeros((n, c_out, out_L), dtype=np.float32)
    for k in range(K):
        # x_slice : (n, c_in, out_L);  w_slice : (c_out, c_in)
        x_slice = x[:, :, k:k + out_L]
        w_slice = w[:, :, k]
        # einsum: 'ncl,oc->nol'
        out += np.einsum("ncl,oc->nol", x_slice, w_slice)
    out += b[None, :, None]
    return out


def _maxpool1d(x: np.ndarray, kernel: int) -> np.ndarray:
    """Non-overlapping max-pool. Mirrors PyTorch MaxPool1d(kernel) defaults."""
    n, c, L = x.shape
    out_L = L // kernel
    x = x[:, :, :out_L * kernel].reshape(n, c, out_L, kernel)
    return x.max(axis=-1)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class CNNSQIModel:
    """Numpy forward pass + per-window acceptance mask."""

    def __init__(self, weights_path: Path = DEFAULT_WEIGHTS,
                 threshold: float = DEFAULT_THRESHOLD):
        if not weights_path.exists():
            raise FileNotFoundError(
                f"{weights_path} not found. The CNN-SQI weights are not "
                f"redistributed with the public repo (see private/ + models/ "
                f"gitignore entries). Train locally via "
                f"`uv run private/train_sqi_cnn.py`.")
        w = np.load(weights_path)
        # Layer 0: Conv1d
        self.w0 = w["0__weight"].astype(np.float32)
        self.b0 = w["0__bias"].astype(np.float32)
        # Layer 3: Conv1d (after MaxPool1d at index 2)
        self.w3 = w["3__weight"].astype(np.float32)
        self.b3 = w["3__bias"].astype(np.float32)
        # Layer 8: Linear (after AdaptiveAvgPool1d at index 6 + Flatten at 7)
        self.w8 = w["8__weight"].astype(np.float32)
        self.b8 = w["8__bias"].astype(np.float32)
        self.threshold = threshold

    def forward(self, X: np.ndarray) -> np.ndarray:
        """X: (N, WIN_LEN) float32 — z-scored band-passed PPG windows.

        Returns (N,) acceptance probabilities in [0, 1].
        """
        if X.ndim == 1:
            X = X[None, :]
        if X.shape[1] != WIN_LEN:
            raise ValueError(f"X must have second dim {WIN_LEN}, got {X.shape}")
        x = X.astype(np.float32)[:, None, :]                # (N, 1, 512)
        x = _conv1d_valid(x, self.w0, self.b0, padding=7)   # (N, 8, 512)
        x = _relu(x)
        x = _maxpool1d(x, 4)                                 # (N, 8, 128)
        x = _conv1d_valid(x, self.w3, self.b3, padding=3)   # (N, 16, 128)
        x = _relu(x)
        x = _maxpool1d(x, 4)                                 # (N, 16, 32)
        x = x.mean(axis=-1)                                  # GAP → (N, 16)
        logits = x @ self.w8.T + self.b8                     # (N, 1)
        return _sigmoid(logits[:, 0])

    def accept(self, X: np.ndarray) -> np.ndarray:
        """Boolean mask: True → window accepted (good quality)."""
        return self.forward(X) >= self.threshold


def cnn_accept_mask(windows: np.ndarray,
                    weights_path: Path = DEFAULT_WEIGHTS,
                    threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    """Convenience wrapper. `windows` : (N, WIN_LEN) z-scored float."""
    return CNNSQIModel(weights_path, threshold).accept(windows)
