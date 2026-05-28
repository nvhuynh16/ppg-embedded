"""Bit-exactness contract: Python (numpy) ↔ C (Cortex-M3 / QEMU) CNN inference.

For the deterministic synthetic PPG window baked into
`firmware/generated/cnn_data.h::cnn_test_input`, verifies that the C kernel
(`firmware/dsp_cnn.c`) produces the same logit as the numpy reference
(`src/sqi_cnn.py`). Holds regardless of whether the header was generated from
real trained weights or the `--stub` placeholder — what's tested is the C/Python
mathematical equivalence, not the model's predictive value.

Run:
  uv run python src/verify_cnn.py
"""
from __future__ import annotations
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
FIRMWARE = ROOT / "firmware"
sys.path.insert(0, str(SRC))

LOGIT_TOL_X1E6 = 100   # ±100 in the ×1e6 encoding = ±1e-4 in float logit space
                       # (libgcc soft-float vs IEEE-754 float64 of numpy gives
                       # a tiny last-place delta on this many MACs)


def _parse_float_array(header: str, name: str) -> np.ndarray:
    m = re.search(rf"{name}\[\d+\]\s*=\s*\{{([^}}]+)\}};", header, re.DOTALL)
    if not m:
        raise RuntimeError(f"{name} not found in cnn_data.h")
    nums = re.findall(r"[+-]?\d+\.\d+e[+-]?\d+", m.group(1))
    return np.asarray([float(n) for n in nums], dtype=np.float32)


def _parse_define(header: str, name: str) -> int:
    m = re.search(rf"#define\s+{name}\s+([+-]?\d+)", header)
    if not m:
        raise RuntimeError(f"{name} #define not found in cnn_data.h")
    return int(m.group(1))


def _conv1d_float(x: np.ndarray, w: np.ndarray, b: np.ndarray, padding: int) -> np.ndarray:
    n, c_in, L = x.shape
    c_out, _, K = w.shape
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding)))
    out = np.zeros((n, c_out, L), dtype=np.float32)
    for k in range(K):
        out += np.einsum("ncl,oc->nol", xp[:, :, k:k + L], w[:, :, k])
    return out + b[None, :, None]


def _maxpool1d(x: np.ndarray, k: int) -> np.ndarray:
    n, c, L = x.shape
    return x[:, :, :(L // k) * k].reshape(n, c, L // k, k).max(axis=-1)


def _cnn_logit_from_header(header_text: str, test_in: np.ndarray) -> float:
    """Forward pass using ONLY the weights parsed out of cnn_data.h.
    Returns the float logit. Mirrors src/sqi_cnn.py + firmware/dsp_cnn.c."""
    w0 = _parse_float_array(header_text, "cnn_w0").reshape(8, 1, 15)
    b0 = _parse_float_array(header_text, "cnn_b0")
    w3 = _parse_float_array(header_text, "cnn_w3").reshape(16, 8, 7)
    b3 = _parse_float_array(header_text, "cnn_b3")
    w8 = _parse_float_array(header_text, "cnn_w8").reshape(1, 16)
    b8 = _parse_float_array(header_text, "cnn_b8")

    x = test_in[None, None, :].astype(np.float32)
    x = np.maximum(_conv1d_float(x, w0, b0, padding=7), 0.0)
    x = _maxpool1d(x, 4)
    x = np.maximum(_conv1d_float(x, w3, b3, padding=3), 0.0)
    x = _maxpool1d(x, 4)
    x = x.mean(axis=-1)
    logit = (x @ w8.T + b8).reshape(-1)
    return float(logit[0])


def _build_and_run() -> tuple[int, bool]:
    """Build firmware_sqi.elf and run it under QEMU; returns (logit_x1e6, accept)."""
    print("  building firmware_sqi.elf...", flush=True)
    subprocess.run(["make", "-C", str(FIRMWARE), "firmware_sqi.elf"],
                   check=True, capture_output=True)
    print("  running under QEMU...", flush=True)
    out = subprocess.run(["bash", str(FIRMWARE / "run_qemu.sh"), "firmware_sqi.elf"],
                         check=True, capture_output=True, text=True, timeout=60)
    # QEMU semihosting output can come on either stdout or stderr depending
    # on host config — concatenate both.
    log = out.stdout + out.stderr
    m_logit = re.search(r"LOGIT_X1E6=(\d+)\s+(pos|neg)", log)
    m_accept = re.search(r"ACCEPT=(\d)", log)
    if not m_logit or not m_accept:
        print("QEMU output didn't parse:\n" + log, file=sys.stderr)
        raise RuntimeError("LOGIT_X1E6 or ACCEPT missing")
    sign = -1 if m_logit.group(2) == "neg" else 1
    return sign * int(m_logit.group(1)), bool(int(m_accept.group(1)))


def main() -> int:
    print("[verify_cnn] reading cnn_data.h")
    header = (FIRMWARE / "generated" / "cnn_data.h").read_text()
    test_in = _parse_float_array(header, "cnn_test_input")
    if test_in.size != 512:
        raise RuntimeError(f"cnn_test_input has {test_in.size} elements; expected 512")
    expected_x1e6 = _parse_define(header, "CNN_TEST_EXPECTED_LOGIT_X1E6")
    print(f"  test input: {test_in.size} floats   header EXPECTED_LOGIT_X1E6 = {expected_x1e6:+d}")

    # Numpy forward pass using the SAME weights baked into cnn_data.h
    # (so the contract holds regardless of whether the header was emitted
    # from real or stub weights).
    print("[verify_cnn] running numpy forward pass on header-parsed weights")
    logit_numpy = _cnn_logit_from_header(header, test_in)
    logit_numpy_x1e6 = int(round(logit_numpy * 1e6))
    print(f"  numpy LOGIT_X1E6 = {logit_numpy_x1e6:+d}")

    # C forward via QEMU
    c_logit_x1e6, c_accept = _build_and_run()
    print(f"  C     LOGIT_X1E6 = {c_logit_x1e6:+d}   ACCEPT = {c_accept}")

    delta_header = abs(c_logit_x1e6 - expected_x1e6)
    delta_numpy = abs(c_logit_x1e6 - logit_numpy_x1e6)
    print(f"  |C − header| = {delta_header}  |C − numpy| = {delta_numpy}")

    ok = delta_header <= LOGIT_TOL_X1E6 and delta_numpy <= LOGIT_TOL_X1E6
    print("=== PASS ===" if ok else "=== FAIL ===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
