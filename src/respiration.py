"""Respiration rate (RR) estimation from PPG — Python reference.

This module ships **only the BW path** — the one the firmware
(`main_rr.c`) implements. The Karlen-2013 three-channel pipeline
(AM + FM + smart fusion) is `src/_respiration_three_channel_draft.py`
(a Python draft, not in firmware).

  - BW (Baseline Wander) — intrathoracic pressure modulates PPG DC at the
    respiration rate (0.1–0.5 Hz, i.e. 6–30 breaths/min). Extracted by
    lowpass-filtering the raw PPG below 0.5 Hz and Goertzel-scanning the
    respiration band.

Why BW-only for the firmware: empirical validation on BIDMC bidmc01–03
(n=12 windows) showed the AM path consistently locked to the lowest
candidate frequency (DC residual artifact) and the FM path was
high-variance. Smart-fusion as written was therefore worse than BW alone.
AM/FM/fusion live in the draft module for future work; firmware ships the
working channel only.

Reference: Karlen W, Raman S, Ansermino JM, Dumont GA. "Multiparameter
respiratory rate estimation from the photoplethysmogram." IEEE TBME
60(7):1946-53 (2013).
"""
from __future__ import annotations

import math
from typing import Tuple

# Goertzel candidate-frequency grid (also imported by make_rr_anim.py and
# the draft three-channel module). 0.10 .. 0.50 Hz at ~0.017 Hz steps =
# 24 bins → 6 .. 30 BrPM at 1 BrPM resolution. Just enough for
# clinical-grade RR.
RR_FREQS_HZ = tuple(0.10 + 0.4 * k / 23 for k in range(24))


# ============================================================================
# FIR lowpass design (Hamming-window, same style as design_fir_bandpass)
# ============================================================================

def design_fir_lowpass(num_taps: int, cutoff_hz: float, fs: float) -> list[float]:
    """Hamming-window lowpass FIR. Even/odd `num_taps` both OK; odd is symmetric.

    Returns a list of float taps summing to ~1.0 (DC gain ≈ 1).
    """
    n_half = (num_taps - 1) / 2.0
    fc = cutoff_hz / fs                              # normalized cutoff
    taps: list[float] = []
    for k in range(num_taps):
        m = k - n_half
        if abs(m) < 1e-12:
            ideal = 2.0 * fc                         # sinc(0) limit
        else:
            ideal = math.sin(2.0 * math.pi * fc * m) / (math.pi * m)
        # Hamming window
        w = 0.54 - 0.46 * math.cos(2.0 * math.pi * k / (num_taps - 1))
        taps.append(ideal * w)
    # Normalize so DC gain = 1 exactly.
    s = sum(taps) or 1.0
    return [t / s for t in taps]


# ============================================================================
# Goertzel algorithm
# ============================================================================

def goertzel(samples: list[float], freq_hz: float, fs: float) -> float:
    """Single-frequency Goertzel. Returns |X[f]|² (real, unnormalized).

    The Q15 firmware version (`dsp_resp.c::goertzel_q15`) implements the
    identical recurrence with scaled coefficients.
    """
    n = len(samples)
    if n == 0:
        return 0.0
    omega = 2.0 * math.pi * freq_hz / fs
    coef = 2.0 * math.cos(omega)
    s_prev = 0.0
    s_prev2 = 0.0
    for x in samples:
        s = x + coef * s_prev - s_prev2
        s_prev2 = s_prev
        s_prev = s
    # Magnitude² (real-valued spectrum estimate at the target frequency)
    return s_prev * s_prev + s_prev2 * s_prev2 - coef * s_prev * s_prev2


def goertzel_scan(samples: list[float], fs: float,
                  freqs_hz: tuple[float, ...] = RR_FREQS_HZ) -> list[float]:
    """Run Goertzel at each candidate frequency; return mag² array."""
    return [goertzel(samples, f, fs) for f in freqs_hz]


def pick_peak_freq(mag2: list[float],
                   freqs_hz: tuple[float, ...] = RR_FREQS_HZ) -> Tuple[float, float]:
    """argmax-mag² frequency. Returns (freq_hz, peak_mag2). The 1-BrPM bin
    width is already the clinically meaningful resolution; no parabolic
    interp here."""
    if not mag2:
        return 0.0, 0.0
    k_peak = max(range(len(mag2)), key=lambda i: mag2[i])
    return freqs_hz[k_peak], mag2[k_peak]


def hz_to_brpm(f_hz: float) -> float:
    return f_hz * 60.0


# ============================================================================
# Channel 1 — Baseline Wander (BW) — the firmware path
# ============================================================================

def bw_path(sig: list[float], fs: float,
            cutoff_hz: float = 0.5, lp_taps: int = 51,
            decim: int = 32) -> Tuple[float, float]:
    """Baseline-wander RR.

    Apply a wide-band lowpass (≤ 0.5 Hz cutoff), decimate to fs_ds ≈ fs/decim,
    then Goertzel-scan the respiration band.

    Returns (rr_brpm, peak_mag2). peak_mag2 is a confidence proxy.
    """
    h = design_fir_lowpass(lp_taps, cutoff_hz, fs)
    # FIR (straightforward; matches `dsp_fixed.c::fir_q15` semantics)
    n = len(sig)
    ntaps = len(h)
    filt: list[float] = []
    for i in range(ntaps - 1, n):
        acc = 0.0
        for k in range(ntaps):
            acc += h[k] * sig[i - k]
        filt.append(acc)
    # Decimate
    fs_ds = fs / decim
    ds = filt[::decim]
    if len(ds) < 8:
        return 0.0, 0.0
    mag2 = goertzel_scan(ds, fs_ds)
    f, peak = pick_peak_freq(mag2)
    return hz_to_brpm(f), peak


# ============================================================================
# Top-level entry point
# ============================================================================

def estimate_rr(sig: list[float], fs: float) -> dict:
    """RR estimator — BW path only. This is the function the firmware
    `main_rr.c` mirrors.

    Inputs:
      sig — raw normalized PPG (same array passed to the existing band-pass
            FIR for the HR path; full sample rate fs).
      fs  — sample rate (Hz).

    Returns:
      {"rr_brpm": float, "rr_bw": float, "quality": {"bw_mag2": float}}
    """
    rr_bw, bw_q = bw_path(sig, fs)
    return {"rr_brpm": rr_bw, "rr_bw": rr_bw, "quality": {"bw_mag2": bw_q}}


# ============================================================================
# Self-test (BW path on a clean synthetic signal)
# ============================================================================

def _self_test() -> int:
    """Synthesise a PPG with known HR + RR, run BW path, check it's close."""
    fs = 125.0
    duration = 60.0                                 # 60 s → enough cycles at 15 BrPM
    hr_bpm = 72.0
    rr_brpm_true = 15.0
    n = int(duration * fs)

    f_hr = hr_bpm / 60.0
    f_rr = rr_brpm_true / 60.0
    # Carrier + baseline modulation at f_rr (what the BW path detects).
    sig: list[float] = []
    for i in range(n):
        t = i / fs
        baseline = 0.4 * math.sin(2 * math.pi * f_rr * t + 0.7)
        pulse = math.sin(2 * math.pi * f_hr * t)
        sig.append(baseline + pulse)

    print(f"=== respiration.py self-test (BW path) ===")
    print(f"  truth: HR={hr_bpm} bpm  RR={rr_brpm_true} BrPM   ({duration:.0f} s synthetic)")
    rr_bw, bw_q = bw_path(sig, fs)
    print(f"  rr_bw = {rr_bw:6.2f} BrPM  (q={bw_q:.3e})")
    err = abs(rr_bw - rr_brpm_true)
    if err > 2.0:
        print(f"FAIL: BW err = {err:.2f} BrPM exceeds 2 BrPM tolerance")
        return 1
    print(f"  PASS: BW within {err:.2f} BrPM of truth")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
