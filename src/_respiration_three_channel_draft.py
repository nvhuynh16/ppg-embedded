"""Karlen-2013 three-channel RR + smart fusion — Python draft.

**Not used by firmware.** Firmware ships the BW path only (see
`src/respiration.py` and `firmware/main_rr.c`). This module holds the
deferred AM (amplitude modulation) and FM (frequency modulation / RSA)
channels plus the smart-fusion combiner.

Why this lives in a separate file:
- Keeps `respiration.py` lean (BW only, all of it shipped in firmware)
- Signals clearly to a code reviewer that this is reference / stretch work
- Future implementation can iterate here without touching the shipped
  surface

What still needs work before this can ship:
1. Quality-weighted smart fusion that disqualifies low-confidence paths
   (current AM path consistently locks onto the lowest candidate frequency
   on real BIDMC PPG — likely DC residual or low-frequency noise winning
   the argmax over the genuine respiratory tone).
2. DC-detrended AM amplitude time series + better trough localisation.
3. FM tuning to suppress the DC residual that's currently winning the
   argmax on some windows.
4. Bit-exactness verifier between this Python reference and the future
   Q15 firmware Goertzel.

Reference: Karlen W, Raman S, Ansermino JM, Dumont GA. "Multiparameter
respiratory rate estimation from the photoplethysmogram." IEEE TBME
60(7):1946-53 (2013).

The self-test passes on a clean synthetic signal — the algorithm is
correct in principle; the failure on real PPG is the DC-residual issue
above.
"""
from __future__ import annotations

from typing import Optional, Tuple

from respiration import goertzel_scan, pick_peak_freq, hz_to_brpm, bw_path

# Smart-fusion agreement threshold (BrPM). Karlen 2013 used 4 BrPM.
FUSION_TOL_BRPM = 4.0


# ============================================================================
# Channel 2 — Amplitude Modulation (AM)
# ============================================================================

def _trough_before(filt: list[float], peak_idx: int, prev_peak_idx: int) -> int:
    """Index of the local minimum between two peaks. If empty range, returns
    `peak_idx - 1` as a safe fallback."""
    lo = max(prev_peak_idx + 1, 0)
    hi = peak_idx
    if hi <= lo:
        return max(peak_idx - 1, 0)
    rng = filt[lo:hi]
    return lo + min(range(len(rng)), key=lambda i: rng[i])


def am_path(filt_bp: list[float], peak_indices: list[int], fs: float,
            fs_beat: float = 4.0) -> Tuple[float, float]:
    """Amplitude-modulation RR.

    For each detected peak (after the first), measure peak amplitude minus
    the preceding trough's value. This gives a beat-rate time series sampled
    irregularly at the peak times. Resample linearly to fs_beat, then
    Goertzel-scan.
    """
    if len(peak_indices) < 4:
        return 0.0, 0.0
    amps_t: list[float] = []
    amps_v: list[float] = []
    for i in range(1, len(peak_indices)):
        p_i = peak_indices[i]
        t_i = _trough_before(filt_bp, p_i, peak_indices[i - 1])
        amps_t.append(p_i / fs)
        amps_v.append(float(filt_bp[p_i] - filt_bp[t_i]))
    uniform = _linear_resample(amps_t, amps_v, fs_beat)
    if len(uniform) < 8:
        return 0.0, 0.0
    mag2 = goertzel_scan(uniform, fs_beat)
    f, peak = pick_peak_freq(mag2)
    return hz_to_brpm(f), peak


# ============================================================================
# Channel 3 — Frequency Modulation (FM)
# ============================================================================

def fm_path(peak_indices: list[int], fs: float,
            fs_beat: float = 4.0) -> Tuple[float, float]:
    """Frequency-modulation (RSA) RR.

    Inter-beat intervals from peak_indices, converted to instantaneous-HR
    samples (60 / interval_s), resampled to fs_beat, Goertzel-scanned.
    """
    if len(peak_indices) < 4:
        return 0.0, 0.0
    intervals_t: list[float] = []
    intervals_v: list[float] = []
    for i in range(1, len(peak_indices)):
        dt = (peak_indices[i] - peak_indices[i - 1]) / fs
        if dt <= 0:
            continue
        # Place the sample at the midpoint between the two peaks
        intervals_t.append((peak_indices[i] + peak_indices[i - 1]) / (2.0 * fs))
        intervals_v.append(60.0 / dt)               # instantaneous BPM
    uniform = _linear_resample(intervals_t, intervals_v, fs_beat)
    if len(uniform) < 8:
        return 0.0, 0.0
    # Subtract mean so the DC bin doesn't dominate the spectral scan.
    mean = sum(uniform) / len(uniform)
    uniform = [v - mean for v in uniform]
    mag2 = goertzel_scan(uniform, fs_beat)
    f, peak = pick_peak_freq(mag2)
    return hz_to_brpm(f), peak


# ============================================================================
# Linear resample (irregular → uniform grid) — shared by AM + FM
# ============================================================================

def _linear_resample(ts: list[float], vs: list[float], fs_out: float) -> list[float]:
    """Linear interpolate (ts, vs) onto a uniform grid at sample rate fs_out."""
    if len(ts) < 2:
        return []
    t0, t1 = ts[0], ts[-1]
    n_out = int((t1 - t0) * fs_out) + 1
    if n_out < 2:
        return []
    out: list[float] = []
    j = 0
    for k in range(n_out):
        t = t0 + k / fs_out
        while j + 1 < len(ts) and ts[j + 1] < t:
            j += 1
        if j + 1 >= len(ts):
            out.append(vs[-1])
        else:
            t_a, t_b = ts[j], ts[j + 1]
            v_a, v_b = vs[j], vs[j + 1]
            frac = 0.0 if t_b == t_a else (t - t_a) / (t_b - t_a)
            out.append(v_a + frac * (v_b - v_a))
    return out


# ============================================================================
# Smart fusion (Karlen 2013)
# ============================================================================

def smart_fusion(rr_bw: float, rr_am: float, rr_fm: float,
                 tol_brpm: float = FUSION_TOL_BRPM) -> Optional[float]:
    """Keep estimate only when ≥ 2 of 3 paths agree within ±tol_brpm.

    Returns mean of agreeing estimates, or None if no two-of-three agreement.
    A `0.0` from any path is treated as "this path declined to estimate".
    """
    paths = [v for v in (rr_bw, rr_am, rr_fm) if v > 0]
    if len(paths) < 2:
        return None
    # All-three agreement check
    if len(paths) == 3:
        pairs = [(paths[i], paths[j]) for i in range(3) for j in range(i + 1, 3)]
        if all(abs(a - b) <= tol_brpm for a, b in pairs):
            return sum(paths) / 3.0
    # Two-of-three: find any pair within tolerance, take their mean
    best_pair = None
    best_diff = float("inf")
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            d = abs(paths[i] - paths[j])
            if d <= tol_brpm and d < best_diff:
                best_pair = (paths[i], paths[j])
                best_diff = d
    if best_pair is not None:
        return (best_pair[0] + best_pair[1]) / 2.0
    return None


# ============================================================================
# Top-level entry point
# ============================================================================

def estimate_rr_three_channel(sig: list[float], fs: float,
                              filt_bp: list[float],
                              peak_indices: list[int]) -> dict:
    """Three-channel pipeline: BW + AM + FM → smart fusion.

    **Not used by firmware.** Reference implementation for future firmware
    port; see module docstring for the deferred-work caveats.
    """
    rr_bw, bw_q = bw_path(sig, fs)
    rr_am, am_q = am_path(filt_bp, peak_indices, fs)
    rr_fm, fm_q = fm_path(peak_indices, fs)
    rr_fused = smart_fusion(rr_bw, rr_am, rr_fm)
    return {
        "rr_brpm": rr_fused,
        "rr_bw": rr_bw, "rr_am": rr_am, "rr_fm": rr_fm,
        "quality": {"bw_mag2": bw_q, "am_mag2": am_q, "fm_mag2": fm_q},
    }
