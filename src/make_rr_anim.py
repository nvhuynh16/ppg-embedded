"""Respiration-rate (BW-path) animation for a README embed.

Per-frame pipeline (mirrors firmware/main_rr.c on a 30-s sliding window):
  lowpass FIR (0.5 Hz) → decimate by 32 → 24-bin Goertzel scan over the
  respiration band → argmax → RR

Three rows:
  • raw PPG with the respiratory baseline modulation visible (top)
  • lowpass + decimated signal (the channel the Goertzel actually sees)
  • 24-bin Goertzel magnitude² bars, peak bin highlighted (middle, large)
  • big RR readout (bottom)

Outputs:
  results/web/pipeline_rr.gif
  results/web/pipeline_rr.mp4

Run:
  uv run python src/make_rr_anim.py
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from respiration import design_fir_lowpass, goertzel_scan, RR_FREQS_HZ  # noqa: E402
from _anim_helpers import (  # noqa: E402
    load_bidmc, style_axes, setup_figure, add_watermark, save_anim,
    BG, FG, SUBTLE, RAW, FILT, SPEC_BAR, SPEC_PK, HR_OK, HR_OFF,
)

ROOT = SRC.parent


def _fir_apply(x: np.ndarray, h: list[float]) -> np.ndarray:
    """Same convolution form as reference.py::fir_apply, but vectorised."""
    return np.convolve(x, np.asarray(h), mode="valid")


def _rr_window(sig_window: np.ndarray, fs: float, h_lp: list[float],
               decim: int, freqs_hz: tuple[float, ...]):
    """One firmware-equivalent RR pipeline pass. Returns
    (decimated_signal, mag2_array, k_peak, rr_brpm)."""
    filt = _fir_apply(sig_window, h_lp)
    ds = filt[::decim].tolist()
    if len(ds) < 8:
        return np.asarray(ds), np.zeros(len(freqs_hz)), 0, 0.0
    fs_ds = fs / decim
    mag2 = goertzel_scan(ds, fs_ds, freqs_hz=freqs_hz)
    mag2_arr = np.asarray(mag2)
    k_peak = int(np.argmax(mag2_arr))
    rr_brpm = freqs_hz[k_peak] * 60.0
    return np.asarray(ds), mag2_arr, k_peak, rr_brpm


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--record",    default="bidmc01")
    ap.add_argument("--cache-dir", default=str(ROOT / "data" / "bidmc_cache"))
    ap.add_argument("--window-s",  type=float, default=30.0,
                    help="RR analysis window length (s; matches firmware default)")
    ap.add_argument("--advance-s", type=float, default=0.5,
                    help="signal-time advance per frame (s). Default 0.5 is the "
                         "slow-motion rate; set 1.0 for a faster slide.")
    ap.add_argument("--frames",    type=int,   default=160)
    ap.add_argument("--fps",       type=int,   default=20)
    ap.add_argument("--lp-taps",   type=int,   default=51)
    ap.add_argument("--lp-cutoff", type=float, default=0.5,
                    help="lowpass cutoff (Hz) — matches reference.py default")
    ap.add_argument("--decim",     type=int,   default=32)
    ap.add_argument("--out-dir",   default=str(ROOT / "results" / "web"))
    args = ap.parse_args()

    sig, fs = load_bidmc(args.record, Path(args.cache_dir))
    h_lp = design_fir_lowpass(args.lp_taps, args.lp_cutoff, fs)
    fir_offset = args.lp_taps - 1

    window_n  = int(round(args.window_s  * fs))
    advance_n = int(round(args.advance_s * fs))

    # Clamp frames if signal too short
    end_n = fir_offset + window_n + advance_n * args.frames
    if end_n > len(sig):
        max_frames = max(1, (len(sig) - fir_offset - window_n) // advance_n + 1)
        print(f"WARNING: clamping frames {args.frames} → {max_frames}", file=sys.stderr)
        args.frames = max_frames
        end_n = fir_offset + window_n + advance_n * args.frames

    sig_max = float(np.max(np.abs(sig[fir_offset:end_n])))

    # Pre-decimate the full signal once for the time-domain "lowpass output" row
    full_filt = _fir_apply(sig, h_lp)
    full_ds = full_filt[::args.decim]
    ds_max = float(np.max(np.abs(full_ds))) or 1.0
    fs_ds = fs / args.decim

    plt.style.use("dark_background")
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.0),
                             gridspec_kw={"height_ratios": [0.9, 1.4, 0.55]})
    setup_figure(fig)
    style_axes(axes)
    ax_time, ax_spec, ax_rr = axes

    ax_time.set_ylim(-sig_max * 1.1, sig_max * 1.1)
    line_raw, = ax_time.plot([], [], color=RAW, lw=1.2, alpha=0.55, label="raw PPG")
    line_ds,  = ax_time.plot([], [], color=FILT, lw=1.8,
                             label=f"lowpass {args.lp_cutoff:.1f} Hz · decim {args.decim}× (fs_ds={fs_ds:.2f} Hz)")
    ax_time.set_title(f"Raw PPG + lowpass output  —  PhysioNet BIDMC {args.record}",
                      color=FG, loc="left", fontsize=12, pad=6, fontweight="semibold")
    ax_time.set_xlabel("time (s)", color=SUBTLE, fontsize=10)
    ax_time.legend(loc="upper right", fontsize=8, facecolor=BG, edgecolor=SUBTLE,
                   labelcolor=FG)
    # Secondary y-axis for the decimated signal so both fit
    ax_ds = ax_time.twinx()
    ax_ds.set_ylim(-ds_max * 1.5, ds_max * 1.5)
    line_ds.set_transform(ax_ds.transData)
    ax_ds.tick_params(colors=SUBTLE, labelsize=7)
    for s in ax_ds.spines.values():
        s.set_color(SUBTLE); s.set_linewidth(0.5)

    # Goertzel scan bars over the 24 candidate frequencies
    freqs_brpm = [f * 60 for f in RR_FREQS_HZ]
    n_bars = len(RR_FREQS_HZ)
    bars = ax_spec.bar(range(n_bars), np.zeros(n_bars),
                       color=SPEC_BAR, width=0.85, alpha=0.85)
    ax_spec.set_xticks(range(0, n_bars, 4))
    ax_spec.set_xticklabels([f"{freqs_brpm[i]:.0f}" for i in range(0, n_bars, 4)])
    ax_spec.set_xlabel("respiration rate (BrPM)", color=SUBTLE, fontsize=10)
    ax_spec.set_ylabel("Goertzel mag² (normalised)", color=SUBTLE, fontsize=10)
    ax_spec.set_title(
        f"24-bin Q15 Goertzel scan  —  6–30 BrPM @ 1-BrPM resolution",
        color=FG, loc="left", fontsize=12, pad=6, fontweight="semibold")

    # RR readout row
    ax_rr.set_xlim(0, 1); ax_rr.set_ylim(0, 1); ax_rr.axis("off")
    rr_text = ax_rr.text(0.5, 0.62, "—", ha="center", va="center",
                         fontsize=58, color=HR_OFF, fontweight="bold")
    ax_rr.text(0.5, 0.18, "breaths / minute  (BW-path argmax)",
               ha="center", va="center", fontsize=11, color=SUBTLE)
    status_text = ax_rr.text(0.02, 0.5, "", ha="left", va="center",
                             fontsize=10, color=SUBTLE, family="monospace")

    add_watermark(fig)
    fig.subplots_adjust(left=0.07, right=0.92, top=0.93, bottom=0.07, hspace=0.55)

    def update(frame: int):
        start_n = fir_offset + frame * advance_n
        end_n_  = start_n + window_n

        t = np.arange(start_n, end_n_) / fs
        line_raw.set_data(t, sig[start_n:end_n_])

        # Decimated signal slice (already pre-computed)
        # filt[i] corresponds to sig[i + fir_offset]; ds = filt[::decim], so
        # ds[j] corresponds to sig[j * decim + fir_offset].
        ds_lo = max(0, (start_n - fir_offset) // args.decim)
        ds_hi = max(ds_lo + 1, (end_n_ - fir_offset) // args.decim)
        ds_hi = min(ds_hi, len(full_ds))
        t_ds = (np.arange(ds_lo, ds_hi) * args.decim + fir_offset) / fs
        line_ds.set_data(t_ds, full_ds[ds_lo:ds_hi])
        ax_time.set_xlim(t[0], t[-1])

        # Goertzel scan on the current 30-s window
        sig_window = sig[start_n:end_n_]
        _ds, mag2, k_peak, rr_brpm = _rr_window(
            sig_window, fs, h_lp, args.decim, RR_FREQS_HZ)

        m_max = float(mag2.max()) or 1.0
        heights = mag2 / m_max
        for i, b in enumerate(bars):
            b.set_height(heights[i])
            b.set_color(SPEC_PK if i == k_peak else SPEC_BAR)
        ax_spec.set_ylim(0, 1.05)

        rr_text.set_text(f"{rr_brpm:.1f}")
        rr_text.set_color(HR_OK)
        status_text.set_text(
            f"window:  {start_n/fs:5.1f} – {end_n_/fs:5.1f} s\n"
            f"k_peak = {k_peak:2d}/24   RR = {rr_brpm:5.2f} BrPM")
        return [line_raw, line_ds, rr_text, status_text, *bars]

    print(f"[anim] rendering {args.frames} frames @ {args.fps} fps "
          f"({args.frames / args.fps:.1f} s clip)")
    ani = FuncAnimation(fig, update, frames=args.frames,
                        interval=1000 / args.fps, blit=False)
    save_anim(ani, Path(args.out_dir), "pipeline_rr", args.fps)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
