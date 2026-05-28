"""FFT spectral-path animation for a README embed.

Per-frame pipeline (mirrors firmware/main_fft.c on a 30-s sliding window):
  band-pass FIR → decimate by 8 → Hamming → FFT → argmax in HR band
  → sub-harmonic check at k_peak/2 → parabolic interp → HR

Three rows:
  • raw PPG + Q15 band-pass FIR output overlaid (top)
  • FFT magnitude² spectrum in the HR band, with the selected bin marked
    in red and the sub-harmonic candidate marked in amber when the swap fires
  • big HR readout

Outputs:
  results/web/pipeline_fft.gif
  results/web/pipeline_fft.mp4

Run:
  uv run python src/make_fft_anim.py
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
from reference import design_fir_bandpass, fir_apply, design_fft_tables  # noqa: E402
from _anim_helpers import (  # noqa: E402
    load_bidmc, style_axes, setup_figure, add_watermark, save_anim,
    BG, FG, SUBTLE, RAW, FILT, SPEC_BAR, SPEC_PK, SPEC_SUB, HR_OK, HR_OFF,
)

ROOT = SRC.parent


def _fft_window_hr(sig_window: np.ndarray, fs: float, h: list[float],
                   fft_n: int, fft_decim: int, hamming_w: np.ndarray,
                   k_min: int, k_max: int, subharmonic_divisor: int = 2):
    """One firmware-equivalent FFT pipeline pass. Returns
    (mag2_in_band, k_peak, k_sub_swapped, hr_bpm).

    `mag2_in_band` is the spectrum slice over [k_min..k_max] for the visual.
    `k_sub_swapped` is the sub-harmonic bin index ONLY if the swap fired,
    else None.
    """
    filt = np.array(fir_apply(sig_window.tolist(), h))
    # Decimate (filt is shorter than sig_window by ntaps-1; take every nth)
    ds = filt[::fft_decim][:fft_n]
    if len(ds) < fft_n:
        ds = np.concatenate([ds, np.zeros(fft_n - len(ds))])
    windowed = ds * hamming_w
    spectrum = np.fft.fft(windowed)
    mag2 = np.abs(spectrum) ** 2

    # Initial peak-bin search in HR band
    band = mag2[k_min:k_max + 1]
    k_peak = int(k_min + np.argmax(band))
    max_mag2 = float(mag2[k_peak])

    # Sub-harmonic check (mirrors main_fft.c stage 5b)
    k_sub_swapped = None
    k_sub = k_peak // 2
    if k_sub >= k_min:
        sub_mag2 = float(mag2[k_sub])
        if sub_mag2 >= max_mag2 / subharmonic_divisor:
            k_sub_swapped = k_peak                # remember the original peak
            k_peak = k_sub
            max_mag2 = sub_mag2

    # Parabolic interpolation for sub-bin resolution
    delta = 0.0
    if 1 <= k_peak < fft_n - 1:
        y_m1, y_0, y_p1 = float(mag2[k_peak - 1]), float(mag2[k_peak]), float(mag2[k_peak + 1])
        den = y_m1 - 2 * y_0 + y_p1
        if den != 0:
            delta = 0.5 * (y_m1 - y_p1) / den
            delta = max(-0.5, min(0.5, delta))

    fs_ds = fs / fft_decim
    f_peak_hz = (k_peak + delta) * fs_ds / fft_n
    hr_bpm = f_peak_hz * 60.0
    return mag2[k_min:k_max + 1], k_peak, k_sub_swapped, hr_bpm


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--record",    default="bidmc01")
    ap.add_argument("--cache-dir", default=str(ROOT / "data" / "bidmc_cache"))
    ap.add_argument("--window-s",  type=float, default=30.0,
                    help="FFT analysis window length (s; matches firmware default)")
    ap.add_argument("--visible-s", type=float, default=8.0,
                    help="raw/filtered visible window length (s)")
    ap.add_argument("--advance-s", type=float, default=0.5,
                    help="signal-time advance per frame (s). Default 0.5 is the "
                         "slow-motion rate; set 1.0 for a faster slide.")
    ap.add_argument("--frames",    type=int,   default=160)
    ap.add_argument("--fps",       type=int,   default=20)
    ap.add_argument("--taps",      type=int,   default=101)
    ap.add_argument("--f1",        type=float, default=0.7)
    ap.add_argument("--f2",        type=float, default=3.5)
    ap.add_argument("--fft-n",     type=int,   default=256)
    ap.add_argument("--fft-decim", type=int,   default=8)
    ap.add_argument("--subharmonic-divisor", type=int, default=3,
                    help="α = 1/DIVISOR (default = 3, i.e. α ≈ 0.33)")
    ap.add_argument("--out-dir",   default=str(ROOT / "results" / "web"))
    args = ap.parse_args()

    sig, fs = load_bidmc(args.record, Path(args.cache_dir))
    h = design_fir_bandpass(args.taps, args.f1, args.f2, fs)
    full_filt = np.array(fir_apply(sig.tolist(), h))
    fir_offset = args.taps - 1

    fs_ds, k_min, k_max, hamming_q, _twiddle = design_fft_tables(
        n_fft=args.fft_n, fs_in=fs, decim=args.fft_decim,
        hr_lo_hz=args.f1, hr_hi_hz=args.f2)
    hamming_w = np.asarray(hamming_q, dtype=float) / 32768.0

    fft_window_n     = int(round(args.window_s  * fs))
    visible_n        = int(round(args.visible_s * fs))
    advance_n        = int(round(args.advance_s * fs))

    # Ensure each frame can fit a full FFT window
    end_n = fir_offset + fft_window_n + advance_n * args.frames
    if end_n > len(sig):
        max_frames = (len(sig) - fir_offset - fft_window_n) // advance_n + 1
        print(f"WARNING: clamping frames {args.frames} → {max_frames}", file=sys.stderr)
        args.frames = max(1, max_frames)

    # Static y-axis limits
    sig_max  = float(np.max(np.abs(sig[fir_offset:end_n])))
    filt_max = float(np.max(np.abs(full_filt[:end_n - fir_offset])))

    plt.style.use("dark_background")
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.0),
                             gridspec_kw={"height_ratios": [0.9, 1.4, 0.55]})
    setup_figure(fig)
    style_axes(axes)
    ax_time, ax_spec, ax_hr = axes

    # Time-domain row: raw on a thin subplot + filtered overlaid
    ax_time.set_ylim(-sig_max * 1.1, sig_max * 1.1)
    line_raw,  = ax_time.plot([], [], color=RAW,  lw=1.2, alpha=0.55, label="raw PPG")
    line_filt, = ax_time.plot([], [], color=FILT, lw=1.6, label="Q15 band-pass FIR")
    ax_time.set_title(f"Raw PPG + band-pass FIR  —  PhysioNet BIDMC {args.record}",
                      color=FG, loc="left", fontsize=12, pad=6, fontweight="semibold")
    ax_time.set_xlabel("time (s)", color=SUBTLE, fontsize=10)
    ax_time.legend(loc="upper right", fontsize=8, facecolor=BG, edgecolor=SUBTLE,
                   labelcolor=FG)

    # Spectrum row: bars over k_min..k_max
    n_bars = k_max - k_min + 1
    bar_x = np.arange(k_min, k_max + 1)
    bars = ax_spec.bar(bar_x, np.zeros(n_bars), color=SPEC_BAR, width=0.85, alpha=0.85)
    ax_spec.set_xlim(k_min - 0.5, k_max + 0.5)
    ax_spec.set_xlabel("FFT bin  (HR band)", color=SUBTLE, fontsize=10)
    ax_spec.set_ylabel("|X[k]|² (normalised)", color=SUBTLE, fontsize=10)
    ax_spec.set_title(
        f"256-pt Q15 FFT spectrum  +  sub-harmonic check  (α = 1/{args.subharmonic_divisor})",
        color=FG, loc="left", fontsize=12, pad=6, fontweight="semibold")
    # Secondary x-axis in bpm so the reader can read HR off the bin axis
    ax_bpm = ax_spec.twiny()
    ax_bpm.set_xlim(ax_spec.get_xlim())
    bpm_tick_bins = np.linspace(k_min, k_max, 5, dtype=int)
    ax_bpm.set_xticks(bpm_tick_bins)
    ax_bpm.set_xticklabels([f"{b * fs_ds / args.fft_n * 60:.0f}" for b in bpm_tick_bins])
    ax_bpm.set_xlabel("HR (bpm)", color=SUBTLE, fontsize=9)
    ax_bpm.tick_params(colors=SUBTLE, labelsize=8)
    for s in ax_bpm.spines.values():
        s.set_color(SUBTLE); s.set_linewidth(0.5)

    # HR row
    ax_hr.set_xlim(0, 1); ax_hr.set_ylim(0, 1); ax_hr.axis("off")
    hr_text = ax_hr.text(0.5, 0.62, "—", ha="center", va="center",
                         fontsize=58, color=HR_OFF, fontweight="bold")
    ax_hr.text(0.5, 0.18, "bpm  (FFT peak-bin + parabolic interp)",
               ha="center", va="center", fontsize=11, color=SUBTLE)
    status_text = ax_hr.text(0.02, 0.5, "", ha="left", va="center",
                             fontsize=10, color=SUBTLE, family="monospace")

    add_watermark(fig)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.93, bottom=0.07, hspace=0.55)

    def update(frame: int):
        fft_start = fir_offset + frame * advance_n
        fft_end   = fft_start + fft_window_n
        # Visible-window slice for top row (last `visible_n` samples)
        vis_end   = fft_end
        vis_start = max(fft_start, vis_end - visible_n)

        t_vis = np.arange(vis_start, vis_end) / fs
        line_raw.set_data(t_vis, sig[vis_start:vis_end])
        f_s = max(0, vis_start - fir_offset)
        f_e = vis_end - fir_offset
        t_filt = (np.arange(f_s, f_e) + fir_offset) / fs
        line_filt.set_data(t_filt, full_filt[f_s:f_e])
        ax_time.set_xlim(t_vis[0], t_vis[-1])

        # FFT for the full 30-s window
        sig_window = sig[fft_start:fft_end]
        mag2_band, k_peak, k_sub_swapped, hr_bpm = _fft_window_hr(
            sig_window, fs, h, args.fft_n, args.fft_decim, hamming_w,
            k_min, k_max, args.subharmonic_divisor)

        # Normalise bar heights to peak = 1.0 for a steady visual
        m_max = float(mag2_band.max()) or 1.0
        heights = mag2_band / m_max

        for i, b in enumerate(bars):
            b.set_height(heights[i])
            bin_idx = k_min + i
            if bin_idx == k_peak:
                b.set_color(SPEC_PK)
            elif k_sub_swapped is not None and bin_idx == k_sub_swapped:
                b.set_color(SPEC_SUB)
            else:
                b.set_color(SPEC_BAR)
        ax_spec.set_ylim(0, 1.05)

        hr_text.set_text(f"{hr_bpm:.1f}")
        hr_text.set_color(HR_OK)
        note = "← sub-harmonic swap fired" if k_sub_swapped is not None else ""
        status_text.set_text(
            f"window:  {fft_start/fs:5.1f} – {fft_end/fs:5.1f} s\n"
            f"k_peak = {k_peak:3d}   HR = {hr_bpm:5.2f} bpm   {note}")
        return [line_raw, line_filt, hr_text, status_text, *bars]

    print(f"[anim] rendering {args.frames} frames @ {args.fps} fps "
          f"({args.frames / args.fps:.1f} s clip)")
    ani = FuncAnimation(fig, update, frames=args.frames,
                        interval=1000 / args.fps, blit=False)
    save_anim(ani, Path(args.out_dir), "pipeline_fft", args.fps)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
