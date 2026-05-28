"""Render the peak-detector pipeline animation for a career-site / README embed.

One of three sister scripts (peak / FFT / RR) sharing src/_anim_helpers.py.

Sliding 8-second window across BIDMC record 01. Three rows:
  • raw PPG (top)
  • Q15 band-passed signal with detected peaks marked (middle)
  • big HR readout (bottom)

Outputs:
  results/web/pipeline.gif   — universal autoplay in <img>
  results/web/pipeline.mp4   — smaller, smoother <video … playsinline>

Run:
  uv run python src/make_peak_anim.py
  uv run python src/make_peak_anim.py --record bidmc02 --frames 200 --fps 24
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
from reference import (design_fir_bandpass, fir_apply, detect_peaks,
                       hr_x100_from_peaks, refractory_samples)
from _anim_helpers import (
    load_bidmc, style_axes, setup_figure, add_watermark, save_anim,
    BG, FG, SUBTLE, RAW, FILT, PEAK, HR_OK, HR_OFF,
)

ROOT = SRC.parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--record",   default="bidmc01")
    ap.add_argument("--cache-dir", default=str(ROOT / "data" / "bidmc_cache"))
    ap.add_argument("--window-s",  type=float, default=8.0,  help="visible window (s)")
    ap.add_argument("--advance-s", type=float, default=0.125,
                    help="signal-time advance per frame (s). Default 0.125 is the "
                         "slow-motion rate — the visible signal traverses the plot "
                         "near real time at fs=125. Set 0.25 for a faster slide.")
    ap.add_argument("--frames",    type=int,   default=240,  help="total animation frames")
    ap.add_argument("--fps",       type=int,   default=30)
    ap.add_argument("--taps",      type=int,   default=101)
    ap.add_argument("--f1",        type=float, default=0.7)
    ap.add_argument("--f2",        type=float, default=3.5)
    ap.add_argument("--out-dir",   default=str(ROOT / "results" / "web"))
    args = ap.parse_args()

    sig, fs = load_bidmc(args.record, Path(args.cache_dir))
    h = design_fir_bandpass(args.taps, args.f1, args.f2, fs)
    filt = np.array(fir_apply(sig.tolist(), h))   # length = len(sig) - (taps-1)
    refractory = refractory_samples(fs)

    # Pre-compute ALL peak indices once (over the entire filt). Per-frame work
    # is then a cheap index-range filter, not a re-scan.
    all_peaks = detect_peaks(filt.tolist(), refractory)
    all_peaks = np.asarray(all_peaks, dtype=int)
    fir_offset = args.taps - 1   # filt[i] corresponds to signal time (i + fir_offset) / fs

    window_n  = int(round(args.window_s  * fs))
    advance_n = int(round(args.advance_s * fs))
    skip_n    = fir_offset                          # start after the FIR transient
    total_n_needed = skip_n + window_n + advance_n * (args.frames - 1)
    if total_n_needed > len(sig):
        max_frames = (len(sig) - skip_n - window_n) // advance_n + 1
        print(f"WARNING: clamping frames {args.frames} → {max_frames} (signal exhausted)",
              file=sys.stderr)
        args.frames = max(1, max_frames)

    # Static y-axis limits so the plot doesn't bounce.
    end_n = skip_n + window_n + advance_n * args.frames
    sig_max  = float(np.max(np.abs(sig[skip_n:end_n])))
    filt_max = float(np.max(np.abs(filt[:end_n - fir_offset])))

    plt.style.use("dark_background")
    fig, axes = plt.subplots(3, 1, figsize=(12, 7.5),
                             gridspec_kw={"height_ratios": [1.0, 1.0, 0.55]})
    setup_figure(fig)
    style_axes(axes)

    ax_raw, ax_filt, ax_hr = axes

    # Raw row
    ax_raw.set_ylim(-sig_max * 1.1, sig_max * 1.1)
    line_raw, = ax_raw.plot([], [], color=RAW, lw=1.4)
    ax_raw.set_title(f"Raw PPG  —  PhysioNet BIDMC {args.record}", color=FG, loc="left",
                     fontsize=12, pad=6, fontweight="semibold")

    # Filtered + peaks row
    ax_filt.set_ylim(-filt_max * 1.1, filt_max * 1.1)
    line_filt, = ax_filt.plot([], [], color=FILT, lw=1.4)
    peaks_scatter = ax_filt.scatter([], [], c=PEAK, s=70, zorder=5,
                                    edgecolors="white", linewidths=1.0)
    ax_filt.set_title("Q15 band-pass FIR  +  peak detection", color=FG, loc="left",
                      fontsize=12, pad=6, fontweight="semibold")
    ax_filt.set_xlabel("time (s)", color=SUBTLE, fontsize=10)

    # HR readout row
    ax_hr.set_xlim(0, 1); ax_hr.set_ylim(0, 1)
    ax_hr.axis("off")
    hr_text = ax_hr.text(0.5, 0.62, "—", ha="center", va="center",
                         fontsize=58, color=HR_OFF, fontweight="bold")
    bpm_text = ax_hr.text(0.5, 0.18, "bpm  (median inter-peak interval)",
                          ha="center", va="center", fontsize=11, color=SUBTLE)
    status_text = ax_hr.text(0.02, 0.5, "", ha="left", va="center",
                             fontsize=10, color=SUBTLE, family="monospace")

    add_watermark(fig)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.94, bottom=0.06, hspace=0.45)

    # Convert filt indices to signal-time once (vectorised).
    peak_times = (all_peaks + fir_offset) / fs
    peak_y     = filt[all_peaks]

    def update(frame: int):
        start_n = skip_n + frame * advance_n
        end_n_  = start_n + window_n
        t = np.arange(start_n, end_n_) / fs

        # Raw
        line_raw.set_data(t, sig[start_n:end_n_])
        ax_raw.set_xlim(t[0], t[-1])

        # Filtered (offset by fir_offset between sig and filt indexing)
        f_start = start_n - fir_offset
        f_end   = end_n_  - fir_offset
        if f_end > 0 and f_start < len(filt):
            f_s = max(0, f_start); f_e = min(len(filt), f_end)
            t_filt = (np.arange(f_s, f_e) + fir_offset) / fs
            line_filt.set_data(t_filt, filt[f_s:f_e])
            ax_filt.set_xlim(t[0], t[-1])

            # Peaks falling inside this window
            mask = (all_peaks >= f_s) & (all_peaks < f_e)
            if mask.any():
                pts = np.column_stack([peak_times[mask], peak_y[mask]])
                peaks_scatter.set_offsets(pts)
                # HR from peaks in this window
                window_peak_idx = all_peaks[mask].tolist()
                hr_x100 = hr_x100_from_peaks(window_peak_idx, fs)
                if hr_x100 > 0:
                    hr_text.set_text(f"{hr_x100 / 100:.1f}")
                    hr_text.set_color(HR_OK)
                else:
                    hr_text.set_text("—")
                    hr_text.set_color(HR_OFF)
                npeaks = int(mask.sum())
            else:
                peaks_scatter.set_offsets(np.empty((0, 2)))
                hr_text.set_text("—"); hr_text.set_color(HR_OFF)
                npeaks = 0
        else:
            npeaks = 0

        status_text.set_text(f"window:  {t[0]:5.1f} – {t[-1]:5.1f} s\npeaks in window: {npeaks}")
        return line_raw, line_filt, peaks_scatter, hr_text, status_text

    print(f"[anim] rendering {args.frames} frames @ {args.fps} fps "
          f"({args.frames / args.fps:.1f} s clip)")
    ani = FuncAnimation(fig, update, frames=args.frames,
                        interval=1000 / args.fps, blit=False)

    # Keep the legacy filename "pipeline" so existing README + Pages embeds
    # don't break. The two sister animations land at pipeline_fft / pipeline_rr.
    save_anim(ani, Path(args.out_dir), "pipeline", args.fps)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
