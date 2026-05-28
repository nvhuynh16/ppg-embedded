"""1D-CNN SQI animation for a README embed (PPG-DaLiA, sit→stairs transition).

Mirrors `make_fft_anim.py`'s 3-row layout but with the held-out PPG-DaLiA
test subject (S6) as the signal source and the CNN-SQI gate as the
acceptance test. Picks a 60-second window centred on the sit → stairs
transition at t ≈ 920 s — the moment the wrist sensor goes from clean
contact to motion-corrupted.

Three rows:
  • Time-domain BVP, with each 2-second segment coloured by the CNN's
    accept/reject decision (mint = accepted, gray = unreliable).
  • Float-pipeline FFT magnitude² spectrum in the HR band — same as the
    BIDMC FFT animation, but for the wrist sensor.
  • Big HR readout. Shows the numeric HR when the CNN accepts the current
    8-s window; shows "--" (the smart-watch convention for low signal
    quality) when it doesn't.

Requires:
  - data/dalia_cache/extracts/S6.npz   (committed)
  - models/sqi_cnn_v1.npz              (LOCAL — see private/train_sqi_cnn.py)

Outputs:
  results/web/pipeline_sqi.gif
  results/web/pipeline_sqi.mp4

Run:
  uv run python src/make_sqi_anim.py
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

from reference import design_fir_bandpass, fir_apply, design_fft_tables, fft_hr_x100_reference  # noqa: E402
from load_dalia import load_dalia_record, ACTIVITY_NAMES  # noqa: E402
from sqi_cnn import CNNSQIModel, WIN_LEN, DEFAULT_THRESHOLD  # noqa: E402
from _anim_helpers import (  # noqa: E402
    style_axes, setup_figure, add_watermark, save_anim,
    BG, FG, SUBTLE, FILT, SPEC_BAR, SPEC_PK, HR_OK, HR_OFF,
)

ACCEPT_COLOR = HR_OK     # mint/blue — reliable
REJECT_COLOR = HR_OFF    # gray      — unreliable (smart-watch "--" cue)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject",   default="S6")
    ap.add_argument("--start-s",   type=float, default=895.0,
                    help="signal time the cursor starts at (s). 920s is the "
                         "sit→stairs transition for S6.")
    ap.add_argument("--frames",    type=int, default=80,
                    help="one frame = 1 s of signal time (default).")
    ap.add_argument("--fps",       type=int, default=20)
    ap.add_argument("--visible-s", type=float, default=30.0)
    ap.add_argument("--window-s",  type=float, default=8.0,
                    help="CNN / FFT analysis window length (s).")
    ap.add_argument("--seg-s",     type=float, default=2.0,
                    help="time-domain segment size for colouring (s)")
    ap.add_argument("--taps",      type=int,   default=51)
    ap.add_argument("--f1",        type=float, default=0.7)
    ap.add_argument("--f2",        type=float, default=3.5)
    ap.add_argument("--fft-n",     type=int,   default=256)
    ap.add_argument("--fft-decim", type=int,   default=4)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--out-dir",   default=str(ROOT / "results" / "web"))
    args = ap.parse_args()

    bvp, fs, _ref_t, _ref_hr, activity = load_dalia_record(args.subject, want_activity=True)
    assert fs == 64.0, fs
    # Normalise to match the inference pipeline's expectations
    bvp = bvp - bvp.mean()
    bvp = 0.9 * bvp / (float(np.max(np.abs(bvp))) or 1.0)

    h = design_fir_bandpass(args.taps, args.f1, args.f2, fs)
    fs_ds, k_min, k_max, hamming_q, _ = design_fft_tables(
        n_fft=args.fft_n, fs_in=fs, decim=args.fft_decim,
        hr_lo_hz=args.f1, hr_hi_hz=args.f2)
    hamming_w = np.asarray(hamming_q, dtype=float) / 32768.0

    cnn = CNNSQIModel(threshold=args.threshold)

    win_n      = int(round(args.window_s  * fs))   # 512
    visible_n  = int(round(args.visible_s * fs))   # 1920
    seg_n      = int(round(args.seg_s     * fs))   # 128
    start_n    = int(round(args.start_s   * fs))

    # Precompute per-frame: HR (FFT-path) + CNN accept-prob over the 8-s
    # window ENDING at the cursor.
    per_frame = []
    for f in range(args.frames):
        cursor_n  = start_n + f * int(fs)            # cursor advances 1 s / frame
        win_start = cursor_n - win_n
        win_end   = cursor_n
        if win_start < 0 or win_end > len(bvp):
            break
        x = bvp[win_start:win_end]
        yf = fir_apply(x.tolist(), h)
        yf_arr = np.asarray(yf, dtype=np.float32)
        if len(yf_arr) < win_n:
            yf_arr = np.pad(yf_arr, (win_n - len(yf_arr), 0), mode="edge")
        sd = float(yf_arr.std()) or 1.0
        yf_z = (yf_arr - float(yf_arr.mean())) / sd
        prob = float(cnn.forward(yf_z.astype(np.float32)[None, :])[0])
        accept = prob >= args.threshold

        hr_x100 = fft_hr_x100_reference(yf, fs, args.fft_n, args.fft_decim,
                                         hamming_q, k_min, k_max)
        hr_bpm = hr_x100 / 100.0

        # Spectrum for the visual
        ds = yf_arr[::args.fft_decim][:args.fft_n]
        if len(ds) < args.fft_n:
            ds = np.concatenate([ds, np.zeros(args.fft_n - len(ds))])
        windowed = ds * hamming_w
        mag2 = np.abs(np.fft.fft(windowed)) ** 2
        band = mag2[k_min:k_max + 1]
        k_peak = int(k_min + np.argmax(band))

        # Activity at cursor (4 Hz grid)
        act_idx = min(len(activity) - 1, int(cursor_n / fs * 4))
        act_name = ACTIVITY_NAMES[int(activity[act_idx])]

        per_frame.append({
            "cursor_n": cursor_n, "win_start": win_start, "win_end": win_end,
            "hr_bpm": hr_bpm, "accept": accept, "prob": prob,
            "band": band, "k_peak": k_peak, "activity": act_name,
        })

    n_frames = len(per_frame)
    print(f"[anim] precomputed {n_frames} frames; CNN accept-rate over clip = "
          f"{100.0 * sum(f['accept'] for f in per_frame) / max(1, n_frames):.1f}%")

    # Per-segment accept/reject map for the line plot. The "segment" is each
    # `seg_s` chunk of signal; its colour comes from the CNN decision of the
    # frame whose cursor falls inside that segment.
    seg_accepts: dict[int, bool] = {}      # key: segment-index in signal-space
    for f in per_frame:
        seg_idx = f["cursor_n"] // seg_n
        seg_accepts[seg_idx] = f["accept"]

    # Y-axis limits for the visible region (use cushion over actual range)
    end_n = per_frame[-1]["win_end"]
    sig_max = float(np.max(np.abs(bvp[max(0, start_n - visible_n):end_n + 1])))

    plt.style.use("dark_background")
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.0),
                             gridspec_kw={"height_ratios": [0.9, 1.4, 0.55]})
    setup_figure(fig)
    style_axes(axes)
    ax_time, ax_spec, ax_hr = axes

    # Time-domain row — coloured by segment via LineCollection
    ax_time.set_ylim(-sig_max * 1.1, sig_max * 1.1)
    ax_time.set_title(f"Wrist PPG (BVP)  —  PPG-DaLiA {args.subject}",
                      color=FG, loc="left", fontsize=12, pad=6, fontweight="semibold")
    ax_time.set_xlabel("time (s)", color=SUBTLE, fontsize=10)
    line_filt, = ax_time.plot([], [], color=FILT, lw=1.2, alpha=0.35, label="band-pass FIR")
    line_filt._zorder = 1
    lc = LineCollection([], linewidths=1.8, alpha=0.95)
    ax_time.add_collection(lc)
    ax_time.legend(loc="upper right", fontsize=8, facecolor=BG, edgecolor=SUBTLE, labelcolor=FG)

    # Spectrum row
    n_bars = k_max - k_min + 1
    bar_x = np.arange(k_min, k_max + 1)
    bars = ax_spec.bar(bar_x, np.zeros(n_bars), color=SPEC_BAR, width=0.85, alpha=0.85)
    ax_spec.set_xlim(k_min - 0.5, k_max + 0.5)
    ax_spec.set_xlabel("FFT bin  (HR band)", color=SUBTLE, fontsize=10)
    ax_spec.set_ylabel("|X[k]|² (normalised)", color=SUBTLE, fontsize=10)
    ax_spec.set_title("256-pt FFT spectrum (same path as firmware_fft)",
                      color=FG, loc="left", fontsize=12, pad=6, fontweight="semibold")
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
    hr_text = ax_hr.text(0.5, 0.62, "--", ha="center", va="center",
                         fontsize=58, color=HR_OFF, fontweight="bold")
    bpm_subtitle = ax_hr.text(0.5, 0.18, "bpm  (1D-CNN SQI gate, threshold = 0.45)",
                              ha="center", va="center", fontsize=11, color=SUBTLE)
    activity_text = ax_hr.text(0.02, 0.5, "", ha="left", va="center",
                               fontsize=10, color=SUBTLE, family="monospace")
    prob_text = ax_hr.text(0.98, 0.5, "", ha="right", va="center",
                           fontsize=10, color=SUBTLE, family="monospace")

    add_watermark(fig)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.93, bottom=0.07, hspace=0.55)

    def update(frame_idx: int):
        f = per_frame[frame_idx]
        cursor_n = f["cursor_n"]
        vis_end   = cursor_n
        vis_start = max(0, vis_end - visible_n)

        t_vis = np.arange(vis_start, vis_end) / fs
        y_vis = bvp[vis_start:vis_end]
        line_filt.set_data(t_vis, y_vis)
        ax_time.set_xlim(t_vis[0], t_vis[-1])

        # Build colour segments: split visible window into `seg_n`-sample chunks,
        # each coloured by the corresponding CNN decision (default to "no decision
        # yet" → gray for segments past the current cursor's reach).
        segs = []
        colors = []
        s = vis_start
        while s < vis_end:
            e = min(s + seg_n, vis_end)
            t_seg = np.arange(s, e) / fs
            y_seg = bvp[s:e]
            pts = np.column_stack([t_seg, y_seg])
            segs.append(pts)
            seg_idx = s // seg_n
            # Use the FIRST frame whose cursor falls AFTER this segment's start.
            # If we haven't seen the cursor cross the segment yet, treat as
            # unknown (gray, lighter alpha).
            ac = seg_accepts.get(seg_idx, None)
            colors.append(ACCEPT_COLOR if ac else
                          (REJECT_COLOR if ac is False else "#3b4148"))
            s = e
        lc.set_segments(segs)
        lc.set_colors(colors)

        # Spectrum
        band = f["band"]
        m_max = float(band.max()) or 1.0
        heights = band / m_max
        for i, b in enumerate(bars):
            b.set_height(heights[i])
            b.set_color(SPEC_PK if (k_min + i) == f["k_peak"] else SPEC_BAR)
        ax_spec.set_ylim(0, 1.05)

        # HR readout — "--" when CNN rejects, like a smart watch
        if f["accept"]:
            hr_text.set_text(f"{f['hr_bpm']:.0f}")
            hr_text.set_color(HR_OK)
        else:
            hr_text.set_text("--")
            hr_text.set_color(HR_OFF)
        activity_text.set_text(f"t = {cursor_n/fs:5.0f}s\nactivity = {f['activity']}")
        prob_text.set_text(f"CNN p(accept) = {f['prob']:.2f}\n"
                           f"{'accepted' if f['accept'] else 'rejected':>14s}")
        return [line_filt, hr_text, activity_text, prob_text, lc, *bars]

    print(f"[anim] rendering {n_frames} frames @ {args.fps} fps "
          f"({n_frames / args.fps:.1f} s clip; saved at slowed playback)")
    ani = FuncAnimation(fig, update, frames=n_frames,
                        interval=1000 / args.fps, blit=False)
    save_anim(ani, Path(args.out_dir), "pipeline_sqi", args.fps,
              gif_speed=1.0 / 3.0, mp4_speed=1.0 / 4.0)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
