"""Shared helpers for the three career-site pipeline animations
(make_peak_anim.py, make_fft_anim.py, make_rr_anim.py).

Keeps the colour palette, signal loader, matplotlib axis styling, and
ffmpeg-bundled save logic in one place so the three sister scripts only
differ in what they actually render per frame.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import wfdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

# Point matplotlib at the bundled ffmpeg from imageio-ffmpeg.
try:
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:  # noqa: BLE001
    pass

# GitHub-dark-ish palette — looks good on a portfolio site, prints cleanly on light too.
BG       = "#0d1117"
PANEL_BG = "#0d1117"
FG       = "#c9d1d9"
SUBTLE   = "#8b949e"
RAW      = "#79c0ff"   # blue — raw PPG
FILT     = "#a5f3a5"   # mint — filtered/lowpass output
PEAK     = "#ff7b72"   # red — peak markers
SPEC_BAR = "#a5f3a5"   # mint bar — Goertzel/FFT bins (default)
SPEC_PK  = "#ff7b72"   # red bar — selected peak bin
SPEC_SUB = "#f0883e"   # amber bar — sub-harmonic candidate (when swap fires)
HR_OK    = "#79c0ff"
HR_OFF   = "#484f58"


def load_bidmc(record_id: str, cache_dir: Path) -> tuple[np.ndarray, float]:
    """Mirror src/batch_validate.py::load_bidmc_record's PPG normalisation
    (mean-subtract, scale to peak |0.9|). No reference HR — these animations
    are visual; numbers come from the live estimator."""
    rec = wfdb.rdrecord(str(cache_dir / record_id))
    names = [s.lower() for s in rec.sig_name]
    pleth_idx = next((i for i, nm in enumerate(names)
                      if "pleth" in nm or "ppg" in nm), 1)
    fs = float(rec.fs)
    sig = np.asarray(rec.p_signal[:, pleth_idx], dtype=float)
    sig = sig[~np.isnan(sig)]
    sig = sig - sig.mean()
    m = float(np.max(np.abs(sig))) or 1.0
    return 0.9 * sig / m, fs


def style_axes(axes: Iterable[plt.Axes]) -> None:
    """Dark-theme spines + tick colours for each axis."""
    for ax in axes:
        ax.set_facecolor(PANEL_BG)
        for spine in ax.spines.values():
            spine.set_color(SUBTLE)
            spine.set_linewidth(0.5)
        ax.tick_params(colors=SUBTLE, labelsize=8)
        ax.grid(True, alpha=0.08)


def setup_figure(fig: plt.Figure) -> None:
    fig.patch.set_facecolor(BG)


def add_watermark(fig: plt.Figure, text: str = "nvhuynh16/ppg-embedded  —  Cortex-M3 / QEMU") -> None:
    fig.text(0.99, 0.012, text, ha="right", va="bottom",
             fontsize=8, color="#484f58", alpha=0.9)


def save_anim(ani: FuncAnimation, out_dir: Path, name: str, fps: int,
              gif_speed: float = 1.0 / 3.0,
              mp4_speed: float = 1.0 / 4.0) -> None:
    """Save the animation as both GIF and MP4.

    Playback-rate trim: the same frames are written to both files, but the
    GIF plays at `fps * gif_speed` and the MP4 at `fps * mp4_speed`. Defaults
    are 1/3 (GIF) and 1/4 (MP4) — career-site embeds need the viewer time to
    actually read the live HR / RR numbers, and an autoplay-loop on a
    portfolio page is much more legible at the slower rate. Pass
    `gif_speed=1.0, mp4_speed=1.0` to disable the trim. File sizes are
    unchanged (same N frames; only the per-frame duration field shifts).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    gif_fps = max(1.0, fps * gif_speed)
    mp4_fps = max(1.0, fps * mp4_speed)

    gif_path = out_dir / f"{name}.gif"
    print(f"[anim] writing {gif_path} (gif @ {gif_fps:.2f} fps, 1/{1.0/gif_speed:.0f}× speed) …")
    ani.save(gif_path, writer=PillowWriter(fps=gif_fps), dpi=72)
    print(f"[anim]   {gif_path.name}: {gif_path.stat().st_size / 1024:.0f} KB")

    if FFMpegWriter.isAvailable():
        mp4_path = out_dir / f"{name}.mp4"
        print(f"[anim] writing {mp4_path} (mp4 @ {mp4_fps:.2f} fps, 1/{1.0/mp4_speed:.0f}× speed) …")
        ani.save(mp4_path, writer=FFMpegWriter(fps=mp4_fps, bitrate=1800,
                                                codec="libx264",
                                                extra_args=["-pix_fmt", "yuv420p"]),
                 dpi=96)
        print(f"[anim]   {mp4_path.name}: {mp4_path.stat().st_size / 1024:.0f} KB")
    else:
        print("[anim] MP4 skipped (ffmpeg not available)")
