"""1D-CNN SQI vs variance SQI — A/B comparison on PPG-DaLiA.

For each held-out PPG-DaLiA subject, runs the FFT-path HR estimator over 8-s
windows, then evaluates both gates:

  variance gate : reject bottom 10 % by band-passed window variance
  CNN gate      : 1D-CNN classifier trained on the other 13 subjects

Metric: MAE on accepted windows (vs ECG-derived ground truth). The gate trade-
off is acceptance rate vs accuracy on what's accepted — a useful gate keeps
most windows AND lowers the error on the ones it keeps.

Outputs:
  results/sqi_comparison.json
  results/sqi_comparison.md
  results/sqi_comparison.png   (per-subject bar + pooled scatter)

Usage:
  uv run python src/sqi_compare.py                          # S6 (only committed test subject)
  uv run python src/sqi_compare.py --subjects S6,S15        # if S15.npz is locally available
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

from load_dalia import load_dalia_record, list_cache  # noqa: E402
from reference import (  # noqa: E402
    design_fir_bandpass, design_fft_tables, fft_hr_x100_reference, fir_apply,
)
from sqi import variance_threshold, window_variance  # noqa: E402
from sqi_cnn import CNNSQIModel, WIN_LEN, DEFAULT_THRESHOLD  # noqa: E402

WIN_S = 8.0
HOP_S = 2.0
FS_BVP = 64.0
WIN_LEN_PPG = int(WIN_S * FS_BVP)         # 512
HOP_LEN_PPG = int(HOP_S * FS_BVP)         # 128
TAPS = 51
F1, F2 = 0.7, 3.5
FFT_N = 256
FFT_DECIM = 4


def evaluate_subject(subject_id: str, cnn: CNNSQIModel) -> dict:
    bvp, fs, ref_t, ref_hr = load_dalia_record(subject_id)
    bvp = bvp - bvp.mean()
    m = float(np.max(np.abs(bvp))) or 1.0
    bvp = 0.9 * bvp / m
    h = design_fir_bandpass(TAPS, F1, F2, fs)
    fs_ds, k_min, k_max, hamming_q, _ = design_fft_tables(
        n_fft=FFT_N, fs_in=fs, decim=FFT_DECIM, hr_lo_hz=F1, hr_hi_hz=F2)
    n = len(bvp)
    rows = []
    for s in range(0, n - WIN_LEN_PPG + 1, HOP_LEN_PPG):
        e = s + WIN_LEN_PPG
        x = bvp[s:e]
        yf_list = fir_apply(x.tolist(), h)
        yf = np.asarray(yf_list, dtype=np.float32)
        if len(yf) < WIN_LEN_PPG:
            yf = np.pad(yf, (WIN_LEN_PPG - len(yf), 0), mode="edge")
        hr_fft_x100 = fft_hr_x100_reference(
            yf_list, fs, FFT_N, FFT_DECIM, hamming_q, k_min, k_max)
        hr_fft = hr_fft_x100 / 100.0
        # Reference HR at window centre
        t_c = s / fs + WIN_S / 2.0
        idx = int(np.argmin(np.abs(ref_t - t_c)))
        ref = float(ref_hr[idx])
        if np.isnan(ref) or ref <= 0:
            continue
        var = window_variance(yf_list)
        rows.append({"yf": yf, "hr_fft": hr_fft, "ref": ref, "variance": var})

    if not rows:
        return {"n": 0}

    # Variance gate: bottom 10 % rejected
    variances = [r["variance"] for r in rows]
    var_thr = variance_threshold(variances, pct=10.0)
    accept_var = [v >= var_thr for v in variances]

    # CNN gate
    X = np.stack([r["yf"] for r in rows])
    sd = X.std(axis=1, keepdims=True); sd[sd == 0] = 1.0
    Xz = (X - X.mean(axis=1, keepdims=True)) / sd
    probs = cnn.forward(Xz.astype(np.float32))
    accept_cnn = probs >= DEFAULT_THRESHOLD

    err = np.array([abs(r["hr_fft"] - r["ref"]) for r in rows])

    def stats(mask):
        m = np.asarray(mask, dtype=bool)
        if not m.any():
            return {"mae_bpm": float("nan"), "acceptance_pct": 0.0, "n_accepted": 0}
        return {
            "mae_bpm": float(err[m].mean()),
            "acceptance_pct": float(100.0 * m.sum() / len(m)),
            "n_accepted": int(m.sum()),
        }

    return {
        "subject": subject_id,
        "n_windows": len(rows),
        "variance": stats(accept_var),
        "cnn": stats(accept_cnn),
        "err": err.tolist(),
        "probs": probs.tolist(),
        "accept_var": [bool(x) for x in accept_var],
        "accept_cnn": [bool(x) for x in accept_cnn],
    }


def _plot(results: list[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    subjects = [r["subject"] for r in results if r.get("n_windows", 0)]
    var_mae = [r["variance"]["mae_bpm"] for r in results if r.get("n_windows", 0)]
    cnn_mae = [r["cnn"]["mae_bpm"] for r in results if r.get("n_windows", 0)]
    var_acc = [r["variance"]["acceptance_pct"] for r in results if r.get("n_windows", 0)]
    cnn_acc = [r["cnn"]["acceptance_pct"] for r in results if r.get("n_windows", 0)]

    x = np.arange(len(subjects))
    width = 0.4
    ax1.bar(x - width/2, var_mae, width, label="variance gate", color="#888")
    ax1.bar(x + width/2, cnn_mae, width, label="CNN gate", color="#1f77b4")
    ax1.set_xticks(x); ax1.set_xticklabels(subjects)
    ax1.set_ylabel("MAE on accepted windows (bpm)")
    ax1.set_title("HR-MAE: CNN-SQI vs variance gate")
    ax1.legend()
    ax1.grid(True, axis="y", alpha=0.3)
    for i, (v, c) in enumerate(zip(var_mae, cnn_mae)):
        ax1.text(i - width/2, v + 0.3, f"{v:.1f}", ha="center", fontsize=9, color="#444")
        ax1.text(i + width/2, c + 0.3, f"{c:.1f}", ha="center", fontsize=9, color="#1f77b4")

    ax2.bar(x - width/2, var_acc, width, label="variance gate", color="#888")
    ax2.bar(x + width/2, cnn_acc, width, label="CNN gate", color="#1f77b4")
    ax2.axhline(65, color="red", linestyle="--", alpha=0.5, label="65 % gate")
    ax2.set_xticks(x); ax2.set_xticklabels(subjects)
    ax2.set_ylabel("Acceptance rate (%)")
    ax2.set_title("Window acceptance rate")
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subjects", default="S6",
                   help="comma list; default S6 (the committed test subject)")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = p.parse_args()

    available = set(list_cache())
    subjects = [s.strip() for s in args.subjects.split(",")
                if s.strip() in available]
    if not subjects:
        print(f"no subjects found in cache. Available: {sorted(available)}", file=sys.stderr)
        return 1

    cnn = CNNSQIModel()
    print(f"[sqi_compare] evaluating {len(subjects)} subjects with CNN threshold = {DEFAULT_THRESHOLD}")
    results = []
    for s in subjects:
        r = evaluate_subject(s, cnn)
        results.append(r)
        v = r["variance"]; c = r["cnn"]
        print(f"  {s}: n={r['n_windows']}  "
              f"variance MAE={v['mae_bpm']:.2f} ({v['acceptance_pct']:.1f}% accept)  "
              f"|  CNN MAE={c['mae_bpm']:.2f} ({c['acceptance_pct']:.1f}% accept)  "
              f"|  Δ-MAE = {v['mae_bpm'] - c['mae_bpm']:+.2f} bpm",
              flush=True)

    # Pool
    if len(results) > 0:
        all_var_n = sum(r["variance"]["n_accepted"] for r in results)
        all_cnn_n = sum(r["cnn"]["n_accepted"] for r in results)
        # Pool MAE = sum of (n * MAE) / sum n
        pool_var_mae = sum(r["variance"]["mae_bpm"] * r["variance"]["n_accepted"]
                           for r in results if r["variance"]["n_accepted"]) / max(1, all_var_n)
        pool_cnn_mae = sum(r["cnn"]["mae_bpm"] * r["cnn"]["n_accepted"]
                           for r in results if r["cnn"]["n_accepted"]) / max(1, all_cnn_n)
        n_total = sum(r["n_windows"] for r in results)
        pool_var_acc = 100.0 * all_var_n / max(1, n_total)
        pool_cnn_acc = 100.0 * all_cnn_n / max(1, n_total)
        print(f"  POOLED ({n_total} windows): variance {pool_var_mae:.2f} bpm "
              f"({pool_var_acc:.1f}%)  |  CNN {pool_cnn_mae:.2f} bpm "
              f"({pool_cnn_acc:.1f}%)  |  Δ-MAE = {pool_var_mae - pool_cnn_mae:+.2f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    out_json = args.out_dir / "sqi_comparison.json"
    summary = {
        "config": {"threshold": DEFAULT_THRESHOLD, "win_s": WIN_S, "hop_s": HOP_S,
                   "fs_bvp": FS_BVP, "subjects": subjects},
        "per_subject": [{"subject": r["subject"],
                         "n_windows": r["n_windows"],
                         "variance": r["variance"],
                         "cnn": r["cnn"]} for r in results],
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out_json}")

    out_png = args.out_dir / "sqi_comparison.png"
    _plot(results, out_png)
    print(f"wrote {out_png}")

    out_md = args.out_dir / "sqi_comparison.md"
    with open(out_md, "w") as f:
        f.write("# 1D-CNN SQI vs variance SQI — held-out PPG-DaLiA subjects\n\n")
        f.write("Both gates run on the same FFT-path HR estimates over 8-s "
                "windows. Variance gate rejects the bottom 10 % of windows by "
                "band-passed variance; CNN gate uses a tiny 1D-CNN "
                "(~1k parameters) trained on the other 13 subjects.\n\n")
        f.write("| Subject | n windows | variance MAE | variance accept | CNN MAE | CNN accept | Δ-MAE |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for r in results:
            v = r["variance"]; c = r["cnn"]
            d = v["mae_bpm"] - c["mae_bpm"]
            f.write(f"| {r['subject']} | {r['n_windows']} | "
                    f"{v['mae_bpm']:.2f} | {v['acceptance_pct']:.1f}% | "
                    f"{c['mae_bpm']:.2f} | {c['acceptance_pct']:.1f}% | "
                    f"{d:+.2f} |\n")
        if len(results) > 1:
            f.write(f"| **POOLED** | {n_total} | "
                    f"{pool_var_mae:.2f} | {pool_var_acc:.1f}% | "
                    f"{pool_cnn_mae:.2f} | {pool_cnn_acc:.1f}% | "
                    f"{pool_var_mae - pool_cnn_mae:+.2f} |\n")
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
