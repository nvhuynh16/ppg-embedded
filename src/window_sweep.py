"""Window-length sweep for the HR validator.

Re-runs the BIDMC validation pipeline with window ∈ {8, 15, 30, 60, 120} s on
a small subset of records (default: bidmc01-03) and plots how MAE, %within±3,
%within±5, and algorithm-failure-rate trade off vs window length.

Defends (or revises) the 30-s default reported in `results/README.md`. The
trade is:
  - shorter windows → lower latency but more failures (peak detector needs
    ≥ 3 peaks for the median-interval HR to be meaningful; FFT needs enough
    samples for sub-bpm bin resolution at the decimated sample rate)
  - longer windows → fewer windows per record (fewer i.i.d. estimates), and
    intra-window non-stationarity (HR changes during a long window) caps how
    close the median-window HR can get to the ECG reference

Outputs:
  results/window_sweep.csv  — one row per (window_s, record) with metrics
  results/window_sweep.png  — 4-panel plot, x = window_s, y = each metric
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from batch_validate import process_record, compute_metrics, failure_stats  # noqa: E402

ROOT = SRC.parent
RESULTS = ROOT / "results"


def run_sweep(records: list[str], window_sizes_s: list[float], cache_dir: Path,
              taps: int, f1: float, f2: float) -> list[dict]:
    """Run process_record for each (record, window_s) combination."""
    summary: list[dict] = []
    for w in window_sizes_s:
        all_rows: list[dict] = []
        for rec_id in records:
            print(f"[w={w:>5.1f}s] {rec_id}...", flush=True)
            rows = process_record(rec_id, cache_dir, w, taps, f1, f2,
                                  limit_windows=None)
            all_rows.extend(rows)
        m = compute_metrics(all_rows, subset="estimable")
        f = failure_stats(all_rows)
        summary.append({
            "window_s": w,
            "n_windows_total": len(all_rows),
            "n_estimable": m.get("n", 0),
            "mae_bpm": m.get("mae_bpm", float("nan")),
            "rmse_bpm": m.get("rmse_bpm", float("nan")),
            "pct_within_3bpm": m.get("pct_within_3bpm", float("nan")),
            "pct_within_5bpm": m.get("pct_within_5bpm", float("nan")),
            "pearson_r": m.get("pearson_r", float("nan")),
            "failure_rate_pct": f.get("failure_rate_pct", float("nan")),
        })
        print(f"  → n_est={m.get('n', 0)}  MAE={m.get('mae_bpm', float('nan')):.2f}  "
              f"%≤5={m.get('pct_within_5bpm', float('nan')):.1f}  "
              f"fail={f.get('failure_rate_pct', float('nan')):.1f}%",
              flush=True)
    return summary


def emit_plot(summary: list[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    w = np.asarray([r["window_s"] for r in summary])
    mae = np.asarray([r["mae_bpm"] for r in summary])
    p3 = np.asarray([r["pct_within_3bpm"] for r in summary])
    p5 = np.asarray([r["pct_within_5bpm"] for r in summary])
    fail = np.asarray([r["failure_rate_pct"] for r in summary])

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)

    axes[0, 0].plot(w, mae, "o-", color="#1f77b4")
    axes[0, 0].set_ylabel("MAE (bpm)")
    axes[0, 0].set_title("Mean Absolute Error vs window length")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(w, p3, "o-", color="#2ca02c", label="±3 bpm")
    axes[0, 1].plot(w, p5, "s-", color="#9467bd", label="±5 bpm")
    axes[0, 1].set_ylabel("Within tolerance (%)")
    axes[0, 1].set_title("Accuracy bands")
    axes[0, 1].legend(loc="lower right")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 105)

    axes[1, 0].plot(w, fail, "o-", color="#d62728")
    axes[1, 0].set_xlabel("Window length (s)")
    axes[1, 0].set_ylabel("Algorithm-failure rate (%)")
    axes[1, 0].set_title("Failure rate (hr_x100 == 0 sentinel)")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].axvline(30, ls="--", color="grey", alpha=0.6, label="default = 30 s")
    axes[1, 1].plot(w, mae, "o-", color="#1f77b4", label="MAE (bpm)")
    axes[1, 1].plot(w, fail / 10.0, "s-", color="#d62728", label="failure (%)/10")
    axes[1, 1].set_xlabel("Window length (s)")
    axes[1, 1].set_ylabel("(both axes rescaled)")
    axes[1, 1].set_title("Joint trade")
    axes[1, 1].legend(loc="upper right")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", default="bidmc01,bidmc02,bidmc03")
    ap.add_argument("--windows", default="8,15,30,60,120",
                    help="comma-separated window sizes in seconds")
    ap.add_argument("--taps", type=int, default=101)
    ap.add_argument("--f1", type=float, default=0.7)
    ap.add_argument("--f2", type=float, default=3.5)
    ap.add_argument("--cache-dir", default=str(ROOT / "data" / "bidmc_cache"))
    args = ap.parse_args()

    records = args.records.split(",")
    window_sizes = [float(w) for w in args.windows.split(",")]
    RESULTS.mkdir(parents=True, exist_ok=True)

    summary = run_sweep(records, window_sizes, Path(args.cache_dir),
                        args.taps, args.f1, args.f2)

    csv_path = RESULTS / "window_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"wrote {csv_path}")

    emit_plot(summary, RESULTS / "window_sweep.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
