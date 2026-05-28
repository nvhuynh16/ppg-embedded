"""Head-to-head comparison: peak-detector vs FFT spectral HR on BIDMC.

For each (record, window), build BOTH firmwares (to /tmp, dodging the NTFS
folio_wait hang under tight build loops), run each under QEMU, parse the HR each
emits, and compare both against the BIDMC ECG-derived reference HR.

Reuses src/batch_validate.py's per-window pipeline (signal load, normalise,
header write, ref-HR alignment, SQI gate) so the only thing that varies between
methods is the C-level HR algorithm.

Emits:
  results/method_comparison.csv  (record, t_start_s, hr_peak, hr_fft, hr_ref, accepted)
  results/method_comparison.md   (per-method MAE/RMSE/within-5bpm/r table)
  results/method_comparison.png  (side-by-side scatter plots)
"""
from __future__ import annotations
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from reference import (design_fir_bandpass, fir_apply, q15,
                       refractory_samples, write_ppg_data_header,
                       design_fft_tables, write_fft_data_header)
from sqi import window_variance, variance_threshold
from batch_validate import load_bidmc_record, normalize_signal

from _firmware_io import (  # noqa: E402
    arm_compile, run_qemu_parse, PEAK_SOURCES, FFT_SOURCES,
)

ROOT = SRC.parent
RESULTS = ROOT / "results"
GEN = ROOT / "firmware" / "generated"

TMP_PEAK = Path("/tmp") / "ppg_peak.elf"
TMP_FFT  = Path("/tmp") / "ppg_fft.elf"


def process_record(record_id: str, cache_dir: Path, window_s: float,
                   taps: int, f1: float, f2: float, fft_n: int, fft_decim: int,
                   limit_windows: int | None = None,
                   subharmonic_divisor: int = 3) -> list[dict]:
    sig, fs, ref_t, ref_hr = load_bidmc_record(record_id, cache_dir)
    sig = normalize_signal(sig)

    h = design_fir_bandpass(taps, f1, f2, fs)
    h_q = [q15(v) for v in h]
    refractory = refractory_samples(fs)
    samples_per_window = int(round(window_s * fs))

    # FFT tables for this fs (re-emitted once per record; rarely changes).
    # subharmonic_divisor: 3 = default (α≈0.33, catches all bidmc47 windows);
    # 2 = looser (α=0.5, leaves 3 bidmc47 windows 2× locked).
    fs_ds, k_min, k_max, hamming_q, twiddle = design_fft_tables(
        n_fft=fft_n, fs_in=fs, decim=fft_decim, hr_lo_hz=f1, hr_hi_hz=f2)
    write_fft_data_header(hamming_q, twiddle, fs, fs_ds, fft_decim,
                          k_min, k_max, GEN / "fft_data.h",
                          subharmonic_divisor=subharmonic_divisor)

    rows: list[dict] = []
    variances: list[float] = []
    n_windows = len(sig) // samples_per_window
    if limit_windows is not None:
        n_windows = min(n_windows, limit_windows)

    for i in range(n_windows):
        start = i * samples_per_window
        end = start + samples_per_window
        t_s = start / fs
        x = sig[start:end].tolist()
        x_q = [q15(v) for v in x]

        yf = fir_apply(x, h)
        var = window_variance(yf)
        variances.append(var)

        write_ppg_data_header(x_q, h_q, fs, len(x_q), refractory,
                              ref_hr_x100=0, path=GEN / "ppg_data.h")
        try:
            arm_compile(PEAK_SOURCES, TMP_PEAK)
            arm_compile(FFT_SOURCES,  TMP_FFT)
        except subprocess.TimeoutExpired:
            print(f"  [{record_id} w{i}] build timeout — skip", file=sys.stderr)
            continue

        try:
            hr_peak = run_qemu_parse(TMP_PEAK, "HR_X100")
            hr_fft  = run_qemu_parse(TMP_FFT,  "HR_X100")
        except Exception as e:  # noqa: BLE001
            print(f"  [{record_id} w{i}] QEMU failed: {e}", file=sys.stderr)
            continue

        t_end = t_s + window_s
        mask = (ref_t >= t_s) & (ref_t < t_end)
        if not np.any(mask):
            hr_ref_med, hr_ref_std = float("nan"), float("nan")
        else:
            valid = ref_hr[mask]
            valid = valid[~np.isnan(valid)]
            valid = valid[valid > 0]
            if len(valid) == 0:
                hr_ref_med, hr_ref_std = float("nan"), float("nan")
            else:
                hr_ref_med = float(np.median(valid))
                hr_ref_std = float(np.std(valid))

        rows.append({
            "record": record_id, "t_start_s": round(t_s, 3),
            "hr_peak": round(hr_peak, 3),
            "hr_fft": round(hr_fft, 3),
            "hr_ref_median": round(hr_ref_med, 3) if not np.isnan(hr_ref_med) else float("nan"),
            "hr_ref_std": round(hr_ref_std, 3) if not np.isnan(hr_ref_std) else float("nan"),
            "variance": round(var, 6),
        })
        if (i + 1) % 5 == 0:
            print(f"  [{record_id}] {i+1}/{n_windows} windows done", flush=True)

    if variances:
        threshold = variance_threshold(variances, pct=10.0)
        for r, v in zip(rows, variances):
            ref_med = r["hr_ref_median"]
            ref_std = r["hr_ref_std"]
            r["accepted"] = (
                v >= threshold
                and not (isinstance(ref_med, float) and np.isnan(ref_med))
                and not (isinstance(ref_std, float) and np.isnan(ref_std))
                and ref_std < 10.0
            )
    return rows


def method_metrics(rows: list[dict], hr_col: str) -> dict:
    """Estimable-subset metrics for one method column (hr_peak or hr_fft).

    Includes patient-cluster bootstrap 95 % CIs on every metric — see
    `src/bootstrap.py` for the rationale (within-patient correlation requires
    cluster resampling, not naive window-level resampling).
    """
    valid = [r for r in rows if r.get("accepted", False) and r[hr_col] > 0]
    if not valid:
        return {"n": 0}
    emb = np.array([r[hr_col] for r in valid])
    ref = np.array([r["hr_ref_median"] for r in valid])
    delta = emb - ref
    abs_d = np.abs(delta)
    out = {
        "n": len(valid),
        "mae_bpm": float(np.mean(abs_d)),
        "rmse_bpm": float(np.sqrt(np.mean(delta ** 2))),
        "pct_within_5bpm": float(100.0 * np.mean(abs_d <= 5.0)),
        "pct_within_3bpm": float(100.0 * np.mean(abs_d <= 3.0)),
        "ba_bias_bpm": float(np.mean(delta)),
        "ba_loa_lower": float(np.mean(delta) - 1.96 * np.std(delta)),
        "ba_loa_upper": float(np.mean(delta) + 1.96 * np.std(delta)),
    }
    if len(emb) > 1:
        out["pearson_r"] = float(np.corrcoef(emb, ref)[0, 1])
    # Algorithm-failure rate: of all SQI-accepted, how many returned 0
    acc = [r for r in rows if r.get("accepted", False)]
    failed = [r for r in acc if r[hr_col] == 0]
    out["n_accepted"] = len(acc)
    out["n_failed"] = len(failed)
    out["failure_rate_pct"] = 100.0 * len(failed) / len(acc) if acc else 0.0

    # Bootstrap CIs (patient-cluster, 10k iter, seeded).
    from bootstrap import cluster_bootstrap_ci
    out.update(cluster_bootstrap_ci(rows, n_iter=10_000, seed=0,
                                    emb_col=hr_col, ref_col="hr_ref_median",
                                    record_col="record"))
    return out


def emit_comparison_outputs(rows: list[dict], peak_m: dict, fft_m: dict,
                            out_dir: Path) -> None:
    """Write method_comparison.md (table) and .png (side-by-side scatter)."""
    md = out_dir / "method_comparison.md"
    with open(md, "w") as f:
        f.write("# Method comparison — peak detector vs FFT spectral HR\n\n")
        f.write("Same input signals, same FIR pre-stage, same SQI gate; only the C-level "
                "HR estimator differs. Metrics computed on the **estimable** subset "
                "(SQI-accepted ∧ embedded HR > 0); algorithm-failure rate reported "
                "separately.\n\n")
        f.write("| Metric | Peak detector | FFT spectral |\n")
        f.write("|---|---:|---:|\n")
        def cell(m: dict, k: str, fmt: str = "%.2f") -> str:
            return (fmt % m[k]) if k in m else "—"
        for label, key, fmt in [
            ("n (estimable)",      "n",                "%d"),
            ("MAE (bpm)",          "mae_bpm",          "%.2f"),
            ("RMSE (bpm)",         "rmse_bpm",         "%.2f"),
            ("% within ±5 bpm",    "pct_within_5bpm",  "%.1f"),
            ("% within ±3 bpm",    "pct_within_3bpm",  "%.1f"),
            ("Bland-Altman bias",  "ba_bias_bpm",      "%+.2f"),
            ("BA 95% LoA (lower)", "ba_loa_lower",     "%+.2f"),
            ("BA 95% LoA (upper)", "ba_loa_upper",     "%+.2f"),
            ("Pearson r",          "pearson_r",        "%.3f"),
            ("Algorithm-failure %","failure_rate_pct", "%.1f"),
        ]:
            f.write(f"| {label} | {cell(peak_m, key, fmt)} | {cell(fft_m, key, fmt)} |\n")
        f.write("\n## Design trade-offs\n\n")
        f.write("- **Peak detector (time-domain):** continuous HR estimate (no bin "
                "quantisation), simple, requires no FFT memory. Fails when peak "
                "detection fails (low SNR, motion).\n")
        f.write("- **FFT spectral (frequency-domain):** HR resolution = fs_ds/N × 60 = "
                f"{60.0 * 15.625 / 256:.2f} bpm/bin "
                "before parabolic interp (~0.4 bpm after). More robust to occasional "
                "missed peaks (energy is integrated across the cycle). Costs +2 kB flash "
                "(FFT code + twiddle + Hamming) and +1 kB SRAM (256-pt complex Q15 "
                "workspace).\n\n")
        f.write("The two methods are not directly substitutable: the peak detector "
                "gives instantaneous HR per beat (with the 1-sample resolution of the "
                "filtered signal), while the FFT averages over the whole window. The "
                "right choice depends on whether the application prefers latency "
                "(peak detector) or robustness (FFT).\n")
    print(f"wrote {md}")

    # Scatter
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    estim_peak = [(r["hr_ref_median"], r["hr_peak"]) for r in rows
                  if r.get("accepted", False) and r["hr_peak"] > 0]
    estim_fft  = [(r["hr_ref_median"], r["hr_fft"]) for r in rows
                  if r.get("accepted", False) and r["hr_fft"] > 0]
    if not estim_peak or not estim_fft:
        print("(skipping scatter — empty estimable set)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, data, title, m in [
        (axes[0], estim_peak, "Peak detector", peak_m),
        (axes[1], estim_fft,  "FFT spectral",  fft_m),
    ]:
        rr = np.array([p[0] for p in data])
        ee = np.array([p[1] for p in data])
        ax.scatter(rr, ee, alpha=0.5, s=18)
        lo = min(float(rr.min()), float(ee.min())) - 5
        hi = max(float(rr.max()), float(ee.max())) + 5
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="y = x")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel("BIDMC reference HR (bpm)")
        ax.set_ylabel("Embedded HR (bpm)")
        mae = m.get("mae_bpm", float("nan"))
        pct = m.get("pct_within_5bpm", float("nan"))
        ax.set_title(f"{title}\nMAE={mae:.2f} bpm | {pct:.1f}% within ±5 | n={m['n']}")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "method_comparison.png", dpi=120)
    plt.close()
    print(f"wrote {out_dir / 'method_comparison.png'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", default="bidmc01,bidmc02,bidmc03,bidmc04,bidmc05,"
                                          "bidmc06,bidmc07,bidmc08,bidmc09,bidmc10",
                    help="comma-separated record IDs (default: 10 records)")
    ap.add_argument("--window", type=float, default=30.0)
    ap.add_argument("--taps", type=int, default=101)
    ap.add_argument("--f1", type=float, default=0.7)
    ap.add_argument("--f2", type=float, default=3.5)
    ap.add_argument("--fft-n", type=int, default=256)
    ap.add_argument("--fft-decim", type=int, default=8)
    ap.add_argument("--cache-dir", default=str(ROOT / "data" / "bidmc_cache"))
    ap.add_argument("--limit-windows", type=int, default=None)
    ap.add_argument("--subharmonic-divisor", type=int, default=3,
                    help="sub-harmonic α = 1/DIVISOR; 3 = default (α≈0.33, "
                         "catches all bidmc47 windows; small bidmc35/40 "
                         "false-1/2× cost); 2 = looser (α=0.5)")
    args = ap.parse_args()

    records = args.records.split(",")
    cache_dir = Path(args.cache_dir)
    RESULTS.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for rec_id in records:
        print(f"[{rec_id}] processing...", flush=True)
        rows = process_record(rec_id, cache_dir, args.window, args.taps,
                              args.f1, args.f2, args.fft_n, args.fft_decim,
                              args.limit_windows,
                              subharmonic_divisor=args.subharmonic_divisor)
        n_acc = sum(1 for r in rows if r.get("accepted", False))
        print(f"[{rec_id}] {len(rows)} windows, {n_acc} accepted", flush=True)
        all_rows.extend(rows)

    csv_path = RESULTS / "method_comparison.csv"
    if all_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"wrote {csv_path}")

    peak_m = method_metrics(all_rows, "hr_peak")
    fft_m  = method_metrics(all_rows, "hr_fft")
    summary = {
        "peak": peak_m,
        "fft":  fft_m,
        "n_records": len(records),
        "n_windows_total": len(all_rows),
        "window_s": args.window,
        "fft_n": args.fft_n,
        "fft_decim": args.fft_decim,
    }
    (RESULTS / "method_comparison.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {RESULTS / 'method_comparison.json'}")
    emit_comparison_outputs(all_rows, peak_m, fft_m, RESULTS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
