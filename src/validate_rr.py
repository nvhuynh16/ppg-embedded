"""Validate the embedded BW-path RR estimator against BIDMC's reference RR.

Per-window pipeline (mirrors src/batch_validate.py's HR sweep):
  1. Load the BIDMC PPG + reference RR (CSV column 3).
  2. Normalize the PPG and Q15-quantize.
  3. Write firmware/generated/ppg_data.h with the current window.
  4. Build firmware_rr.elf to /tmp/ (per the NTFS folio_wait workaround) and
     run under QEMU `lm3s6965evb`.
  5. Parse RR_X100= from semihosting output, divide by 100 → BrPM.
  6. Compare against median ref_rr in the same 30-s span.

Three-way decomposition (estimable / accepted / ungated) mirrors the HR
validator; bootstrap CIs from src/bootstrap.py are merged into the emitted
metrics. Outputs land in `results/respiration.{csv,json,png}`.

This validates the BW path only; AM/FM channels live in
src/_respiration_three_channel_draft.py and are not exercised here.
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
from reference import (q15, write_ppg_data_header, refractory_samples,  # noqa: E402
                       design_fir_bandpass)
from sqi import window_variance, variance_threshold  # noqa: E402
from batch_validate import load_bidmc_record, normalize_signal  # noqa: E402
from bootstrap import cluster_bootstrap_ci  # noqa: E402
from _firmware_io import arm_compile, run_qemu_parse, RR_SOURCES  # noqa: E402

ROOT = SRC.parent
RESULTS = ROOT / "results"
GEN = ROOT / "firmware" / "generated"
TMP_RR = Path("/tmp") / "ppg_rr.elf"


def process_record(record_id: str, cache_dir: Path, window_s: float, taps: int,
                   f1: float, f2: float,
                   limit_windows: int | None = None) -> list[dict]:
    sig, fs, ref_t, ref_hr, resp_sig, ref_rr = load_bidmc_record(
        record_id, cache_dir, include_rr=True)
    sig = normalize_signal(sig)

    h = design_fir_bandpass(taps, f1, f2, fs)
    h_q = [q15(v) for v in h]
    refractory = refractory_samples(fs)
    samples_per_window = int(round(window_s * fs))

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

        # SQI is variance of band-passed signal (matches batch_validate.py)
        from reference import fir_apply
        yf = fir_apply(x, h)
        var = window_variance(yf)
        variances.append(var)

        write_ppg_data_header(x_q, h_q, fs, len(x_q), refractory,
                              ref_hr_x100=0, path=GEN / "ppg_data.h")
        try:
            arm_compile(RR_SOURCES, TMP_RR)
        except subprocess.TimeoutExpired:
            print(f"  [{record_id} w{i}] build timeout — skip", file=sys.stderr)
            continue
        try:
            rr_emb = run_qemu_parse(TMP_RR, "RR_X100")
        except Exception as e:  # noqa: BLE001
            print(f"  [{record_id} w{i}] QEMU failed: {e}", file=sys.stderr)
            continue

        t_end = t_s + window_s
        mask = (ref_t >= t_s) & (ref_t < t_end)
        if not np.any(mask):
            rr_ref_med, rr_ref_std = float("nan"), float("nan")
        else:
            valid = ref_rr[mask]
            valid = valid[~np.isnan(valid)]
            valid = valid[valid > 0]
            if len(valid) == 0:
                rr_ref_med, rr_ref_std = float("nan"), float("nan")
            else:
                rr_ref_med = float(np.median(valid))
                rr_ref_std = float(np.std(valid))

        rows.append({
            "record": record_id, "t_start_s": round(t_s, 3),
            "rr_embedded": round(rr_emb, 3),
            "rr_ref_median": round(rr_ref_med, 3) if not np.isnan(rr_ref_med) else float("nan"),
            "rr_ref_std": round(rr_ref_std, 3) if not np.isnan(rr_ref_std) else float("nan"),
            "variance": round(var, 6),
        })
        if (i + 1) % 5 == 0:
            print(f"  [{record_id}] {i+1}/{n_windows} windows done", flush=True)

    if variances:
        threshold = variance_threshold(variances, pct=10.0)
        for r, v in zip(rows, variances):
            ref_med = r["rr_ref_median"]
            ref_std = r["rr_ref_std"]
            r["accepted"] = (
                v >= threshold
                and not (isinstance(ref_med, float) and np.isnan(ref_med))
                and not (isinstance(ref_std, float) and np.isnan(ref_std))
                and ref_std < 5.0     # ±5 BrPM ground-truth variability gate
            )
    return rows


def compute_rr_metrics(rows: list[dict], subset: str = "estimable") -> dict:
    if subset == "estimable":
        valid = [r for r in rows if r.get("accepted", False) and r["rr_embedded"] > 0]
    elif subset == "accepted":
        valid = [r for r in rows if r.get("accepted", False)]
    else:
        valid = [r for r in rows
                 if not (isinstance(r["rr_ref_median"], float) and np.isnan(r["rr_ref_median"]))]
    if not valid:
        return {"n": 0}
    emb = np.array([r["rr_embedded"] for r in valid])
    ref = np.array([r["rr_ref_median"] for r in valid])
    delta = emb - ref
    abs_d = np.abs(delta)
    out = {
        "n": len(valid),
        "mae_brpm": float(np.mean(abs_d)),
        "rmse_brpm": float(np.sqrt(np.mean(delta ** 2))),
        "pct_within_2brpm": float(100.0 * np.mean(abs_d <= 2.0)),
        "pct_within_4brpm": float(100.0 * np.mean(abs_d <= 4.0)),
        "ba_bias_brpm": float(np.mean(delta)),
        "ba_loa_lower": float(np.mean(delta) - 1.96 * np.std(delta)),
        "ba_loa_upper": float(np.mean(delta) + 1.96 * np.std(delta)),
    }
    if len(emb) > 1 and float(np.std(emb)) > 0 and float(np.std(ref)) > 0:
        out["pearson_r"] = float(np.corrcoef(emb, ref)[0, 1])
    return out


def emit_plots(rows: list[dict], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    estim = [r for r in rows if r.get("accepted", False) and r["rr_embedded"] > 0]
    if len(estim) < 2:
        print("(skipping RR plots — < 2 estimable windows)")
        return
    emb = np.array([r["rr_embedded"] for r in estim])
    ref = np.array([r["rr_ref_median"] for r in estim])

    mean_rr = (emb + ref) / 2
    delta = emb - ref
    bias = float(np.mean(delta))
    sd = float(np.std(delta))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(mean_rr, delta, alpha=0.5, s=18)
    ax.axhline(bias, color="k", ls="-", label=f"bias = {bias:+.2f}")
    ax.axhline(bias + 1.96 * sd, color="r", ls="--",
               label=f"+1.96 SD = {bias + 1.96*sd:+.2f}")
    ax.axhline(bias - 1.96 * sd, color="r", ls="--",
               label=f"−1.96 SD = {bias - 1.96*sd:+.2f}")
    ax.set_xlabel("Mean RR (BrPM)")
    ax.set_ylabel("Embedded − BIDMC reference (BrPM)")
    ax.set_title(f"Bland-Altman, RR (n={len(estim)})")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "respiration_ba.png", dpi=120)
    plt.close()
    print(f"wrote {out_dir / 'respiration_ba.png'}")

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(ref, emb, alpha=0.5, s=18)
    lo = min(float(ref.min()), float(emb.min())) - 2
    hi = max(float(ref.max()), float(emb.max())) + 2
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="y = x")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("BIDMC reference RR (BrPM)")
    ax.set_ylabel("Embedded RR (BrPM)")
    ax.set_title(f"Embedded vs reference RR (n={len(estim)})")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "respiration_scatter.png", dpi=120)
    plt.close()
    print(f"wrote {out_dir / 'respiration_scatter.png'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", default="bidmc01,bidmc02,bidmc03")
    ap.add_argument("--window", type=float, default=30.0)
    ap.add_argument("--taps", type=int, default=101)
    ap.add_argument("--f1", type=float, default=0.7)
    ap.add_argument("--f2", type=float, default=3.5)
    ap.add_argument("--cache-dir", default=str(ROOT / "data" / "bidmc_cache"))
    ap.add_argument("--limit-windows", type=int, default=None)
    args = ap.parse_args()

    records = args.records.split(",")
    cache_dir = Path(args.cache_dir)
    RESULTS.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for rec_id in records:
        print(f"[{rec_id}] processing...", flush=True)
        rows = process_record(rec_id, cache_dir, args.window, args.taps,
                              args.f1, args.f2, args.limit_windows)
        n_acc = sum(1 for r in rows if r.get("accepted", False))
        print(f"[{rec_id}] {len(rows)} windows, {n_acc} accepted", flush=True)
        all_rows.extend(rows)

    csv_path = RESULTS / "respiration.csv"
    if all_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"wrote {csv_path}")

    estimable = compute_rr_metrics(all_rows, subset="estimable")
    # Bootstrap CIs on the same subset
    estimable.update(cluster_bootstrap_ci(
        all_rows, n_iter=10_000, seed=0,
        emb_col="rr_embedded", ref_col="rr_ref_median", record_col="record"))
    metrics = {
        "estimable": estimable,
        "accepted":  compute_rr_metrics(all_rows, subset="accepted"),
        "ungated":   compute_rr_metrics(all_rows, subset="ungated"),
        "n_records": len(records),
        "n_windows_total": len(all_rows),
        "window_s": args.window,
        "bootstrap_iter": 10_000,
        "bootstrap_seed": 0,
        "rr_path": "BW only (AM/FM/smart-fusion are draft, not in firmware)",
    }
    (RESULTS / "respiration.json").write_text(json.dumps(metrics, indent=2))
    print(f"wrote {RESULTS / 'respiration.json'}")

    e = estimable
    if e.get("n", 0) > 0:
        print(f"  estimable : MAE={e['mae_brpm']:.2f} BrPM  "
              f"±2={e['pct_within_2brpm']:.1f}%  ±4={e['pct_within_4brpm']:.1f}%  "
              f"n={e['n']}")

    emit_plots(all_rows, RESULTS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
