"""BIDMC batch validator — the credibility-phase centerpiece.

For each (record, window): regenerate ppg_data.h with the window's signal,
rebuild firmware, run QEMU, parse HR_X100, align to the median BIDMC reference
HR within the window's time span, append a row to results/bidmc.csv.

At the end: compute MAE, RMSE, Pearson r, % within ±5 bpm, Bland-Altman bias
+ 95% LoA. Emit results/{bidmc.csv, metrics.json, bland_altman.png, hr_scatter.png}.

Imports project modules — does not duplicate DSP logic.

Usage:
    python src/batch_validate.py --records bidmc01,bidmc02,bidmc03 --window 8 --limit-windows 5
    python src/batch_validate.py --records $(seq -f 'bidmc%02g' 1 53 | paste -sd,) --window 8
"""
from __future__ import annotations
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import wfdb

# project modules (relative to src/)
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from reference import (
    design_fir_bandpass, fir_apply,
    q15, refractory_samples, write_ppg_data_header,
)
from sqi import window_variance, variance_threshold

from _firmware_io import arm_compile, run_qemu_parse, PEAK_SOURCES  # noqa: E402

ROOT = SRC.parent
RESULTS = ROOT / "results"
GEN = ROOT / "firmware" / "generated"

# Per-window build artifact on tmpfs (/tmp) — writing ELFs to /media on this
# machine triggers a `folio_wait` D-state hang in `ld` under sustained per-
# window builds. Routing to /tmp sidesteps it.
TMP_ELF = Path("/tmp") / "ppg_sweep_firmware.elf"


def load_bidmc_record(record_id: str, cache_dir: Path, include_rr: bool = False):
    """Return record signals + reference HR for a cached BIDMC record.

    Default (include_rr=False):  (pleth_signal, fs, ref_time_s, ref_hr_bpm)
    With include_rr=True:        (pleth_signal, fs, ref_time_s, ref_hr_bpm,
                                  resp_signal, ref_rr_brpm)

    `resp_signal` is BIDMC's impedance-pneumography RESP channel (WFDB index
    0 on every record; verified against bidmc01-53.hea). `ref_rr_brpm` is
    column 3 of the Numerics CSV (same time axis as ref_hr at 1 Hz).
    The 6-tuple variant is used by src/validate_rr.py.
    """
    rec = wfdb.rdrecord(str(cache_dir / record_id))
    names = [s.lower() for s in rec.sig_name]
    pleth_idx = next((i for i, nm in enumerate(names) if "pleth" in nm or "ppg" in nm), 1)
    fs = float(rec.fs)
    sig = np.asarray(rec.p_signal[:, pleth_idx], dtype=float)

    # Numerics CSV: bidmc_csv/bidmc_NN_Numerics.csv  (cols: Time [s], HR, PULSE, RESP, SpO2)
    n = int(record_id.replace("bidmc", ""))
    csv_path = cache_dir / "bidmc_csv" / f"bidmc_{n:02d}_Numerics.csv"
    arr = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    ref_t = arr[:, 0]
    ref_hr = arr[:, 1]

    if not include_rr:
        return sig, fs, ref_t, ref_hr

    # RESP channel: WFDB index 0 on BIDMC; impedance pneumography in /pm.
    resp_idx = next((i for i, nm in enumerate(names) if nm == "resp"), 0)
    resp_sig = np.asarray(rec.p_signal[:, resp_idx], dtype=float)
    # Reference RR is Numerics CSV column 3 (breaths/min, 1 Hz).
    ref_rr = arr[:, 3]
    return sig, fs, ref_t, ref_hr, resp_sig, ref_rr


def normalize_signal(sig: np.ndarray) -> np.ndarray:
    """Mean-subtract + scale to peak |0.9| (matches reference.py load_record behavior)."""
    sig = sig[~np.isnan(sig)]
    sig = sig - sig.mean()
    m = float(np.max(np.abs(sig))) or 1.0
    return 0.9 * sig / m


def process_record(record_id: str, cache_dir: Path, window_s: float,
                   taps: int, f1: float, f2: float,
                   limit_windows: int | None = None,
                   verbose: bool = True) -> list[dict]:
    """Per-window pipeline for one record. Returns list of result dicts."""
    sig, fs, ref_t, ref_hr = load_bidmc_record(record_id, cache_dir)
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

        # Float-domain band-passed window for the variance gate
        yf = fir_apply(x, h)
        var = window_variance(yf)
        variances.append(var)

        # Embedded pipeline via QEMU: build peak firmware to /tmp, run, parse.
        write_ppg_data_header(x_q, h_q, fs, len(x_q), refractory,
                              ref_hr_x100=0, path=GEN / "ppg_data.h")
        try:
            arm_compile(PEAK_SOURCES, TMP_ELF)
            hr_embedded = run_qemu_parse(TMP_ELF, "HR_X100")
        except subprocess.TimeoutExpired:
            print(f"  [{record_id} w{i}] build/QEMU timeout — skip", file=sys.stderr)
            continue
        except Exception as e:  # noqa: BLE001
            print(f"  [{record_id} w{i}] build/QEMU failed: {e}", file=sys.stderr)
            continue

        # Align to BIDMC reference HR over the window's time span
        t_end = t_s + window_s
        mask = (ref_t >= t_s) & (ref_t < t_end)
        if not np.any(mask):
            hr_ref_med, hr_ref_std = float("nan"), float("nan")
        else:
            valid = ref_hr[mask]
            valid = valid[~np.isnan(valid)]
            valid = valid[valid > 0]  # BIDMC encodes "no data" as 0
            if len(valid) == 0:
                hr_ref_med, hr_ref_std = float("nan"), float("nan")
            else:
                hr_ref_med = float(np.median(valid))
                hr_ref_std = float(np.std(valid))

        rows.append({
            "record": record_id,
            "t_start_s": round(t_s, 3),
            "hr_embedded": round(hr_embedded, 3),
            "hr_ref_median": round(hr_ref_med, 3) if not np.isnan(hr_ref_med) else float("nan"),
            "hr_ref_std": round(hr_ref_std, 3) if not np.isnan(hr_ref_std) else float("nan"),
            "variance": round(var, 6),
        })

        if verbose and (i + 1) % 10 == 0:
            print(f"  [{record_id}] {i+1}/{n_windows} windows done", flush=True)

    # SQI gate: reject bottom 10% by variance within this record
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


def compute_metrics(rows: list[dict], subset: str = "estimable") -> dict:
    """Aggregate metrics over the rows.

    Subsets:
      - "estimable" : accepted by SQI AND hr_embedded > 0 (the honest performance number).
      - "accepted"  : SQI-accepted (includes algorithm-failure sentinels — pulls MAE down).
      - "ungated"   : all rows with a valid reference HR (no SQI gate at all).

    hr_embedded == 0 is the C estimator's "I cannot estimate HR" sentinel (every
    inter-peak interval fell outside the [0.4 s, 1.5 s] physiological gate). Treating
    that as a literal HR of 0 bpm in MAE would conflate algorithm-failure with
    estimation-error — report the failure rate separately instead.
    """
    if subset == "estimable":
        valid = [r for r in rows if r.get("accepted", False) and r["hr_embedded"] > 0]
    elif subset == "accepted":
        valid = [r for r in rows if r.get("accepted", False)]
    else:  # "ungated"
        valid = [
            r for r in rows
            if not (isinstance(r["hr_ref_median"], float) and np.isnan(r["hr_ref_median"]))
        ]
    if not valid:
        return {"n": 0}
    emb = np.array([r["hr_embedded"] for r in valid])
    ref = np.array([r["hr_ref_median"] for r in valid])
    delta = emb - ref
    abs_d = np.abs(delta)
    out: dict = {
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
    return out


def failure_stats(rows: list[dict]) -> dict:
    """Algorithm-failure-rate diagnostics: how often hr_x100_from_peaks returns 0."""
    acc = [r for r in rows if r.get("accepted", False)]
    failed = [r for r in acc if r["hr_embedded"] == 0]
    if not failed:
        return {"n_accepted": len(acc), "n_failed": 0, "failure_rate_pct": 0.0}
    ref_hrs = sorted(r["hr_ref_median"] for r in failed
                     if not (isinstance(r["hr_ref_median"], float) and np.isnan(r["hr_ref_median"])))
    return {
        "n_accepted": len(acc),
        "n_failed": len(failed),
        "failure_rate_pct": 100.0 * len(failed) / len(acc),
        "failed_ref_hr_min": float(min(ref_hrs)) if ref_hrs else None,
        "failed_ref_hr_max": float(max(ref_hrs)) if ref_hrs else None,
        "failed_ref_hr_median": float(ref_hrs[len(ref_hrs) // 2]) if ref_hrs else None,
    }


def emit_plots(rows: list[dict], out_dir: Path) -> None:
    """Bland-Altman + scatter plots. SQI-accepted windows are split into estimable
    (hr_embedded > 0) and algorithm-failure (hr_embedded == 0); bias/LoA are computed
    from estimable only, and failure points are overlaid in red so the failure mode
    is visually obvious without polluting the agreement statistics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    acc = [r for r in rows if r.get("accepted", False)]
    estim = [r for r in acc if r["hr_embedded"] > 0]
    failed = [r for r in acc if r["hr_embedded"] == 0]
    if len(estim) < 2:
        print("(skipping plots — fewer than 2 estimable windows)")
        return
    emb = np.array([r["hr_embedded"] for r in estim])
    ref = np.array([r["hr_ref_median"] for r in estim])
    emb_f = np.array([r["hr_embedded"] for r in failed])
    ref_f = np.array([r["hr_ref_median"] for r in failed])

    # Bland-Altman — bias/LoA from estimable only
    mean_hr = (emb + ref) / 2
    delta = emb - ref
    bias = float(np.mean(delta))
    sd = float(np.std(delta))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(mean_hr, delta, alpha=0.5, s=18, label=f"estimable (n={len(estim)})")
    if len(failed) > 0:
        ax.scatter((emb_f + ref_f) / 2, emb_f - ref_f, alpha=0.7, s=22,
                   color="red", marker="x",
                   label=f"algorithm failure, hr_embedded=0 (n={len(failed)})")
    ax.axhline(bias, color="red", linestyle="-", label=f"bias = {bias:+.2f} bpm")
    ax.axhline(bias + 1.96 * sd, color="red", linestyle="--",
               label=f"+1.96 SD = {bias + 1.96*sd:+.2f}")
    ax.axhline(bias - 1.96 * sd, color="red", linestyle="--",
               label=f"-1.96 SD = {bias - 1.96*sd:+.2f}")
    ax.set_xlabel("Mean of embedded + BIDMC reference HR (bpm)")
    ax.set_ylabel("Embedded HR − BIDMC reference HR (bpm)")
    ax.set_title(f"Bland-Altman: embedded Q15 vs BIDMC ECG-derived HR")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "bland_altman.png", dpi=120)
    plt.close()

    # Scatter
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(ref, emb, alpha=0.5, s=18, label=f"estimable (n={len(estim)})")
    if len(failed) > 0:
        ax.scatter(ref_f, emb_f, alpha=0.7, s=22, color="red", marker="x",
                   label=f"algorithm failure (n={len(failed)})")
    all_emb = np.concatenate([emb, emb_f]) if len(failed) > 0 else emb
    all_ref = np.concatenate([ref, ref_f]) if len(failed) > 0 else ref
    lo = min(float(all_emb.min()), float(all_ref.min())) - 5
    hi = max(float(all_emb.max()), float(all_ref.max())) + 5
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="y = x")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("BIDMC ECG-derived HR (bpm)")
    ax.set_ylabel("Embedded Q15 HR (bpm)")
    ax.set_title(f"HR agreement: embedded vs BIDMC")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "hr_scatter.png", dpi=120)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", default="bidmc01,bidmc02,bidmc03",
                    help="comma-separated record IDs (default: 3 cached records)")
    ap.add_argument("--window", type=float, default=8.0,
                    help="window length in seconds (default 8)")
    ap.add_argument("--taps", type=int, default=101)
    ap.add_argument("--f1", type=float, default=0.7)
    ap.add_argument("--f2", type=float, default=3.5)
    ap.add_argument("--cache-dir", default=str(ROOT / "data" / "bidmc_cache"))
    ap.add_argument("--limit-windows", type=int, default=None,
                    help="cap per-record window count (for smoke testing)")
    ap.add_argument("--metrics-only", action="store_true",
                    help="skip QEMU sweep; recompute metrics.json + plots from existing bidmc.csv")
    args = ap.parse_args()

    records = args.records.split(",")
    cache_dir = Path(args.cache_dir)
    RESULTS.mkdir(parents=True, exist_ok=True)

    if args.metrics_only:
        csv_path = RESULTS / "bidmc.csv"
        if not csv_path.exists():
            print(f"--metrics-only requires existing {csv_path}", file=sys.stderr)
            return 2
        all_rows = []
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                for k in ("hr_embedded", "hr_ref_median", "hr_ref_std", "variance"):
                    try: r[k] = float(r[k])
                    except (TypeError, ValueError): r[k] = float("nan")
                r["accepted"] = (r.get("accepted", "").strip() == "True")
                all_rows.append(r)
        records = sorted({r["record"] for r in all_rows})
        print(f"loaded {len(all_rows)} rows from {csv_path} (no sweep)")
    else:
        all_rows: list[dict] = []
        for rec_id in records:
            print(f"[{rec_id}] processing...", flush=True)
            rows = process_record(rec_id, cache_dir, args.window,
                                  args.taps, args.f1, args.f2,
                                  args.limit_windows)
            n_acc = sum(1 for r in rows if r.get("accepted", False))
            print(f"[{rec_id}] {len(rows)} windows, {n_acc} accepted", flush=True)
            all_rows.extend(rows)

    # CSV (only write when sweep ran; --metrics-only leaves existing CSV alone)
    csv_path = RESULTS / "bidmc.csv"
    if not args.metrics_only and all_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"wrote {csv_path}")

    # Metrics — three-way decomposition (see compute_metrics docstring) +
    # patient-cluster bootstrap CIs (10k iter, seeded). The bootstrap is
    # computed only on the estimable subset (the headline number); the
    # accepted/ungated subsets serve as decomposition context, not headline.
    from bootstrap import cluster_bootstrap_ci  # local import — keeps the default
    # path of batch_validate.py runnable on a clean numpy/stdlib environment.

    estimable_metrics = compute_metrics(all_rows, subset="estimable")
    estimable_ci = cluster_bootstrap_ci(all_rows, n_iter=10_000, seed=0)
    estimable_metrics.update(estimable_ci)

    metrics = {
        "estimable": estimable_metrics,
        "accepted":  compute_metrics(all_rows, subset="accepted"),
        "ungated":   compute_metrics(all_rows, subset="ungated"),
        "failures":  failure_stats(all_rows),
        "n_records": len(records),
        "n_windows_total": len(all_rows),
        "window_s": args.window,
        "fir_taps": args.taps,
        "bootstrap_iter": 10_000,
        "bootstrap_seed": 0,
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"wrote {RESULTS / 'metrics.json'}")

    def _fmt(m: dict, key: str, fmt: str) -> str:
        return (fmt % m[key]) if key in m else "n/a"
    e = metrics["estimable"]; a = metrics["accepted"]; f = metrics["failures"]
    print(f"  estimable : MAE={_fmt(e, 'mae_bpm', '%.2f')} bpm  "
          f"pct_within_5bpm={_fmt(e, 'pct_within_5bpm', '%.1f')}%  "
          f"n={e.get('n', 0)}  (excludes algorithm-failure sentinels)")
    print(f"  accepted  : MAE={_fmt(a, 'mae_bpm', '%.2f')} bpm  "
          f"pct_within_5bpm={_fmt(a, 'pct_within_5bpm', '%.1f')}%  "
          f"n={a.get('n', 0)}")
    print(f"  failures  : {f.get('n_failed', 0)} / {f.get('n_accepted', 0)} "
          f"({f.get('failure_rate_pct', 0):.1f}%) "
          f"— hr_embedded=0 sentinel from C estimator")

    emit_plots(all_rows, RESULTS)
    print("wrote bland_altman.png and hr_scatter.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
