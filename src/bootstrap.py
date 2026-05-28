"""Patient-level cluster bootstrap for the BIDMC validation metrics.

Why cluster-bootstrap, not window-bootstrap: windows from the same patient
share recording artefacts, sensor coupling, and physiological state — they are
not statistically independent. A naive window-level resample undercovers the
true variance because resampling the dependence structure within a patient
doesn't change the autocorrelated noise. Cluster bootstrap resamples
**patient IDs** with replacement and pools all windows from each sampled
patient; this preserves within-patient correlation while estimating between-
patient variability — the relevant variance when generalizing to a new
patient.

10 000 iterations; seeded for reproducibility. Returns 2.5 / 97.5 percentiles
on every metric in compute_metrics().

Usage:
  from bootstrap import cluster_bootstrap_ci
  ci = cluster_bootstrap_ci(rows, n_iter=10_000)
  # → {"mae_bpm_ci95": [lo, hi], "rmse_bpm_ci95": [lo, hi], ...}

Then merge `ci` into the metrics dict emitted to metrics.json.
"""
from __future__ import annotations
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))


def _metric_point(emb: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    """Compute the same metrics as batch_validate.compute_metrics on a slice."""
    if len(emb) == 0:
        return {}
    delta = emb - ref
    abs_d = np.abs(delta)
    out = {
        "mae_bpm": float(np.mean(abs_d)),
        "rmse_bpm": float(np.sqrt(np.mean(delta ** 2))),
        "pct_within_5bpm": float(100.0 * np.mean(abs_d <= 5.0)),
        "pct_within_3bpm": float(100.0 * np.mean(abs_d <= 3.0)),
        "ba_bias_bpm": float(np.mean(delta)),
        "ba_loa_lower": float(np.mean(delta) - 1.96 * np.std(delta)),
        "ba_loa_upper": float(np.mean(delta) + 1.96 * np.std(delta)),
    }
    if len(emb) > 1 and float(np.std(emb)) > 0 and float(np.std(ref)) > 0:
        out["pearson_r"] = float(np.corrcoef(emb, ref)[0, 1])
    return out


def cluster_bootstrap_ci(
    rows: list[dict],
    n_iter: int = 10_000,
    seed: int = 0,
    subset: str = "estimable",
    emb_col: str = "hr_embedded",
    ref_col: str = "hr_ref_median",
    record_col: str = "record",
) -> dict[str, list[float]]:
    """Patient-cluster bootstrap 95 % CI for compute_metrics' output.

    Returns a dict keyed by `<metric>_ci95` with `[low, high]` values.
    Pass-through args allow reuse for the cross-corpus validators
    (validate_dalia / validate_capnobase / validate_rr) and the FFT-vs-peak
    comparison (where emb_col is `hr_peak` or `hr_fft`).
    """
    if subset == "estimable":
        valid = [r for r in rows if r.get("accepted", False) and r[emb_col] > 0]
    elif subset == "accepted":
        valid = [r for r in rows if r.get("accepted", False)]
    else:
        valid = [r for r in rows
                 if not (isinstance(r[ref_col], float) and np.isnan(r[ref_col]))]
    if len(valid) < 2:
        return {}

    # Group rows by patient (cluster ID).
    by_patient: dict[str, list[dict]] = defaultdict(list)
    for r in valid:
        by_patient[r[record_col]].append(r)
    patient_ids = list(by_patient.keys())
    n_patients = len(patient_ids)
    if n_patients < 2:
        return {}

    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = defaultdict(list)

    for _ in range(n_iter):
        # Resample patients with replacement; pool all of each sampled
        # patient's windows.
        picked = rng.choice(n_patients, size=n_patients, replace=True)
        emb_pool: list[float] = []
        ref_pool: list[float] = []
        for idx in picked:
            for r in by_patient[patient_ids[idx]]:
                emb_pool.append(float(r[emb_col]))
                ref_pool.append(float(r[ref_col]))
        if len(emb_pool) < 2:
            continue
        m = _metric_point(np.asarray(emb_pool), np.asarray(ref_pool))
        for k, v in m.items():
            samples[k].append(v)

    out: dict[str, list[float]] = {}
    for k, vs in samples.items():
        arr = np.asarray(vs)
        lo = float(np.percentile(arr, 2.5))
        hi = float(np.percentile(arr, 97.5))
        out[f"{k}_ci95"] = [lo, hi]
    return out


def fmt_ci(point: float, ci: list[float] | None, fmt: str = "%.2f") -> str:
    """Format `point [lo, hi]` for inclusion in markdown tables."""
    if ci is None:
        return fmt % point
    return f"{fmt % point} [{fmt % ci[0]}, {fmt % ci[1]}]"


def _self_test() -> int:
    """Quick smoke test: synthetic rows, sanity-check CI fields exist."""
    rng = np.random.default_rng(42)
    rows = []
    for rec in [f"bidmc{i:02d}" for i in range(1, 11)]:
        for w in range(8):
            ref = float(rng.uniform(60, 100))
            emb = ref + float(rng.normal(0, 2.0))
            rows.append({
                "record": rec,
                "hr_embedded": emb,
                "hr_ref_median": ref,
                "accepted": True,
            })
    ci = cluster_bootstrap_ci(rows, n_iter=1000, seed=0)
    expected_keys = {"mae_bpm_ci95", "rmse_bpm_ci95", "pct_within_5bpm_ci95",
                     "pct_within_3bpm_ci95", "ba_bias_bpm_ci95",
                     "ba_loa_lower_ci95", "ba_loa_upper_ci95", "pearson_r_ci95"}
    missing = expected_keys - set(ci.keys())
    if missing:
        print(f"FAIL: missing CI keys: {missing}")
        return 1
    for k, (lo, hi) in ci.items():
        if lo > hi:
            print(f"FAIL: {k} has lo > hi: {lo} > {hi}")
            return 1
    print("=== bootstrap self-test PASS ===")
    for k in sorted(ci):
        lo, hi = ci[k]
        print(f"  {k:35s} = [{lo:7.3f}, {hi:7.3f}]")
    return 0


if __name__ == "__main__":
    sys.exit(_self_test())
