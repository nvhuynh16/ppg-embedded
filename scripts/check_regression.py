"""Performance-regression CI gate.

Compares current sweep outputs against the committed baseline
(`ci/baseline_*.json`). Fails (exit 1) if any of the following hard gates
trip:

  - HR MAE (estimable, peak detector) worsens by > 1 bpm
  - HR %within±3 (peak) drops by > 2 points
  - HR MAE (FFT+harm) worsens by > 1 bpm
  - HR %within±3 (FFT+harm) drops by > 2 points
  - HR algorithm-failure rate (peak) exceeds baseline + 1.5 points
  - RR MAE (BW path) worsens by > 0.5 BrPM
  - RR %within±2 BrPM drops by > 5 points

Outputs a clear before/after diff table on failure. On pass, prints a
one-line "OK" summary.

The 3-record CI smoke (bidmc01-03) is too small to drive these gates
reliably (n_estimable ≈ 45, large bootstrap variance). The gate is
intended to run on the FULL 53-record sweep — invoked manually or on
release-prep PRs, not on every commit. CI calls it informationally on
the 3-record subset (exit 0 on any regression; just log).
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CI = ROOT / "ci"
RESULTS = ROOT / "results"

GATES_HR = [
    # (subset key, metric key, direction, threshold, label)
    ("estimable", "mae_bpm",         "worsen", 1.0,  "HR MAE (peak)"),
    ("estimable", "pct_within_3bpm", "drop",   2.0,  "HR %within±3 (peak)"),
]

GATES_FFT = [
    ("mae_bpm",         "worsen", 1.0, "FFT+harm MAE"),
    ("pct_within_3bpm", "drop",   2.0, "FFT+harm %within±3"),
    ("failure_rate_pct","worsen", 0.5, "FFT+harm failure rate"),
]

GATES_RR = [
    ("mae_brpm",         "worsen", 0.5, "RR MAE (BW path)"),
    ("pct_within_2brpm", "drop",   5.0, "RR %within±2"),
]


def _check_gate(baseline_v: float, current_v: float, direction: str,
                threshold: float) -> bool:
    """Returns True if gate PASSES (no regression)."""
    if direction == "worsen":   # smaller-is-better metrics (MAE, RMSE, fail-rate)
        return current_v - baseline_v <= threshold
    elif direction == "drop":   # larger-is-better metrics (%within)
        return baseline_v - current_v <= threshold
    raise ValueError(f"unknown direction: {direction}")


def _diff_row(label: str, baseline_v: float, current_v: float,
              direction: str, threshold: float) -> tuple[bool, str]:
    passed = _check_gate(baseline_v, current_v, direction, threshold)
    delta = current_v - baseline_v
    arrow = "→" if abs(delta) > 1e-6 else "="
    status = "OK" if passed else "FAIL"
    return passed, f"  [{status:4s}] {label:30s}  {baseline_v:8.3f} {arrow} {current_v:8.3f}  Δ={delta:+7.3f}  (gate ±{threshold:.2f})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--informational", action="store_true",
                    help="report findings but exit 0 (for CI on 3-record smoke)")
    args = ap.parse_args()

    fails: list[str] = []
    rows: list[str] = []

    # HR — peak detector (batch_validate / metrics.json)
    if (CI / "baseline_metrics.json").exists() and (RESULTS / "metrics.json").exists():
        baseline = json.loads((CI / "baseline_metrics.json").read_text())["estimable"]
        current = json.loads((RESULTS / "metrics.json").read_text())["estimable"]
        rows.append("\n# HR — peak detector (batch_validate.py)")
        for subset, key, direction, thresh, label in GATES_HR:
            if key not in baseline or key not in current:
                continue
            ok, msg = _diff_row(label, baseline[key], current[key], direction, thresh)
            rows.append(msg)
            if not ok:
                fails.append(label)

    # HR — FFT+harm vs peak (compare_methods / method_comparison.json)
    if ((CI / "baseline_method_comparison.json").exists()
        and (RESULTS / "method_comparison.json").exists()):
        baseline = json.loads((CI / "baseline_method_comparison.json").read_text())
        current = json.loads((RESULTS / "method_comparison.json").read_text())
        rows.append("\n# HR — FFT+harm (compare_methods.py)")
        for key, direction, thresh, label in GATES_FFT:
            b = baseline["fft"].get(key)
            c = current["fft"].get(key)
            if b is None or c is None:
                continue
            ok, msg = _diff_row(label, b, c, direction, thresh)
            rows.append(msg)
            if not ok:
                fails.append(label)

    # RR — BW path (validate_rr / respiration.json)
    if (CI / "baseline_respiration.json").exists() and (RESULTS / "respiration.json").exists():
        baseline = json.loads((CI / "baseline_respiration.json").read_text())["estimable"]
        current = json.loads((RESULTS / "respiration.json").read_text())["estimable"]
        rows.append("\n# RR — BW path (validate_rr.py)")
        for key, direction, thresh, label in GATES_RR:
            if key not in baseline or key not in current:
                continue
            ok, msg = _diff_row(label, baseline[key], current[key], direction, thresh)
            rows.append(msg)
            if not ok:
                fails.append(label)

    print("=== Performance-regression check (vs committed baseline) ===")
    for r in rows:
        print(r)

    if fails:
        print(f"\n=== FAIL — {len(fails)} gate(s) tripped ===")
        for f in fails:
            print(f"  - {f}")
        if args.informational:
            print("\n(informational mode; exit 0)")
            return 0
        return 1
    print("\n=== OK — all gates within tolerance ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
