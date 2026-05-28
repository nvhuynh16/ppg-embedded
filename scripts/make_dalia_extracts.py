#!/usr/bin/env python3
"""Strip PPG-DaLiA pickles down to BVP + HR labels + activity.

Each full PPG-DaLiA subject pickle is ~1.5 GB (chest ECG @ 700 Hz, EMG, EDA,
respiration, accelerometer, etc.). For our heart-rate work we only need:
  - wrist BVP @ 64 Hz
  - ECG-derived HR labels @ 0.5 Hz
  - activity labels @ 4 Hz (lets us split clean/motion windows for SQI)

The compact .npz extract (~2.4 MB per subject) is what gets committed to the
repo so a clean clone + CI can demonstrate the pipeline. Full pickles remain
local-only (gitignored), fetched via scripts/fetch_dalia.py.

Usage:
  uv run python scripts/make_dalia_extracts.py                 # all subjects
  uv run python scripts/make_dalia_extracts.py --subjects S1,S6
"""
from __future__ import annotations
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PKL_DIR = ROOT / "data" / "dalia_cache" / "PPG_FieldStudy"
OUT_DIR = ROOT / "data" / "dalia_cache" / "extracts"


def extract_subject(subject_id: str, pkl_dir: Path = PKL_DIR,
                    out_dir: Path = OUT_DIR) -> Path:
    pkl = pkl_dir / subject_id / f"{subject_id}.pkl"
    if not pkl.exists():
        raise FileNotFoundError(f"{pkl} not found — run scripts/fetch_dalia.py first")
    with open(pkl, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    bvp = np.asarray(d["signal"]["wrist"]["BVP"], dtype=np.float32).reshape(-1)
    hr = np.asarray(d["label"], dtype=np.float32).reshape(-1)
    activity = np.asarray(d["activity"], dtype=np.uint8).reshape(-1)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{subject_id}.npz"
    np.savez_compressed(out, bvp=bvp, hr=hr, activity=activity)
    print(f"  {subject_id}: bvp={bvp.nbytes/1e6:.2f} MB  hr={len(hr)}  act={len(activity)}  "
          f"-> {out.relative_to(ROOT)} ({out.stat().st_size/1e6:.2f} MB)")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subjects", default=None,
                   help="comma list e.g. S1,S6 (default: all S1..S15)")
    p.add_argument("--pkl-dir", type=Path, default=PKL_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = p.parse_args()

    if args.subjects:
        subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    else:
        subjects = sorted(p.name for p in args.pkl_dir.iterdir()
                          if p.is_dir() and p.name.startswith("S")
                          and (p / f"{p.name}.pkl").exists())
    if not subjects:
        print("no subjects found", file=sys.stderr)
        return 1

    print(f"[make_dalia_extracts] {len(subjects)} subjects")
    for s in subjects:
        try:
            extract_subject(s, args.pkl_dir, args.out_dir)
        except FileNotFoundError as e:
            print(f"  skip {s}: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
