"""TROIKA loader (Zhang 2014 treadmill PPG dataset).

Reference:
  Zhang, Z., Pi, Z., and Liu, B. (2014). "TROIKA: A general framework for
  heart rate monitoring using wrist-type photoplethysmographic signals during
  intensive physical exercise." IEEE Trans. Biomed. Eng., 62(2), 522-531.

Returns the same 4-tuple contract as `src/batch_validate.py::load_bidmc_record`
so the sweep infrastructure plugs in without per-dataset branching.

TROIKA conventions:
  - 12 subjects running on treadmill (~5 min each, intense motion)
  - Two PPG channels (1000 nm, 525 nm) + tri-axial accelerometer + ECG ground
    truth — all at 125 Hz
  - Files DATA_NN_TYPENN.mat. TYPE01 = baseline (walking + running);
    TYPE02 = harder cases (intense arm motion mid-recording)
  - Variables inside each .mat: `sig` (channels × samples), `BPM0` (ground-
    truth HR labels @ 1 Hz). `sig` rows: 0=ECG, 1=PPG ch1, 2=PPG ch2,
    3..5=ACC x/y/z. We use sig[2] (PPG ch2, 525 nm) — the convention used in
    most reproductions of the paper.

Status:
  Scaffolded. The TROIKA archive download isn't currently wired up
  (scripts/fetch_troika.py is a stub — the original Zhang 2014 hosting moved
  and the mirror landscape is fragile). Drop the .mat files into
  data/troika_cache/ manually and this loader will work.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = ROOT / "data" / "troika_cache"
PPG_FS = 125.0          # PPG / ECG / ACC sample rate (Hz)
HR_LABEL_FS = 1.0       # ground-truth BPM0 labels at 1 Hz
PPG_CHANNEL_ROW = 2     # sig[2] = PPG ch2 (525 nm) — standard reproduction choice


def load_troika_record(record_id: str,
                       cache_dir: Path = DEFAULT_CACHE
                       ) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Load one TROIKA recording.

    `record_id`: "DATA_01_TYPE01" .. "DATA_12_TYPE02" (i.e., filename without
    extension).

    Returns (pleth_signal, fs, ref_time_s, ref_hr_bpm) matching the BIDMC and
    PPG-DaLiA loader contracts.
    """
    try:
        from scipy.io import loadmat
    except ImportError as e:
        raise ImportError("TROIKA loader needs scipy (already in pyproject deps)") from e

    mat_path = Path(cache_dir) / f"{record_id}.mat"
    if not mat_path.exists():
        raise FileNotFoundError(
            f"{mat_path} not found. TROIKA archive must be staged manually — "
            f"see scripts/fetch_troika.py for the current mirror landscape "
            f"(the original Zhang 2014 hosting moved and we don't ship the "
            f"download script wired up yet).")
    d = loadmat(mat_path)
    if "sig" not in d:
        raise ValueError(f"{mat_path} missing 'sig' key (got {list(d.keys())})")
    sig = np.asarray(d["sig"], dtype=np.float64)        # shape (6, N)
    if sig.ndim != 2 or sig.shape[0] < 3:
        raise ValueError(f"{mat_path}: unexpected sig shape {sig.shape}")
    pleth = sig[PPG_CHANNEL_ROW, :]                      # PPG ch2 (525 nm)

    # Ground-truth labels. Some mat files name them BPM0; others use BPM_0.
    bpm = None
    for k in ("BPM0", "BPM_0", "BPMTrace", "BPM"):
        if k in d:
            bpm = np.asarray(d[k], dtype=np.float64).reshape(-1)
            break
    if bpm is None:
        raise ValueError(f"{mat_path}: no BPM0/BPM ground-truth key (got {list(d.keys())})")

    # Per the TROIKA convention, BPM0[i] = HR at t = i (1 Hz grid). Centre of
    # each estimation window is offset by some seconds in the original paper;
    # we keep the canonical 1-Hz indexing here.
    ref_time_s = np.arange(len(bpm), dtype=np.float64)
    return pleth, PPG_FS, ref_time_s, bpm


def list_cache(cache_dir: Path = DEFAULT_CACHE) -> list[str]:
    """Return sorted list of TROIKA record IDs present in cache_dir."""
    if not cache_dir.exists():
        return []
    return sorted(p.stem for p in cache_dir.glob("DATA_*.mat"))
