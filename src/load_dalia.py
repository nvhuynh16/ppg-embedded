"""PPG-DaLiA loader — wrist BVP + ECG-derived HR labels.

Returns the same 4-tuple contract as `src/batch_validate.py::load_bidmc_record`
so the sweep infrastructure plugs in without per-dataset branching.

PPG-DaLiA conventions (Reiss et al. 2019):
  - wrist BVP @ 64 Hz (signal.wrist.BVP, shape (N, 1))
  - chest ECG @ 700 Hz (signal.chest.ECG) — used to derive HR labels
  - HR labels (key 'label'): one HR estimate per 8-s window, sliding by 2 s
    (so labels are at 0.5 Hz). Conventional time-stamping puts label[i] at
    the centre of its 8-s window: t = 2·i + 4 s.

Pickled with Python 2 — must load with `encoding="latin1"`.
"""
from __future__ import annotations
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PKL_DIR = ROOT / "data" / "dalia_cache" / "PPG_FieldStudy"      # full pickles (gitignored)
EXTRACT_DIR = ROOT / "data" / "dalia_cache" / "extracts"        # compact .npz (S1, S6 committed)
BVP_FS = 64.0           # wrist BVP sample rate (Hz)
ACTIVITY_FS = 4.0       # activity-label sample rate (Hz)
LABEL_PERIOD_S = 2.0    # HR labels every 2 s
LABEL_WINDOW_S = 8.0    # each label averages over an 8-s window

# Activity-label code → name (Reiss et al. 2019, Table 1).
ACTIVITY_NAMES = {
    0: "transient", 1: "sit", 2: "stairs", 3: "table-soccer",
    4: "cycling", 5: "driving", 6: "lunch", 7: "walking", 8: "working",
}


def load_dalia_record(subject_id: str | int,
                      pkl_dir: Path = PKL_DIR,
                      extract_dir: Path = EXTRACT_DIR,
                      want_activity: bool = False,
                      ) -> tuple[np.ndarray, float, np.ndarray, np.ndarray] | tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    """Load one PPG-DaLiA subject. Prefers the compact .npz extract.

    `subject_id` can be "S1".."S15" or an int 1..15.

    Returns (pleth_signal, fs, ref_time_s, ref_hr_bpm). Set `want_activity=True`
    to also receive the activity label array (4-Hz grid, 0..8 codes — see
    ACTIVITY_NAMES) as a 5th return.
    """
    if isinstance(subject_id, int):
        subject_id = f"S{subject_id}"

    npz = Path(extract_dir) / f"{subject_id}.npz"
    if npz.exists():
        d = np.load(npz)
        bvp = d["bvp"].astype(np.float64)
        hr_labels = d["hr"].astype(np.float64)
        activity = d["activity"].astype(np.uint8) if want_activity else None
    else:
        pkl = Path(pkl_dir) / subject_id / f"{subject_id}.pkl"
        if not pkl.exists():
            raise FileNotFoundError(
                f"neither {npz} nor {pkl} found — run "
                f"scripts/fetch_dalia.py + scripts/make_dalia_extracts.py")
        with open(pkl, "rb") as f:
            d = pickle.load(f, encoding="latin1")
        bvp = np.asarray(d["signal"]["wrist"]["BVP"], dtype=np.float64).reshape(-1)
        hr_labels = np.asarray(d["label"], dtype=np.float64).reshape(-1)
        activity = (np.asarray(d["activity"], dtype=np.uint8).reshape(-1)
                    if want_activity else None)

    # Convention: label[i] is HR over window [2i, 2i+8] s; centre time = 2i + 4
    ref_time_s = np.arange(len(hr_labels), dtype=np.float64) * LABEL_PERIOD_S + (LABEL_WINDOW_S / 2.0)
    if want_activity:
        return bvp, BVP_FS, ref_time_s, hr_labels, activity
    return bvp, BVP_FS, ref_time_s, hr_labels


def list_cache(pkl_dir: Path = PKL_DIR, extract_dir: Path = EXTRACT_DIR) -> list[str]:
    """Return sorted subject IDs available via extract or full pickle."""
    found: set[str] = set()
    if extract_dir.exists():
        found.update(p.stem for p in extract_dir.glob("S*.npz"))
    if pkl_dir.exists():
        found.update(p.name for p in pkl_dir.iterdir()
                     if p.is_dir() and p.name.startswith("S")
                     and (p / f"{p.name}.pkl").exists())
    return sorted(found, key=lambda s: int(s[1:]))
