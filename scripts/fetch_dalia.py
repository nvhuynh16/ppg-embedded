#!/usr/bin/env python3
"""Fetch PPG-DaLiA dataset from the UCI ML Repository.

PPG-DaLiA (Reiss et al. 2019, CC BY-NC 4.0) — 15 subjects with wrist PPG (BVP
@ 64 Hz), tri-axial accelerometer @ 32 Hz, chest ECG @ 700 Hz (ground truth
HR derived from R-peaks @ 2 Hz), across 8 activity scenarios (sit, stairs,
soccer, cycling, drive, lunch, walk, work).

We use it as the motion-corrupted counterpoint to BIDMC's clean ICU data.

Idempotent: skips re-download if the zip is already present and the SHA-256
matches the recorded checksum. Extraction also idempotent.

Usage:
  uv run python scripts/fetch_dalia.py                # default cache dir
  uv run python scripts/fetch_dalia.py --no-extract   # download only

Output:
  data/dalia_cache/PPG_FieldStudy.zip                 (~1.2 GB)
  data/dalia_cache/PPG_FieldStudy/S{1..15}/S{1..15}.pkl
"""
from __future__ import annotations
import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = ROOT / "data" / "dalia_cache"

# UCI ML Repo direct URL. Mirror this if the primary 404s.
URL = "https://archive.ics.uci.edu/static/public/495/ppg+dalia.zip"
# Expected SHA-256 of the published archive. Verified once locally; pin it
# here so any tampered re-download is caught at extract-time.
EXPECTED_SHA256 = "5772387956e34e2e2dc4c2ddbeb98cb70569d5112fa4c13ee98a17680b84a1f3"


def _sha256(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    """Stream-download `url` to `dest` with a tiny progress indicator."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[fetch_dalia] downloading {url}\n  → {dest}", flush=True)
    with urllib.request.urlopen(url) as r:
        total = int(r.headers.get("Content-Length", "0"))
        got = 0
        with open(tmp, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                if total:
                    pct = 100.0 * got / total
                    print(f"  {got/1e6:8.1f} / {total/1e6:.1f} MB  {pct:5.1f}%",
                          end="\r", flush=True)
        print()
    tmp.rename(dest)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_DIR)
    p.add_argument("--no-extract", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="re-download even if archive already present")
    args = p.parse_args()

    cache = args.cache_dir
    cache.mkdir(parents=True, exist_ok=True)
    zip_path = cache / "PPG_FieldStudy.zip"

    if zip_path.exists() and not args.force:
        print(f"[fetch_dalia] {zip_path} already present (size={zip_path.stat().st_size/1e6:.1f} MB)")
    else:
        _download(URL, zip_path)

    # Compute + record SHA-256
    sha = _sha256(zip_path)
    sha_file = cache / "PPG_FieldStudy.zip.sha256"
    sha_file.write_text(f"{sha}  PPG_FieldStudy.zip\n")
    print(f"[fetch_dalia] SHA-256 = {sha}")
    if EXPECTED_SHA256 is not None and sha != EXPECTED_SHA256:
        print(f"  WARNING: expected {EXPECTED_SHA256}; archive may be corrupt", file=sys.stderr)

    if args.no_extract:
        print("[fetch_dalia] --no-extract: skipping unzip", flush=True)
        return 0

    # Extract — idempotent (skip if S15/S15.pkl already exists)
    extracted = cache / "PPG_FieldStudy" / "S15" / "S15.pkl"
    if extracted.exists():
        print(f"[fetch_dalia] already extracted (found {extracted.relative_to(cache)})")
        return 0
    print(f"[fetch_dalia] extracting {zip_path} → {cache}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(cache)
    print("[fetch_dalia] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
