#!/usr/bin/env python3
"""Run the canonical firmware/firmware.elf under QEMU and compare the embedded
(fixed-point) HR to the Python float golden reference and the synthetic ground
truth. The synthetic-input smoke step in run_all.sh — distinct from the
per-window BIDMC sweep in src/batch_validate.py."""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from _firmware_io import run_qemu  # noqa: E402

ROOT = SRC.parent
GEN = ROOT / "firmware" / "generated"
FIRMWARE_ELF = ROOT / "firmware" / "firmware.elf"
TOL_VS_FLOAT = 3.0   # bpm
TOL_VS_GT = 5.0      # bpm


def run_qemu_and_parse(timeout_s: float = 60.0) -> dict:
    """Run firmware/firmware.elf under QEMU; parse PEAKS, VALID_SAMPLES, HR_X100
    from the semihosted output. Raises RuntimeError if HR_X100 is missing."""
    out = run_qemu(FIRMWARE_ELF, timeout_s=timeout_s)
    m_hr = re.search(r"HR_X100=(\d+)", out)
    if not m_hr:
        raise RuntimeError(f"no HR_X100 in QEMU output:\n{out}")
    m_peaks = re.search(r"PEAKS=(\d+)", out)
    m_L = re.search(r"VALID_SAMPLES=(\d+)", out)
    return {
        "hr_bpm": int(m_hr.group(1)) / 100.0,
        "peaks": int(m_peaks.group(1)) if m_peaks else None,
        "valid_samples": int(m_L.group(1)) if m_L else None,
        "raw_output": out,
    }


def main() -> int:
    golden = json.loads((GEN / "golden.json").read_text())
    try:
        result = run_qemu_and_parse()
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: could not run QEMU ({e}). Is qemu-system-arm installed and firmware built?")
        return 2

    print("---- QEMU output ----")
    print(result["raw_output"].strip())
    print("---------------------")

    hr = result["hr_bpm"]
    ref = golden["ref_hr_float"]
    gt = golden["gt_hr"]

    print(f"embedded (fixed-point, Cortex-M3) HR = {hr:.2f} bpm")
    print(f"python   (float reference)        HR = {ref:.2f} bpm   |delta| = {abs(hr - ref):.2f}")
    ok = abs(hr - ref) <= TOL_VS_FLOAT
    if gt == gt:  # not NaN
        print(f"ground truth (synthetic)          HR = {gt:.2f} bpm   |delta| = {abs(hr - gt):.2f}")
        ok = ok and abs(hr - gt) <= TOL_VS_GT

    print("PASS" if ok else "FAIL (delta exceeds tolerance)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
