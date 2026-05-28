"""Shared firmware build + QEMU invocation helpers for the sweep scripts.

The three callers (batch_validate.py, compare_methods.py, validate_rr.py)
share `arm_compile()` + `run_qemu_parse()` here so they don't drift on
flags or parse regex. New validators get the helpers for free.

Why per-window /tmp build: the NTFS layer on the development host has a
`folio_wait` D-state hang that wedges `ld` under sustained per-window
build pressure. Routing ELFs through `/tmp` (tmpfs) sidesteps the slow
underlying device.
"""
from __future__ import annotations
import re
import subprocess
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
FIRMWARE = ROOT / "firmware"

ARM_CFLAGS = [
    "-mcpu=cortex-m3", "-mthumb", "-O2", "-g3", "-ffreestanding",
    "-Wall", "-Wextra", "-I.", "-ffunction-sections", "-fdata-sections",
    "-T", "lm3s6965.ld", "-nostartfiles", "--specs=nosys.specs",
    "-Wl,--gc-sections",
]

# Per-firmware source lists. Kept here so the sweep callers
# (batch_validate.py, compare_methods.py, validate_rr.py) don't drift.
# New validators reuse these by import.
PEAK_SOURCES = ["startup.c", "main.c",     "dsp_fixed.c"]
FFT_SOURCES  = ["startup.c", "main_fft.c", "dsp_fixed.c", "dsp_fft.c"]
RR_SOURCES   = ["startup.c", "main_rr.c",  "dsp_fixed.c", "dsp_resp.c"]

QEMU_CMD_TEMPLATE = [
    "qemu-system-arm", "-M", "lm3s6965evb", "-cpu", "cortex-m3",
    "-nographic", "-semihosting-config", "enable=on,target=native",
    "-kernel",  # ELF path appended at call time
]


def arm_compile(sources: Iterable[str], out_elf: Path,
                timeout_s: float = 60.0) -> None:
    """Cross-compile `sources` (relative to firmware/) to `out_elf` with
    the project's standard CFLAGS. Raises subprocess.CalledProcessError on
    non-zero exit; subprocess.TimeoutExpired on timeout."""
    cmd = ["arm-none-eabi-gcc", *ARM_CFLAGS, "-o", str(out_elf), *sources]
    subprocess.run(cmd, cwd=str(FIRMWARE), check=True,
                   capture_output=True, timeout=timeout_s)


def run_qemu(elf: Path, timeout_s: float = 30.0) -> str:
    """Run QEMU on the given ELF; return combined stdout+stderr.

    Raises subprocess.TimeoutExpired on timeout; subprocess.CalledProcessError
    is not raised (QEMU's exit code with semihosting is not always 0 even on
    success — we don't gate on it).
    """
    cmd = [*QEMU_CMD_TEMPLATE, str(elf)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    return proc.stdout + proc.stderr


def run_qemu_parse(elf: Path, key: str,
                   timeout_s: float = 30.0, scale: float = 100.0) -> float:
    """Run QEMU and extract `<KEY>=<int>` from semihosting output, returning
    the value divided by `scale` (default 100 — matches the firmware's
    HR_X100 / RR_X100 convention).

    Returns 0.0 if the key isn't found — callers treat that as the firmware's
    "I can't estimate" sentinel.
    """
    out = run_qemu(elf, timeout_s=timeout_s)
    m = re.search(rf"{re.escape(key)}=(\d+)", out)
    return int(m.group(1)) / scale if m else 0.0
