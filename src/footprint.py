"""Embedded footprint report: parses `arm-none-eabi-size` on firmware.elf.

Emits results/footprint.md with .text/.data/.bss byte sizes and the top symbols.
Does NOT include cycle counts — QEMU lm3s6965evb is functional-only, not cycle-
accurate; emitting cycle numbers would violate the project's honesty guardrail.
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIRMWARE_ELF = ROOT / "firmware" / "firmware.elf"

FLASH_BYTES = 256 * 1024   # LM3S6965 flash
SRAM_BYTES = 64 * 1024     # LM3S6965 SRAM


def parse_size(elf_path: Path) -> dict:
    out = subprocess.check_output(["arm-none-eabi-size", str(elf_path)], text=True)
    lines = out.strip().splitlines()
    if len(lines) < 2:
        raise RuntimeError(f"unexpected arm-none-eabi-size output:\n{out}")
    cols = lines[1].split()  # text  data  bss  dec  hex  filename
    return {"text": int(cols[0]), "data": int(cols[1]), "bss": int(cols[2]),
            "total_dec": int(cols[3])}


def top_symbols(elf_path: Path, top_n: int = 10) -> list[tuple[str, int]]:
    """Top N largest symbols by size from arm-none-eabi-nm --size-sort --print-size."""
    try:
        out = subprocess.check_output(
            ["arm-none-eabi-nm", "--size-sort", "--print-size", str(elf_path)],
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    rows: list[tuple[str, int]] = []
    for line in out.strip().splitlines():
        parts = line.split()
        if len(parts) >= 4:
            try:
                size = int(parts[1], 16)
                name = parts[-1]
                if size > 0:
                    rows.append((name, size))
            except ValueError:
                continue
    rows.sort(key=lambda r: -r[1])
    return rows[:top_n]


def gcc_version() -> str:
    try:
        return subprocess.check_output(["arm-none-eabi-gcc", "--version"],
                                        text=True).splitlines()[0]
    except Exception:
        return "unknown"


def main() -> int:
    if not FIRMWARE_ELF.exists():
        print(f"FAIL: {FIRMWARE_ELF} not found. Run `make -C firmware` first.",
              file=sys.stderr)
        return 1
    sz = parse_size(FIRMWARE_ELF)
    syms = top_symbols(FIRMWARE_ELF, top_n=10)

    flash_use = sz["text"] + sz["data"]
    sram_use = sz["data"] + sz["bss"]

    RESULTS.mkdir(parents=True, exist_ok=True)
    md = RESULTS / "footprint.md"
    with open(md, "w") as f:
        f.write("# Firmware footprint (Cortex-M3 / QEMU lm3s6965evb)\n\n")
        f.write(f"Toolchain: `{gcc_version()}`. Binary: `firmware/firmware.elf`.\n\n")

        f.write("## Section sizes\n\n")
        f.write("| Section | Bytes | Lives in |\n|---|---:|---|\n")
        f.write(f"| `.text` (code) | {sz['text']:,} | flash |\n")
        f.write(f"| `.data` (initialised RW) | {sz['data']:,} | flash + SRAM |\n")
        f.write(f"| `.bss` (zero-init RW) | {sz['bss']:,} | SRAM |\n")
        f.write(f"| **Total ELF (dec)** | **{sz['total_dec']:,}** | |\n\n")
        f.write(f"Target: LM3S6965 — FLASH {FLASH_BYTES // 1024} KB, SRAM {SRAM_BYTES // 1024} KB. ")
        f.write(f"**Flash use: {flash_use:,} bytes ({100*flash_use/FLASH_BYTES:.1f}%)**, ")
        f.write(f"**SRAM use: {sram_use:,} bytes ({100*sram_use/SRAM_BYTES:.1f}%)**.\n\n")

        f.write("## Top 10 symbols by size\n\n")
        if syms:
            f.write("| Symbol | Bytes |\n|---|---:|\n")
            for name, sze in syms:
                f.write(f"| `{name}` | {sze:,} |\n")
        else:
            f.write("(arm-none-eabi-nm unavailable or no symbols)\n")
        f.write("\n## Honesty note\n\n")
        f.write("Cycle counts and timing claims are **omitted deliberately**: QEMU "
                "`lm3s6965evb` is a functional emulator, not cycle-accurate. The "
                "footprint numbers above are exact (from `arm-none-eabi-size` on "
                "the binary). For cycle-accurate measurement, port to Renode "
                "(Cortex-M4F) or real hardware with a DWT cycle counter — "
                "see Future Work in the top-level README.\n")
    print(f"wrote {md}")
    print(f"  .text={sz['text']:,}  .data={sz['data']:,}  .bss={sz['bss']:,}  "
          f"flash_use={flash_use:,} ({100*flash_use/FLASH_BYTES:.1f}%)  "
          f"sram_use={sram_use:,} ({100*sram_use/SRAM_BYTES:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
