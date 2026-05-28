# Firmware footprint (Cortex-M3 / QEMU lm3s6965evb)

Toolchain: `arm-none-eabi-gcc (15:13.2.rel1-2) 13.2.1 20231009`. Binary: `firmware/firmware.elf`.

## Section sizes

| Section | Bytes | Lives in |
|---|---:|---|
| `.text` (code) | 4,540 | flash |
| `.data` (initialised RW) | 0 | flash + SRAM |
| `.bss` (zero-init RW) | 7,048 | SRAM |
| **Total ELF (dec)** | **11,588** | |

Target: LM3S6965 — FLASH 256 KB, SRAM 64 KB. **Flash use: 4,540 bytes (1.7%)**, **SRAM use: 7,048 bytes (10.8%)**.

## Top 10 symbols by size

| Symbol | Bytes |
|---|---:|
| `filt` | 5,000 |
| `ppg_q15` | 2,500 |
| `intervals.0` | 1,024 |
| `peak_idx` | 1,024 |
| `__udivmoddi4` | 692 |
| `hr_x100_from_peaks` | 212 |
| `fir_q15_coef` | 202 |
| `main` | 188 |
| `detect_peaks` | 124 |
| `fir_q15` | 98 |

## Honesty note

Cycle counts and timing claims are **omitted deliberately**: QEMU `lm3s6965evb` is a functional emulator, not cycle-accurate. The footprint numbers above are exact (from `arm-none-eabi-size` on the binary). For cycle-accurate measurement, port to Renode (Cortex-M4F) or real hardware with a DWT cycle counter — see Future Work in the top-level README.
