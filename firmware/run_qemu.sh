#!/usr/bin/env bash
# Run the firmware on an emulated Cortex-M3 (no hardware). Output via semihosting.
# Optional arg: ELF filename (default firmware.elf). Use firmware_fft.elf for the
# spectral HR path.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
ELF="${1:-firmware.elf}"

exec timeout 30 qemu-system-arm \
  -M lm3s6965evb \
  -cpu cortex-m3 \
  -nographic \
  -semihosting-config enable=on,target=native \
  -kernel "$DIR/$ELF"
