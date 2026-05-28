#!/usr/bin/env bash
# End-to-end: generate reference -> build firmware -> run on QEMU -> validate.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "==> 1/3  Python golden reference (generates firmware/generated/ppg_data.h)"
python3 "$ROOT/src/reference.py"

echo "==> 2/3  Build firmware (arm-none-eabi-gcc)"
make -C "$ROOT/firmware"

echo "==> 3/3  Run on QEMU Cortex-M3 + validate against reference"
python3 "$ROOT/src/validate.py"
