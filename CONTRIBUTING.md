# Contributing to ppg-embedded

PRs are welcome. The priorities lean toward keeping every claim honest and
emulation-accurate, not feature breadth. Read this file before opening a PR.

## Project scope

`ppg-embedded` demonstrates three skills together — embedded systems, DSP
architecture, biomedical signal processing — in C on ARM Cortex-M3 under QEMU,
validated against the PhysioNet BIDMC corpus. There is no physical hardware.
Every claim in `README.md` / `results/README.md` must remain framed as
*"validated in emulation."*

## Data licensing

BIDMC records 04–53 are downloaded locally on demand and **never committed**
(the `.gitignore` allowlist enforces ship-only-CI-subset = bidmc01-03).
`data/bidmc_cache/NOTICE` carries the dataset citation and license; don't
remove it.

If you add another dataset (PPG-DaLiA, CapnoBase, MIMIC-III, …), follow the
same pattern: a `NOTICE` file with citation + license, and a `.gitignore`
allowlist that ships only the CI smoke-test subset.

## Style

- **C**: freestanding (no `printf` / `malloc`), Q15 fixed-point in DSP hot
  paths; float-emulation OK in post-spectral or per-window post-processing
  (≤ ~5–10 ops; documented in `main_fft.c` and `main_rr.c`). Static-analysis
  CI runs `cppcheck` + `clang-tidy` informationally.
- **Python**: bare-stdlib default for `src/reference.py` — numpy / scipy /
  wfdb are optional. Type hints encouraged; `ruff check src/` should pass.
- **Commits**: present-tense imperative; scope-tagged when natural
  (`feat(fft):`, `test:`, `chore(ci):`); single-purpose. The 1-2-sentence
  *why* matters more than the *what*.

## Running everything locally

```bash
# Reference + firmware build + QEMU smoke (peak path)
./run_all.sh

# Sub-second host canaries (no QEMU)
make -C firmware host-test host-fft-test test     # peak + FFT + Unity

# Coverage (gcov)
make -C firmware coverage                          # → results/coverage.md

# Python validators
uv run python src/verify_fft.py                    # FFT bit-exactness
uv run python src/qformat_proof.py                 # Q15 dynamic-range proof
uv run python src/batch_validate.py --records bidmc01,bidmc02,bidmc03 --window 30
uv run python src/compare_methods.py --records bidmc01,bidmc02,bidmc03 --window 30
uv run python src/validate_rr.py --records bidmc01,bidmc02,bidmc03 --window 30
uv run python src/window_sweep.py
uv run python src/footprint.py                     # arm-none-eabi-size

# Release validation — full 53-record sweep (~35 min, opt-in)
RECORDS=$(seq -f 'bidmc%02g' 1 53 | paste -sd,)
uv run python src/batch_validate.py --records $RECORDS --window 30
uv run python src/compare_methods.py --records $RECORDS --window 30
uv run python src/validate_rr.py --records $RECORDS --window 30
```

## What I'd love help with

- Cycle-accurate timing claims on Cortex-M4F (Renode or real hardware).
- TFLite-Micro signal-quality classifier replacing the variance-based SQI.
- Pan-Tompkins adaptive-threshold variant of the peak detector.
- Karlen-2013 AM/FM RR paths + smart fusion. The Python draft is at
  `src/_respiration_three_channel_draft.py`; the firmware port is open.
- PPG-DaLiA / CapnoBase loader work — the loader contract is
  `src/batch_validate.py::load_bidmc_record`; both datasets share its shape.
