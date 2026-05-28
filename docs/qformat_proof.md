# Q15 dynamic-range proof

Closed-form worst-case bounds for every fixed-point variable in the
ppg-embedded pipeline, validated empirically by `src/qformat_proof.py`. The
goal is to demonstrate that no intermediate variable can overflow its
declared type under any input, so the firmware is **provably overflow-free
at the current hyperparameters** (FIR 101 taps, 0.7–3.5 Hz, fs=125 Hz; FFT
N=256, per-stage `>>1` scaling).

## Pipeline + Q-formats

```
   PPG int16          FIR_acc int64       FIR_out int32 (Q15)
  ┌──────────┐       ┌────────────┐      ┌────────────────┐
  │ x ∈ Q15  │──fir──│ Σ h·x      │──>>15│ y ∈ Q15        │──sat16──┐
  │ |x|<2¹⁵  │       │ acc ∈ int64│      │ |y| ≤ G·|x_max|│         │
  └──────────┘       └────────────┘      └────────────────┘         │
                                                                      ▼
                  bin re/im int16 (Q15)    cplx Q15·Q15 → int32      ┌────────┐
                  ┌──────────────────┐    ┌──────────────────┐       │ × win  │
                  │ |X[k]| ≤ max|x|  │←FFT│ s∈Q15 (decimated)│←──────│ Q15·Q15│
                  │ per-stage >>1    │    │                  │       │ >>15   │
                  └──────────────────┘    └──────────────────┘       └────────┘
                            │
                            ▼
                       mag² uint32
                  ┌──────────────────┐    ┌──────────────────┐    ┌──────────┐
                  │ re²+im² ≤ 2·2³⁰  │    │ parabolic interp │    │ HR_x100  │
                  │ ≈ 2.15·10⁹       │───→│ float (libgcc)   │───→│ uint32   │
                  └──────────────────┘    └──────────────────┘    └──────────┘
```

## Stage-by-stage bounds

All bounds derived for the current hyperparameters. The script
`src/qformat_proof.py` re-derives them for any other `(taps, f1, f2, fs)`
choice — this document just lists the numbers for the current design.

### Stage 1 — Input PPG (`ppg_q15[]`, int16)

Q15 representation, `|x| ≤ 32767 ≈ 2¹⁵`. No proof needed; the loader
`reference.py::q15(v)` clamps to ±32767 before emission.

### Stage 2 — FIR accumulator (`acc`, int64, `dsp_fixed.c:6`)

The accumulator sums `ntaps` products of `int16 × int16`:

```
|acc| ≤ Σ_k |h_q[k]| · max_x = Σ|h_q| · 32767
```

For the current band-pass FIR (`design_fir_bandpass(101, 0.7, 3.5, 125)`):

| Quantity         | Value          |
|------------------|---------------:|
| `Σ|h_q[k]|`      | **41 802 Q15** |
| `Σ|h_q| · 32767` | **1.37 × 10⁹** |
| int64 max        | **9.22 × 10¹⁸** |
| Headroom         | **~6.7 × 10⁹×** |

✅ The int64 accumulator has overwhelming headroom. Headroom collapses only
for `ntaps > ~10¹⁴`, far beyond any conceivable filter length.

### Stage 3 — FIR output (`filt[]`, int32, `dsp_fixed.c:10`)

`filt[i] = acc >> 15`, so:

```
|filt[i]| ≤ (Σ|h_q| · max_x) / 2¹⁵ = Σ|h_q| · max_x / 32768
         ≤ 41 802 · 32767 / 32768
         ≈ 41 800   (≈ 1.28 · 2¹⁵)
```

So in the **adversarial worst case** (an impossible all-coefficient-signs-
matched input), `|filt[i]|` can exceed int16 range by ~28 %. This is the only
non-trivial bound in the pipeline.

For **physically realisable inputs** the worst case is a full-scale sinusoid
at the FIR's passband peak (~2.0 Hz, gain 0.997 per `qformat_proof.py`):
`|filt[i]| ≤ 0.997 · 32767 ≈ 32 669`. Comfortably within int16. ✓

The `sat16` saturator at `main_fft.c:52` is the safety net that absorbs the
adversarial-input case before passing to the FFT path — by construction, the
peak-detector path operates directly on `filt` as int32 and is immune.

### Stage 4 — Window multiply (`main_fft.c:54`)

`(int32_t)(s * w) >> 15` where `s ∈ int16` (post-sat16) and `w ∈ Q15` with
`|w| ≤ 32767`. Result range:

```
|s · w| ≤ 32767 · 32767 ≈ 1.07 × 10⁹     (fits int32)
|(s · w) >> 15| ≤ 32767                  (fits int16)
```

✅ No overflow. The Hamming window's actual peak is 32767 (`q15(1.0)`).

### Stage 5 — FFT bins (`fft_buf[].re/.im`, int16, `dsp_fft.c`)

The FFT is a 256-pt radix-2 DIT with **per-stage `>>1` scaling** across 8
butterfly stages. For complex input `x[n]`:

```
X[k] = Σ_{n=0}^{N-1} x[n] · W_N^{kn}
|X[k]| ≤ Σ_n |x[n]| ≤ N · max|x|     (raw, pre-scaling)
```

Total per-stage `>>1` scaling factor = `2^log₂(N)` = `N`. So:

```
|X[k]|_scaled ≤ max|x|     (per re or im component)
```

✅ FFT bins are bounded by the input range — int16 is sufficient.

The `verify_fft.py` bit-exactness contract gives an empirical check: per-bin
|re|/|im| error vs `numpy.fft.fft` ≤ `log₂(N) + 1 = 9 LSB` worst case
(measured: 5 LSB on the 72 bpm test sinusoid).

### Stage 6 — Magnitude² (`cplx_mag2_q15`, uint32, `dsp_fft.h`)

```
|X[k]|² = re² + im²  ≤ 2 · 32767² ≈ 2.15 × 10⁹
uint32 max           = 4.29 × 10⁹
```

✅ Fits uint32 with 2× headroom.

### Stage 7 — Sub-harmonic check (`main_fft.c:75`)

Integer compare `sub_mag2 >= max_mag2 / 2`. Both ≤ 2.15 × 10⁹; their division
by 2 is bounded by the same number. No risk. ✓

### Stage 8 — Parabolic interpolation (`main_fft.c:86-96`)

Five-op float computation; libgcc soft-float on Cortex-M3. Out of scope for
fixed-point analysis. The HR output `hr_x100` is then bounded by:

```
hr_x100 = round(f_peak_hz · 6000)
f_peak_hz ≤ FFT_K_MAX · fs_ds / N  ≤ 57 · 15.625 / 256  ≈ 3.479 Hz
hr_x100 ≤ 3.479 · 6000  ≈ 20 874
```

✅ Fits uint32 (with vast headroom).

## Worst-case summary

| Stage            | Variable     | Type   | Worst-case  | Type max     | Status |
|------------------|--------------|--------|------------:|-------------:|:------:|
| 1. Input         | `ppg_q15[i]` | int16  | 32 767      | 32 767       | ✅      |
| 2. FIR acc       | `acc`        | int64  | 1.37 × 10⁹  | 9.22 × 10¹⁸  | ✅      |
| 3. FIR out (adv) | `filt[i]`    | int32  | 41 800      | 2.15 × 10⁹   | ✅      |
| 3. FIR out (real)| `filt[i]`    | int32  | 32 669      | 2.15 × 10⁹   | ✅      |
| 4. Window mult   | `fft_buf.re` | int16  | 32 767      | 32 767       | ✅      |
| 5. FFT bin       | `fft_buf.re` | int16  | 32 767      | 32 767       | ✅      |
| 6. mag²          | `mag2`       | uint32 | 2.15 × 10⁹  | 4.29 × 10⁹   | ✅      |
| 7. sub-harm cmp  | `mag2/2`     | uint32 | 1.07 × 10⁹  | 4.29 × 10⁹   | ✅      |
| 8. HR output     | `hr_x100`    | uint32 | 20 874      | 4.29 × 10⁹   | ✅      |

**Conclusion:** every fixed-point variable is provably bounded by its
declared type for any input the loader can produce. The only stage where the
adversarial (sign-aligned, impossible-to-realise) bound exceeds the natural
type range is the post-FIR `filt[]` value — handled by the `sat16` step on
the FFT path; on the peak-detector path `filt` is int32 (room to spare).

## Empirical validation

`src/qformat_proof.py` exercises this analysis with three input families:

1. **Adversarial** — sign-aligned full-scale impulses; checks the worst-case
   bound `Σ|h_q|`. Expected outcome: `|filt|` reaches ≈ 41 800 (int16 overflow,
   absorbed by `sat16`).
2. **Realistic full-scale sinusoids** — at every frequency from 0.1 to fs/2
   in 0.1 Hz steps; checks the gain envelope. Expected outcome: `|filt|`
   peaks at ≈ 32 669 (within int16).
3. **Random PPG-like (Gaussian noise filtered by a 1 Hz lowpass + DC trend)**
   — 10⁵ samples; checks the typical empirical envelope. Expected outcome:
   well below the realistic-worst bound.

The script asserts each measured intermediate stays within its analytical
bound and prints a summary table. Run with `uv run python src/qformat_proof.py`.
