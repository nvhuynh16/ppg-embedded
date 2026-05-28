/* Fixed-point (Q15) DSP primitives for the embedded PPG heart-rate estimator.
 * These mirror ARM CMSIS-DSP semantics (e.g. arm_fir_q15) but are self-contained
 * so the demo builds with only arm-none-eabi-gcc. See README for the CMSIS-DSP upgrade.
 *
 * Python <-> C invariance audit. Any change to either implementation must
 * preserve this contract — Python and C peak detectors must agree on the
 * detected peak indices, and the HR formulas (median inter-peak interval,
 * integer arithmetic, [0.4 s, 1.5 s] gate) must agree bit-for-bit.
 *   refractory window    : SHARED via REFRACTORY_SAMPLES emitted from src/reference.py
 *   HR formula           : peaks*60*fs/L; Python float, C int64*100; agree to 0.01 bpm
 *   threshold = max|y|/2 : Python on float yf, C on Q15-quantized filt[]; absorbed by tolerance
 *   FIR transient skip   : i in [ntaps-1, len(x)); identical bounds
 *   peak-loop bounds     : i in [1, len-1); identical
 *   local-max compare    : y[i] >= y[i-1] && y[i] > y[i+1]; identical asymmetric tie-break
 *   coefficient quant    : computed in Python, C reads quantized values from ppg_data.h
 */
#ifndef DSP_FIXED_H
#define DSP_FIXED_H
#include <stdint.h>

/* Q15 FIR filter. x[n], h[k] are Q15 (int16). Output y is int32 (acc >> 15).
 * Writes (n - ntaps + 1) samples; returns that count. */
int fir_q15(const int16_t *x, int n, const int16_t *h, int ntaps, int32_t *y);

/* Find peaks in y[len]: local maxima above (max|y|/2), separated by >= refractory samples.
 * Stores the first up-to max_peaks indices into peak_indices[]; returns total peak count
 * (which may exceed max_peaks; caller should size the buffer to the worst-case HR x window). */
int detect_peaks(const int32_t *y, int len, int refractory,
                 int *peak_indices, int max_peaks);

/* HR (× 100) from median inter-peak interval. Returns 0 if fewer than 2 valid intervals.
 * Intervals outside [0.4 s, 1.5 s] (HR ∉ [40, 150] bpm) are rejected as outliers — this
 * is what makes the estimator robust to occasional missed/spurious peaks. */
uint32_t hr_x100_from_peaks(const int *peak_indices, int n_peaks, int fs);

#endif /* DSP_FIXED_H */
