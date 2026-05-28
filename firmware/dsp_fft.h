/*
 * dsp_fft.h — radix-2 decimation-in-time FFT in Q15 for Cortex-M3.
 *
 * Self-contained, freestanding, in-place. Each butterfly stage scales its outputs
 * by 1/2 to prevent Q15 overflow as magnitudes grow through log2(N) stages —
 * total scaling of the magnitude spectrum is 1/N relative to a float reference.
 *
 * Memory:
 *   - cplx_q15_t buffer of N samples (4 bytes each) → 1 KB at N=256.
 *   - Twiddle table of N/2 entries, expected in flash (caller-provided).
 *
 * Bit-exactness: matches numpy.fft.fft(x) / N to within ±1 LSB per bin given
 * Q15 truncation in the twiddle multiply (each butterfly does
 *   t = (W * b) >> 15
 * which is round-toward-zero — the worst-case integer truncation).
 */
#ifndef DSP_FFT_H
#define DSP_FFT_H

#include <stdint.h>

typedef struct { int16_t re, im; } cplx_q15_t;

/* In-place radix-2 DIT FFT. n_log2 = log2(N); N must be a power of 2.
 * twiddle[N/2] = {Re(W_N^k), Im(W_N^k)} for k=0..N/2-1, where W_N = exp(-j 2π/N).
 *
 * Caller must populate x[].re with the (windowed) real signal and x[].im with zeros
 * before calling. */
void fft_q15(cplx_q15_t *x, int n_log2, const cplx_q15_t *twiddle);

/* |X[k]|² as uint32_t — used for peak-bin search; avoids sqrt. */
static inline uint32_t cplx_mag2_q15(cplx_q15_t v) {
    return (uint32_t)((int32_t)v.re * v.re) + (uint32_t)((int32_t)v.im * v.im);
}

#endif /* DSP_FFT_H */
