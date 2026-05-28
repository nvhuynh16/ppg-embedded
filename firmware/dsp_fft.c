/*
 * dsp_fft.c — radix-2 DIT FFT in Q15. See dsp_fft.h.
 *
 * Algorithm: standard Cooley-Tukey decimation-in-time. log2(N) stages; each stage
 * processes N/2 butterflies. Within a stage of "size" m, butterflies span pairs
 * (k, k + m/2) for k = j, j+m, j+2m, ...
 *
 * Q15 multiplication semantics: (a * b) >> 15 — implements Q15 × Q15 → Q15 with
 * truncation. Per-stage >>1 scaling keeps the running magnitude bounded by the
 * input magnitude (max growth factor per stage = 2, mitigated by the divide by 2).
 */
#include "dsp_fft.h"

/* In-place bit-reversal permutation. x has 2^n_log2 entries.
 * Iterative O(N) algorithm: for each i, compute its bit-reversed index j and swap
 * if i < j (each pair is swapped exactly once). */
static void bit_reverse(cplx_q15_t *x, int n_log2) {
    int n = 1 << n_log2;
    int j = 0;
    for (int i = 1; i < n; i++) {
        int bit = n >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) {
            cplx_q15_t t = x[i];
            x[i] = x[j];
            x[j] = t;
        }
    }
}

void fft_q15(cplx_q15_t *x, int n_log2, const cplx_q15_t *twiddle) {
    int n = 1 << n_log2;
    bit_reverse(x, n_log2);

    /* Stage s = 1..n_log2; butterfly span m = 2^s; half-span h = 2^(s-1).
     * The twiddle index step at stage s is N / m = N >> s = 2^(n_log2 - s). */
    for (int s = 1; s <= n_log2; s++) {
        int m = 1 << s;
        int h = m >> 1;
        int t_step = n >> s;  /* index step into twiddle[] for this stage */

        for (int k = 0; k < n; k += m) {
            for (int j = 0; j < h; j++) {
                cplx_q15_t w = twiddle[j * t_step];
                cplx_q15_t a = x[k + j];
                cplx_q15_t b = x[k + j + h];

                /* t = W * b, Q15.Q15 → Q15 via (>> 15). */
                int32_t t_re = ((int32_t)w.re * b.re - (int32_t)w.im * b.im) >> 15;
                int32_t t_im = ((int32_t)w.re * b.im + (int32_t)w.im * b.re) >> 15;

                /* Butterfly: (a, b) → ((a + t)/2, (a - t)/2). The >>1 prevents
                 * overflow as the spectrum magnitude grows across stages. */
                x[k + j].re     = (int16_t)((a.re + t_re) >> 1);
                x[k + j].im     = (int16_t)((a.im + t_im) >> 1);
                x[k + j + h].re = (int16_t)((a.re - t_re) >> 1);
                x[k + j + h].im = (int16_t)((a.im - t_im) >> 1);
            }
        }
    }
}
