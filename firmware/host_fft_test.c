/*
 * host_fft_test.c — host-side FFT bit-exactness canary.
 *
 * Runs the same windowed+FFT pipeline as main_fft.c (without the band-pass FIR —
 * the test starts from a known synthetic input directly). Prints the resulting
 * magnitude² spectrum + the parabolically-interpolated peak HR. The Python harness
 * (src/verify_fft.py) regenerates the same signal in numpy, runs numpy.fft.fft,
 * and checks each bin agrees with the C result within ±1 LSB (the worst case for
 * Q15 truncation across log2(N) butterfly stages).
 *
 * Builds with the system gcc — no cross-compile, no QEMU. The point is to validate
 * the FFT *algorithm* in isolation; the QEMU run validates the full firmware.
 */
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#include "dsp_fft.h"
#include "generated/fft_data.h"

static cplx_q15_t fft_buf[FFT_N];

/* Generate a Q15 test sinusoid at frequency f_hz (sampled at FFT_FS_DS_X1000 / 1000 Hz).
 * Returns the integer bin nearest to f_hz under the FFT's sampling. */
static int gen_test_signal(int16_t *out, int n, double f_hz, double amp_q15) {
    double fs_ds = (double)FFT_FS_DS_X1000 / 1000.0;
    for (int i = 0; i < n; i++) {
        double v = amp_q15 * sin(2.0 * M_PI * f_hz * (double)i / fs_ds);
        if (v > 32767.0)  v = 32767.0;
        if (v < -32768.0) v = -32768.0;
        out[i] = (int16_t)v;
    }
    return (int)(f_hz * (double)n / fs_ds + 0.5);
}

int main(void) {
    /* Test signal: a 1.2 Hz (72 bpm) sinusoid at moderate amplitude. */
    int16_t signal[FFT_N];
    double f_hz = 1.2;
    int k_expect = gen_test_signal(signal, FFT_N, f_hz, 16384.0 /* ~0.5 full-scale */);

    /* Fill the FFT buffer with windowed signal (matches main_fft.c stage 3). */
    for (int i = 0; i < FFT_N; i++) {
        int16_t w = fft_window_q15[i];
        fft_buf[i].re = (int16_t)(((int32_t)signal[i] * w) >> 15);
        fft_buf[i].im = 0;
    }

    fft_q15(fft_buf, FFT_N_LOG2, fft_twiddle_q15);

    /* Print every bin as |X[k]|² so Python can diff. Format: "k,re,im,mag2\n" */
    printf("# host_fft_test: f_hz=%.3f expected_k=%d FFT_N=%d FFT_FS_DS_X1000=%d\n",
           f_hz, k_expect, FFT_N, FFT_FS_DS_X1000);
    printf("k,re,im,mag2\n");
    for (int k = 0; k < FFT_N; k++) {
        uint32_t m2 = cplx_mag2_q15(fft_buf[k]);
        printf("%d,%d,%d,%u\n", k, fft_buf[k].re, fft_buf[k].im, m2);
    }

    /* Sanity-check: the peak magnitude should be at or very near k_expect. */
    uint32_t max_m2 = 0;
    int k_peak = 0;
    for (int k = 1; k < FFT_N / 2; k++) {
        uint32_t m2 = cplx_mag2_q15(fft_buf[k]);
        if (m2 > max_m2) { max_m2 = m2; k_peak = k; }
    }
    fprintf(stderr, "host_fft_test: peak bin = %d (expected ~%d), |X|² = %u\n",
            k_peak, k_expect, max_m2);
    return (k_peak == k_expect || k_peak == k_expect - 1 || k_peak == k_expect + 1) ? 0 : 1;
}
