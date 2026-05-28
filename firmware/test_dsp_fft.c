/*
 * test_dsp_fft.c — Unity unit tests for firmware/dsp_fft.{c,h}.
 *
 * Covers:
 *   fft_q15         — impulse → flat spectrum; pure cosine → peak at expected bin
 *   cplx_mag2_q15   — known re/im → known mag²
 *
 * The full spectrum bit-exactness vs numpy.fft is verified separately by
 * src/verify_fft.py (running firmware/host_fft_test against numpy). These
 * Unity tests cover algebraic properties; the Python verifier covers numerical
 * agreement with the reference implementation.
 *
 * Twiddle tables are reused from firmware/generated/fft_data.h.
 */
#include "unity.h"
#include "dsp_fft.h"
#include "generated/fft_data.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

void setUp(void) {}
void tearDown(void) {}

/* ============================================================================
 * cplx_mag2_q15 tests
 * ========================================================================== */

static void test_cplx_mag2_q15_zero(void) {
    cplx_q15_t v = {0, 0};
    TEST_ASSERT_EQUAL_UINT32(0, cplx_mag2_q15(v));
}

static void test_cplx_mag2_q15_real_only(void) {
    cplx_q15_t v = {100, 0};
    TEST_ASSERT_EQUAL_UINT32(10000, cplx_mag2_q15(v));
}

static void test_cplx_mag2_q15_imag_only(void) {
    cplx_q15_t v = {0, -200};
    TEST_ASSERT_EQUAL_UINT32(40000, cplx_mag2_q15(v));
}

static void test_cplx_mag2_q15_pythagorean(void) {
    /* 3-4-5 triangle */
    cplx_q15_t v = {3000, 4000};
    TEST_ASSERT_EQUAL_UINT32(9000000 + 16000000, cplx_mag2_q15(v));
}

static void test_cplx_mag2_q15_extreme(void) {
    /* Max Q15 in both axes: 32767² + 32767² ≈ 2.15·10⁹, fits uint32. */
    cplx_q15_t v = {32767, -32767};
    uint32_t expected = 2u * (uint32_t)32767 * (uint32_t)32767;
    TEST_ASSERT_EQUAL_UINT32(expected, cplx_mag2_q15(v));
}

/* ============================================================================
 * fft_q15 tests
 *
 * These work on the auto-generated FFT tables (FFT_N=256, twiddle table from
 * fft_data.h). Each test populates the FFT buffer with a known input, runs
 * fft_q15 in-place, and checks the resulting spectrum.
 * ========================================================================== */

static cplx_q15_t fft_buf[FFT_N];

/* Impulse at n=0 (real-only) → flat magnitude spectrum: X[k] = x[0] for all k
 * in the unscaled DFT. The firmware's per-stage `>>1` scaling divides by N,
 * so each bin's expected magnitude is amp/N. We don't test exact equality
 * (Q15 truncation in 8 butterfly stages introduces a few LSB of noise) — we
 * verify *flatness*: every bin's mag² is within a small factor of bin 1's. */
static void test_fft_q15_impulse_flat_spectrum(void) {
    const int16_t amp = 16384;                               /* 0.5 in Q15 */
    const int expected_re_im = amp / FFT_N;                  /* = 64 at N=256 */
    for (int i = 0; i < FFT_N; i++) {
        fft_buf[i].re = (i == 0) ? amp : 0;
        fft_buf[i].im = 0;
    }
    fft_q15(fft_buf, FFT_N_LOG2, fft_twiddle_q15);

    /* Reference magnitude² ≈ (amp/N)². */
    const uint32_t mag2_expected = (uint32_t)expected_re_im * (uint32_t)expected_re_im;
    /* Allow each bin to be within [0.25×, 2×] of expected mag² — wide enough
     * to absorb Q15-truncation noise across 8 stages, tight enough to fail
     * if the spectrum is *not* flat. */
    for (int k = 1; k < FFT_N / 2; k++) {
        uint32_t m = cplx_mag2_q15(fft_buf[k]);
        TEST_ASSERT_TRUE(m >= mag2_expected / 4);
        TEST_ASSERT_TRUE(m <= mag2_expected * 2);
    }
}

/* Pure cosine at k_target → peak magnitude at bin k_target (and its mirror
 * at N-k_target). */
static void test_fft_q15_cosine_peak_at_expected_bin(void) {
    const int k_target = 20;                                 /* arbitrary in-band bin */
    const double amp = 8000.0;                               /* well below saturation */
    for (int i = 0; i < FFT_N; i++) {
        fft_buf[i].re = (int16_t)(amp * cos(2.0 * M_PI * k_target * i / FFT_N));
        fft_buf[i].im = 0;
    }
    fft_q15(fft_buf, FFT_N_LOG2, fft_twiddle_q15);

    /* Find the magnitude peak in [1, N/2). */
    uint32_t max_mag = 0;
    int k_peak = 0;
    for (int k = 1; k < FFT_N / 2; k++) {
        uint32_t m = cplx_mag2_q15(fft_buf[k]);
        if (m > max_mag) { max_mag = m; k_peak = k; }
    }
    TEST_ASSERT_EQUAL_INT(k_target, k_peak);
}

/* DC input (constant non-zero real) → energy concentrated at bin 0. */
static void test_fft_q15_dc_peak_at_bin_zero(void) {
    const int16_t v = 5000;
    for (int i = 0; i < FFT_N; i++) {
        fft_buf[i].re = v;
        fft_buf[i].im = 0;
    }
    fft_q15(fft_buf, FFT_N_LOG2, fft_twiddle_q15);

    uint32_t m0 = cplx_mag2_q15(fft_buf[0]);
    /* Every other bin should be near zero compared to bin 0. */
    for (int k = 1; k < FFT_N / 2; k++) {
        uint32_t m = cplx_mag2_q15(fft_buf[k]);
        TEST_ASSERT_TRUE(m * 1000u < m0);                    /* >1000× less than DC */
    }
}

/* ============================================================================
 * Test runner
 * ========================================================================== */

int main(void) {
    UNITY_BEGIN();

    RUN_TEST(test_cplx_mag2_q15_zero);
    RUN_TEST(test_cplx_mag2_q15_real_only);
    RUN_TEST(test_cplx_mag2_q15_imag_only);
    RUN_TEST(test_cplx_mag2_q15_pythagorean);
    RUN_TEST(test_cplx_mag2_q15_extreme);

    RUN_TEST(test_fft_q15_impulse_flat_spectrum);
    RUN_TEST(test_fft_q15_cosine_peak_at_expected_bin);
    RUN_TEST(test_fft_q15_dc_peak_at_bin_zero);

    return UNITY_END();
}
