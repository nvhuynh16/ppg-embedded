/*
 * test_dsp_fixed.c — Unity unit tests for firmware/dsp_fixed.{c,h}.
 *
 * Covers:
 *   fir_q15            — impulse response, DC rejection (AC-coupled FIR)
 *   detect_peaks       — peak count on sinusoid, refractory enforcement,
 *                        below-threshold rejection, empty input
 *   hr_x100_from_peaks — known-interval HR, out-of-gate rejection, <2 peaks
 *
 * Builds host-side (no QEMU, no cross-compile). Sub-second feedback.
 */
#include "unity.h"
#include "dsp_fixed.h"
#include <math.h>
#include <stdint.h>

/* Unity calls these around each test case; no-op for our pure-function tests. */
void setUp(void) {}
void tearDown(void) {}

/* ============================================================================
 * fir_q15 tests
 * ========================================================================== */

/* Impulse response: with x = [0, ..., 0, 32767, 0, ...] (impulse at index ntaps-1)
 * the first ntaps outputs of fir_q15 should equal h_q[0..ntaps-1] scaled by
 * 32767/32768 (i.e. off by at most 1 LSB due to the right-shift truncation). */
static void test_fir_q15_impulse_response(void) {
    const int16_t h_q[3] = {10000, 20000, 30000};
    const int n_in = 8;
    /* Impulse delayed by ntaps-1 = 2 so the filter sees the impulse against
     * every tap in turn as i sweeps through ntaps-1, ntaps, ntaps+1. */
    int16_t x[8] = {0, 0, 32767, 0, 0, 0, 0, 0};
    int32_t y[8];
    int n_out = fir_q15(x, n_in, h_q, 3, y);

    TEST_ASSERT_EQUAL_INT(6, n_out);                         /* n - ntaps + 1 */
    TEST_ASSERT_INT32_WITHIN(1, 10000, y[0]);                /* h_q[0] */
    TEST_ASSERT_INT32_WITHIN(1, 20000, y[1]);                /* h_q[1] */
    TEST_ASSERT_INT32_WITHIN(1, 30000, y[2]);                /* h_q[2] */
    TEST_ASSERT_EQUAL_INT32(0, y[3]);
    TEST_ASSERT_EQUAL_INT32(0, y[4]);
    TEST_ASSERT_EQUAL_INT32(0, y[5]);
}

/* DC rejection: a length-2 AC-coupled FIR (h = [+0.5, -0.5] in Q15) on a DC
 * input must produce exactly zero output. This is the simplest possible
 * band-pass; the real PPG FIR has DC gain ≈ 0.5, so we use a synthetic FIR
 * here purely to exercise the primitive. */
static void test_fir_q15_dc_rejection(void) {
    const int16_t h_q[2] = {16384, -16384};                  /* +0.5, −0.5 */
    int16_t x[6];
    for (int i = 0; i < 6; i++) x[i] = 30000;                /* DC */
    int32_t y[6];
    int n_out = fir_q15(x, 6, h_q, 2, y);

    TEST_ASSERT_EQUAL_INT(5, n_out);
    for (int i = 0; i < n_out; i++) {
        TEST_ASSERT_EQUAL_INT32(0, y[i]);
    }
}

/* ============================================================================
 * detect_peaks tests
 * ========================================================================== */

/* A 5-cycle full-amplitude sinusoid sampled densely enough that each cycle's
 * crest can be resolved. With refractory shorter than one period, we expect
 * exactly 5 detected peaks. */
static void test_detect_peaks_sinusoid_count(void) {
    const int len = 500;
    const int cycles = 5;
    const int refractory = 30;
    int32_t y[500];
    for (int i = 0; i < len; i++) {
        y[i] = (int32_t)(30000.0 * sin(2.0 * M_PI * cycles * i / (double)len));
    }
    int peaks[64];
    int n = detect_peaks(y, len, refractory, peaks, 64);
    TEST_ASSERT_EQUAL_INT(cycles, n);
}

/* Two adjacent peaks separated by less than the refractory window: the second
 * must be suppressed. Construct y with two triangular bumps spaced 10 samples
 * apart and refractory = 50. */
static void test_detect_peaks_refractory_enforced(void) {
    int32_t y[200] = {0};
    /* Bump 1 centred at 50, bump 2 at 60 (gap = 10 < refractory 50). */
    for (int i = 30; i <= 70; i++) {
        int d1 = i - 50; if (d1 < 0) d1 = -d1;
        int d2 = i - 60; if (d2 < 0) d2 = -d2;
        int v1 = 20000 - 1000 * d1; if (v1 < 0) v1 = 0;
        int v2 = 20000 - 1000 * d2; if (v2 < 0) v2 = 0;
        y[i] = v1 > v2 ? v1 : v2;
    }
    int peaks[16];
    int n = detect_peaks(y, 200, 50, peaks, 16);
    TEST_ASSERT_EQUAL_INT(1, n);                             /* second bump suppressed */
}

/* Monotonic input has no local maxima → no peaks regardless of threshold.
 * (The detect_peaks contract gates on local-max AND threshold AND refractory;
 * the local-max gate alone suppresses peaks here.) */
static void test_detect_peaks_monotonic_no_peaks(void) {
    int32_t y[200];
    for (int i = 0; i < 200; i++) y[i] = i * 10;             /* strictly increasing */
    int peaks[16];
    int n = detect_peaks(y, 200, 30, peaks, 16);
    TEST_ASSERT_EQUAL_INT(0, n);
}

/* Empty / single-sample input: no peaks. */
static void test_detect_peaks_empty(void) {
    int32_t y[1] = {0};
    int peaks[4];
    int n = detect_peaks(y, 0, 10, peaks, 4);
    TEST_ASSERT_EQUAL_INT(0, n);
    n = detect_peaks(y, 1, 10, peaks, 4);
    TEST_ASSERT_EQUAL_INT(0, n);
}

/* ============================================================================
 * hr_x100_from_peaks tests
 * ========================================================================== */

/* Peaks at constant 1-second intervals (fs=125) → 60 bpm → HR_x100 = 6000. */
static void test_hr_x100_from_peaks_60bpm(void) {
    int peaks[6] = {0, 125, 250, 375, 500, 625};
    uint32_t hr = hr_x100_from_peaks(peaks, 6, 125);
    TEST_ASSERT_UINT32_WITHIN(50, 6000, hr);                 /* ±0.5 bpm */
}

/* Peaks at ~0.5 s intervals at fs=125 → ~120 bpm. Integer sample positions
 * quantize the period to 62 or 63 samples (true 62.5), so the median-interval
 * HR lands at 119.05 bpm (63-sample period) or 120.97 bpm (62-sample).
 * Tolerance ±200 (2 bpm) absorbs that sample-quantization noise. */
static void test_hr_x100_from_peaks_120bpm(void) {
    int peaks[6] = {0, 63, 125, 188, 250, 313};
    uint32_t hr = hr_x100_from_peaks(peaks, 6, 125);
    TEST_ASSERT_UINT32_WITHIN(200, 12000, hr);
}

/* Intervals outside [0.4 s, 1.5 s] are rejected. All-too-fast intervals
 * (5 ms) yield zero valid intervals → sentinel 0. */
static void test_hr_x100_from_peaks_out_of_gate_returns_zero(void) {
    int peaks[6] = {0, 1, 2, 3, 4, 5};                       /* 1-sample = 8 ms apart */
    uint32_t hr = hr_x100_from_peaks(peaks, 6, 125);
    TEST_ASSERT_EQUAL_UINT32(0, hr);
}

/* Too few peaks (need ≥2 intervals → ≥3 peaks for the median to be meaningful):
 * 0, 1, or 2 peaks should return 0. */
static void test_hr_x100_from_peaks_too_few(void) {
    int peaks[2] = {0, 125};
    TEST_ASSERT_EQUAL_UINT32(0, hr_x100_from_peaks(peaks, 0, 125));
    TEST_ASSERT_EQUAL_UINT32(0, hr_x100_from_peaks(peaks, 1, 125));
}

/* Mixed intervals: 4 valid 1-s intervals + 1 outlier (0.05 s) — median should
 * pick the 1-s intervals and report 60 bpm. */
static void test_hr_x100_from_peaks_outlier_rejection(void) {
    int peaks[6] = {0, 125, 250, 375, 380, 505};             /* 380→505 = 1 s; 375→380 = outlier */
    uint32_t hr = hr_x100_from_peaks(peaks, 6, 125);
    TEST_ASSERT_UINT32_WITHIN(100, 6000, hr);
}

/* ============================================================================
 * Test runner
 * ========================================================================== */

int main(void) {
    UNITY_BEGIN();

    /* fir_q15 */
    RUN_TEST(test_fir_q15_impulse_response);
    RUN_TEST(test_fir_q15_dc_rejection);

    /* detect_peaks */
    RUN_TEST(test_detect_peaks_sinusoid_count);
    RUN_TEST(test_detect_peaks_refractory_enforced);
    RUN_TEST(test_detect_peaks_monotonic_no_peaks);
    RUN_TEST(test_detect_peaks_empty);

    /* hr_x100_from_peaks */
    RUN_TEST(test_hr_x100_from_peaks_60bpm);
    RUN_TEST(test_hr_x100_from_peaks_120bpm);
    RUN_TEST(test_hr_x100_from_peaks_out_of_gate_returns_zero);
    RUN_TEST(test_hr_x100_from_peaks_too_few);
    RUN_TEST(test_hr_x100_from_peaks_outlier_rejection);

    return UNITY_END();
}
