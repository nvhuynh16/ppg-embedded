/* dsp_resp.c — Goertzel + decimation primitives for the BW-path
 * respiration-rate estimator. See dsp_resp.h for the recurrence + dynamic-
 * range analysis. */
#include "dsp_resp.h"

uint32_t goertzel_q15(const int16_t *x, int n, int16_t cos_q15) {
    int32_t s_prev = 0;
    int32_t s_prev2 = 0;
    for (int i = 0; i < n; i++) {
        /* 2·cos·s_prev in Q0:   (cos_q15 * s_prev) >> 14  ≡  (2·cos·s_prev / 2^15) << 1
         * keeps s in the same domain as x (int16-scale samples, growing linearly with N). */
        int32_t s = (int32_t)x[i] + (int32_t)(((int64_t)cos_q15 * s_prev) >> 14) - s_prev2;
        s_prev2 = s_prev;
        s_prev = s;
    }
    /* mag² = s_prev² + s_prev2² - (2·cos·s_prev·s_prev2)
     *      = s_prev² + s_prev2² - ((cos_q15·s_prev·s_prev2) >> 14)
     * int64 intermediate to avoid wrap; >>8 to fit uint32 for argmax. */
    int64_t m1 = (int64_t)s_prev * s_prev;
    int64_t m2 = (int64_t)s_prev2 * s_prev2;
    int64_t m3 = ((int64_t)cos_q15 * s_prev * s_prev2) >> 14;
    int64_t mag2 = m1 + m2 - m3;
    if (mag2 < 0) mag2 = 0;
    return (uint32_t)(mag2 >> 8);
}

void goertzel_scan(const int16_t *x, int n,
                   const int16_t *cos_q15_arr, int k,
                   uint32_t *mag2_out) {
    for (int i = 0; i < k; i++) {
        mag2_out[i] = goertzel_q15(x, n, cos_q15_arr[i]);
    }
}

int decimate_to_int16(const int32_t *in, int n_in, int decim, int16_t *out) {
    int n_out = 0;
    for (int i = 0; i < n_in; i += decim) {
        int32_t v = in[i];
        if (v >  32767) v =  32767;
        if (v < -32768) v = -32768;
        out[n_out++] = (int16_t)v;
    }
    return n_out;
}
