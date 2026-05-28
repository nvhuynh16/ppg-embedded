/*
 * main_fft.c — alternative HR-estimation entry point using the spectral path.
 *
 * Pipeline:
 *   1. Q15 band-pass FIR on ppg_q15[]                 (existing fir_q15)
 *   2. Decimate filtered signal by FFT_DECIM           (anti-aliasing-free: the
 *      band-pass already cut at f2 < fs/(2·decim))
 *   3. Saturate-cast to int16, multiply by Hamming Q15 window
 *   4. In-place 256-pt radix-2 Q15 FFT                 (dsp_fft.c)
 *   5. Peak-magnitude search in bins [FFT_K_MIN, FFT_K_MAX] (the HR pass-band)
 *   6. Parabolic interpolation on (k-1, k, k+1) → fractional bin δ
 *   7. HR_x100 = round((k + δ) · fs_ds · 6000 / N)
 *
 * Emits PEAK_BIN, FFT_K, HR_X100, REF_HR_X100 via semihosting; format mirrors
 * the peak-detector main.c so validate.py / batch_validate.py parse it the same way.
 */
#include <stdint.h>
#include "semihost.h"
#include "dsp_fixed.h"
#include "dsp_fft.h"
#include "generated/ppg_data.h"
#include "generated/fft_data.h"

static int32_t filt[PPG_N];        /* FIR output (.bss in SRAM) */
static cplx_q15_t fft_buf[FFT_N];  /* FFT workspace (.bss) */

static inline int16_t sat16(int32_t v) {
    if (v >  32767) return  32767;
    if (v < -32768) return -32768;
    return (int16_t)v;
}

int main(void) {
    /* Stage 1: band-pass FIR (re-use existing Q15 implementation). */
    int L = fir_q15(ppg_q15, PPG_N, fir_q15_coef, FIR_NTAPS, filt);

    /* Stage 2-3: decimate + window into the FFT workspace. */
    for (int i = 0; i < FFT_N; i++) {
        int idx = i * FFT_DECIM;
        int16_t s  = (idx < L) ? sat16(filt[idx]) : 0;
        int16_t w  = fft_window_q15[i];
        fft_buf[i].re = (int16_t)(((int32_t)s * w) >> 15);
        fft_buf[i].im = 0;
    }

    /* Stage 4: 256-pt radix-2 DIT FFT. */
    fft_q15(fft_buf, FFT_N_LOG2, fft_twiddle_q15);

    /* Stage 5: find magnitude peak in the HR band. */
    uint32_t max_mag2 = 0;
    int k_peak = FFT_K_MIN;
    for (int k = FFT_K_MIN; k <= FFT_K_MAX; k++) {
        uint32_t m = cplx_mag2_q15(fft_buf[k]);
        if (m > max_mag2) { max_mag2 = m; k_peak = k; }
    }

    /* Stage 5b: sub-harmonic check. PPG morphologies where the 2nd harmonic
     * dominates the fundamental (bidmc47, bidmc23 on the 53-record sweep) make
     * the raw peak-bin search lock at 2× HR. If the bin at k_peak/2 is in band
     * AND carries ≥ α the energy of the peak (α = 1 / FFT_SUBHARMONIC_DIVISOR,
     * emitted from src/reference.py via fft_data.h), prefer it as the true
     * fundamental. Integer compare on mag² avoids float here. */
    int k_sub = k_peak / 2;
    if (k_sub >= FFT_K_MIN) {
        uint32_t sub_mag2 = cplx_mag2_q15(fft_buf[k_sub]);
        if (sub_mag2 >= max_mag2 / FFT_SUBHARMONIC_DIVISOR) {
            k_peak   = k_sub;
            max_mag2 = sub_mag2;
        }
    }

    /* Stage 6: parabolic interpolation on log/linear magnitudes. We use mag² —
     * shape-equivariant under √ for sufficiently peaked distributions; introduces
     * a small bias proportional to peak narrowness, dwarfed by Q15 noise in
     * practice. Float emulation via libgcc is fine for ~5 ops. */
    float delta = 0.0f;
    uint32_t y_m1 = (k_peak > 0)         ? cplx_mag2_q15(fft_buf[k_peak - 1]) : 0;
    uint32_t y_p1 = (k_peak < FFT_N / 2) ? cplx_mag2_q15(fft_buf[k_peak + 1]) : 0;
    float fm1 = (float)y_m1, f0 = (float)max_mag2, fp1 = (float)y_p1;
    float den = fm1 - 2.0f * f0 + fp1;
    if (den != 0.0f) {
        delta = 0.5f * (fm1 - fp1) / den;
        if (delta < -0.5f) delta = -0.5f;
        if (delta >  0.5f) delta =  0.5f;
    }

    /* Stage 7: HR. fs_ds is emitted ×1000 to keep the header integer. */
    float f_peak_hz = ((float)k_peak + delta) * (float)FFT_FS_DS_X1000 / 1000.0f / (float)FFT_N;
    uint32_t hr_x100 = (uint32_t)(f_peak_hz * 6000.0f + 0.5f);

    sh_write0("PPG embedded HR estimator (Q15 FFT spectral path)\n");
    sh_write0("PEAK_BIN=");      sh_print_uint((uint32_t)k_peak);              sh_write0("\n");
    sh_write0("VALID_SAMPLES="); sh_print_uint((uint32_t)L);                   sh_write0("\n");
    sh_write0("HR_X100=");       sh_print_uint(hr_x100);                       sh_write0("\n");
    sh_write0("REF_HR_X100=");   sh_print_uint((uint32_t)REF_HR_X100);         sh_write0("\n");
    sh_write0("PEAKS=");         sh_print_uint((uint32_t)k_peak);              sh_write0("\n");

    sh_exit();   /* clean QEMU exit (matches main.c) */
    return 0;
}
