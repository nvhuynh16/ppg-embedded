/*
 * main_rr.c — third entry point: BW-path respiration-rate estimator.
 *
 * Pipeline:
 *   1. Q15 lowpass FIR (0.5 Hz cutoff) on the original ppg_q15[] — reuses
 *      `fir_q15` from dsp_fixed.c, just with different coefficients
 *      (fir_rr_coef[] auto-emitted by reference.py).
 *   2. Decimate the FIR output by DECIM_RR (default 32; fs=125 → fs_ds≈3.9 Hz).
 *   3. Goertzel-scan the decimated signal at 24 candidate frequencies
 *      covering 0.10–0.50 Hz (6–30 BrPM at 1-BrPM resolution).
 *   4. argmax over mag²; RR_X100 = freq_x1000[k_peak] · 6.
 *
 * Mirrors main.c / main_fft.c's semihosting conventions; emits RR_X100=
 * so the validator can parse it the same way main.c emits HR_X100=.
 *
 * AM (amplitude-modulation) and FM (frequency-modulation / RSA) channels +
 * Karlen-2013 smart fusion are NOT in firmware — Python draft only, see
 * src/_respiration_three_channel_draft.py.
 */
#include <stdint.h>
#include "semihost.h"
#include "dsp_fixed.h"
#include "dsp_resp.h"
#include "generated/ppg_data.h"
#include "generated/resp_data.h"

#define MAX_DS_SAMPLES   (PPG_N / DECIM_RR + 1)   /* worst-case decimated length */

static int32_t filt_rr[PPG_N];                    /* .bss — FIR output */
static int16_t ds_buf[MAX_DS_SAMPLES];            /* .bss — decimated */
static uint32_t mag2_out[GOERTZEL_K];             /* .bss — per-bin mag² */

int main(void) {
    /* Stage 1: lowpass FIR. Output range bounded by Σ|h_q|·max_x ≤ ~int16;
     * see docs/qformat_proof.md for the analytical bound on a similar FIR. */
    int L = fir_q15(ppg_q15, PPG_N, fir_rr_coef, FIR_RR_NTAPS, filt_rr);

    /* Stage 2: decimate to fs_ds ≈ 3.9 Hz with sat16 cast. */
    int n_ds = decimate_to_int16(filt_rr, L, DECIM_RR, ds_buf);

    /* Stage 3: Goertzel scan over the respiration band. */
    goertzel_scan(ds_buf, n_ds, goertzel_cos_q15, GOERTZEL_K, mag2_out);

    /* Stage 4: argmax peak-bin search. */
    uint32_t max_mag2 = 0;
    int k_peak = 0;
    for (int k = 0; k < GOERTZEL_K; k++) {
        if (mag2_out[k] > max_mag2) { max_mag2 = mag2_out[k]; k_peak = k; }
    }

    /* RR_X100 = freq_x1000[k] · 60 / 1000 · 100 = freq_x1000[k] · 6.
     * E.g. 250 mHz = 0.25 Hz = 15 BrPM → RR_X100 = 1500. */
    uint32_t rr_x100 = (uint32_t)goertzel_freqs_x1000[k_peak] * 6u;

    sh_write0("PPG embedded RR estimator (Q15 BW-path Goertzel)\n");
    sh_write0("FIR_RR_VALID="); sh_print_uint((uint32_t)L);          sh_write0("\n");
    sh_write0("N_DS=");         sh_print_uint((uint32_t)n_ds);       sh_write0("\n");
    sh_write0("K_PEAK=");       sh_print_uint((uint32_t)k_peak);     sh_write0("\n");
    sh_write0("PEAK_MAG2=");    sh_print_uint(max_mag2);             sh_write0("\n");
    sh_write0("RR_X100=");      sh_print_uint(rr_x100);              sh_write0("\n");

    sh_exit();
    return 0;
}
