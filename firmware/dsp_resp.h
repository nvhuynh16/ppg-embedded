/* dsp_resp.h — Q15 Goertzel for the BW-path respiration-rate estimator.
 *
 * The Goertzel filter is a second-order IIR resonator tuned to a single
 * frequency. Running it against N samples gives the equivalent of one DFT
 * bin at that frequency — much cheaper than a full FFT when only a handful
 * of bins are needed. Here we scan 24 candidate frequencies covering the
 * respiration band (6–30 BrPM at 1 BrPM resolution).
 *
 * Recurrence (with cos_q15 = q15(cos(2π·f/fs_ds))):
 *   s_new   = x[i] + (2·cos_q15 · s_prev >> 15) - s_prev2
 *           = x[i] + (cos_q15 · s_prev >> 14) - s_prev2
 *   s_prev2 = s_prev
 *   s_prev  = s_new
 *
 * Worst case |s|: |s| ≤ N · max|x| ≈ N · 32767 (Goertzel resonator gain at
 * resonance grows linearly with N). For N ≤ 256 we have |s| ≤ 8.4·10⁶,
 * comfortably inside int32.
 *
 * Output magnitude² (real-valued spectrum estimate at the target frequency):
 *   mag² = s_prev² + s_prev2² - (2·cos · s_prev · s_prev2)
 *        = s_prev² + s_prev2² - ((cos_q15 · s_prev · s_prev2) >> 14)
 *
 * Worst case |mag²|: 3·(N·32767)² ≈ 3·(8.4·10⁶)² ≈ 2.1·10¹⁴, fits in int64.
 * We right-shift by 8 to fit into uint32 for the argmax search.
 */
#ifndef DSP_RESP_H
#define DSP_RESP_H

#include <stdint.h>

/* Single-frequency Goertzel. Returns |X[f]|² (right-shifted by 8 to fit uint32). */
uint32_t goertzel_q15(const int16_t *x, int n, int16_t cos_q15);

/* Scan K candidate frequencies; fills mag2_out[K] with each bin's mag². */
void goertzel_scan(const int16_t *x, int n,
                   const int16_t *cos_q15_arr, int k,
                   uint32_t *mag2_out);

/* Decimate int32 input by `decim`, saturating-cast to int16. Returns count
 * written to `out` (= n_in / decim). */
int decimate_to_int16(const int32_t *in, int n_in, int decim, int16_t *out);

#endif /* DSP_RESP_H */
