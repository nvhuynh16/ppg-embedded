/* Host-side DSP canary — runs fir_q15 + detect_peaks under the system gcc, no
 * ARM toolchain or QEMU needed. Fast pre-QEMU regression: compiles in ms,
 * executes in microseconds.
 *
 * Exits 0 if the host Q15 pipeline matches REF_HR_X100 within 3 bpm (300 hundredths).
 * NOT a substitute for the QEMU run — only the cross-compile + QEMU path exercises
 * arm-none-eabi-gcc, libgcc int64 helpers, the linker script, and semihosting.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "dsp_fixed.h"
#include "generated/ppg_data.h"

static int32_t filt[PPG_N];
static int peak_idx[256];

int main(void) {
    int L = fir_q15(ppg_q15, PPG_N, fir_q15_coef, FIR_NTAPS, filt);
    int refractory = REFRACTORY_SAMPLES;
    int peaks = detect_peaks(filt, L, refractory, peak_idx, 256);
    uint32_t hr_x100 = hr_x100_from_peaks(peak_idx, peaks, PPG_FS);

    printf("host-test: PEAKS=%d VALID_SAMPLES=%d HR_X100=%u REF_HR_X100=%u\n",
           peaks, L, hr_x100, (unsigned)REF_HR_X100);

    long delta = (long)hr_x100 - (long)REF_HR_X100;
    if (delta < 0) delta = -delta;
    if (delta > 300) {
        printf("FAIL: |delta| = %ld hundredths of bpm exceeds 300 (3 bpm) tolerance\n", delta);
        return 1;
    }
    printf("PASS: |delta| = %ld hundredths of bpm within 3 bpm tolerance\n", delta);
    return 0;
}
