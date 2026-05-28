/* SQI-CNN firmware entry point — Cortex-M3 / QEMU lm3s6965evb.
 *
 * Runs the 1D-CNN SQI inference on a deterministic synthetic PPG window
 * (defined in generated/cnn_data.h) and reports the logit + accept decision
 * via semihosting. Bit-exact contract with src/sqi_cnn.py — verify via
 * src/verify_cnn.py.
 *
 * Why a fixed test input: this entry point is the embedded-side proof of
 * the CNN port. Production sensing wiring is out of scope (no hardware on
 * the QEMU target). The test signal makes the bit-exactness check
 * reproducible without any external dataset.
 */
#include <stdint.h>
#include "semihost.h"
#include "dsp_cnn.h"
#include "generated/cnn_data.h"

int main(void) {
    sh_write0("PPG embedded SQI-CNN gate (float kernel, Cortex-M3 / QEMU)\n");

    float logit;
    int accept;
    cnn_sqi_forward(cnn_test_input, &logit, &accept);

    /* Encode logit as int (×1e6) so we can transmit via the existing
     * integer-only semihosting helper. verify_cnn.py decodes the same way. */
    int32_t logit_x1e6 = (int32_t)(logit * 1.0e6f);
    sh_write0("LOGIT_X1E6=");
    sh_print_uint((uint32_t)(logit_x1e6 < 0 ? -logit_x1e6 : logit_x1e6));
    sh_write0(logit_x1e6 < 0 ? " neg\n" : " pos\n");

    sh_write0("EXPECTED_LOGIT_X1E6=");
    sh_print_uint((uint32_t)(CNN_TEST_EXPECTED_LOGIT_X1E6 < 0
                             ? -CNN_TEST_EXPECTED_LOGIT_X1E6
                             : CNN_TEST_EXPECTED_LOGIT_X1E6));
    sh_write0(CNN_TEST_EXPECTED_LOGIT_X1E6 < 0 ? " neg\n" : " pos\n");

    sh_write0(accept ? "ACCEPT=1\n" : "ACCEPT=0\n");

    sh_exit();
    return 0;
}
