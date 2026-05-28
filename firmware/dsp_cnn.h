/* 1D-CNN SQI inference kernel (float32) for Cortex-M3. */
#ifndef DSP_CNN_H
#define DSP_CNN_H

#include <stdint.h>

/* Forward pass of the SQI CNN on one 8-second band-passed PPG window
 * (CNN_WIN_LEN samples, z-scored float). Writes the raw logit and the
 * accept decision (probability >= 0.45) to the out-parameters.
 *
 * The input MUST be z-scored (per-window mean-zero, unit-std) — same
 * preprocessing as the numpy inference path in src/sqi_cnn.py.
 *
 * Returns nothing; both *logit and *accept are mandatory out-params.
 */
void cnn_sqi_forward(const float input[/*CNN_WIN_LEN*/],
                     float *logit, int *accept);

#endif  /* DSP_CNN_H */
