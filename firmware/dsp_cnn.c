/* 1D-CNN SQI inference (float32) on Cortex-M3.
 *
 * Architecture mirror of src/sqi_cnn.py:
 *
 *   input (1, 512)
 *     │
 *     ▼  Conv1d(1 → 8, k=15, pad=7) + ReLU
 *     ▼  MaxPool1d(4)              → (8, 128)
 *     ▼  Conv1d(8 → 16, k=7, pad=3) + ReLU
 *     ▼  MaxPool1d(4)              → (16, 32)
 *     ▼  AdaptiveAvgPool1d(1)      → (16,)
 *     ▼  Linear(16 → 1)            → scalar logit
 *     ▼  compare to CNN_LOGIT_THRESH → accept / reject
 *
 * Float use is intentional and isolated. The M3 has no FPU; libgcc
 * provides soft-float (linked in via `--specs=nosys.specs` in the
 * Makefile). Inference is ~14k MACs total — negligible for any
 * sensible window cadence (BVP @ 64 Hz, one inference per ~2s window).
 *
 * Memory footprint of the kernel (excluding weights in cnn_data.h):
 *   bufA  : 16 KB  (CNN_C0_OUT * CNN_WIN_LEN floats — conv0 output)
 *   bufB  :  8 KB  (CNN_C3_OUT * CNN_POST_POOL0_LEN floats — conv3 output)
 *   gap   :   64 B (16 floats)
 *   Total :  24 KB BSS for activations + ~5 KB const flash for weights.
 */
#include "dsp_cnn.h"
#include "generated/cnn_data.h"

/* Activation scratch — sized for the worst case at each layer. */
static float bufA[CNN_C0_OUT * CNN_WIN_LEN];     /* 8 × 512 */
static float bufB[CNN_C3_OUT * CNN_POST_POOL0_LEN]; /* 16 × 128 */

static inline float relu(float x) { return x > 0.0f ? x : 0.0f; }

static inline float fmaxf_(float a, float b) { return a > b ? a : b; }

void cnn_sqi_forward(const float input[],
                     float *logit_out, int *accept_out) {
    /* Layer 0: Conv1d(1 → 8, k=15, pad=7) + ReLU */
    for (int co = 0; co < CNN_C0_OUT; co++) {
        const float bias = cnn_b0[co];
        for (int i = 0; i < CNN_WIN_LEN; i++) {
            float sum = bias;
            for (int k = 0; k < CNN_C0_KERNEL; k++) {
                int p = i + k - CNN_C0_PAD;
                if (p >= 0 && p < CNN_WIN_LEN) {
                    sum += cnn_w0[co * CNN_C0_KERNEL + k] * input[p];
                }
            }
            bufA[co * CNN_WIN_LEN + i] = relu(sum);
        }
    }

    /* MaxPool1d(k=4) on bufA, in-place. Output layout: (8, 128) — first 128
     * samples of each channel slot. */
    for (int c = 0; c < CNN_C0_OUT; c++) {
        for (int i = 0; i < CNN_POST_POOL0_LEN; i++) {
            float m = bufA[c * CNN_WIN_LEN + i * CNN_POOL_K];
            for (int k = 1; k < CNN_POOL_K; k++) {
                m = fmaxf_(m, bufA[c * CNN_WIN_LEN + i * CNN_POOL_K + k]);
            }
            /* Pack into the new (8, 128) layout. Channels are laid out
             * contiguously, so destination index uses the post-pool length
             * (128), not the conv-output length (512). */
            bufA[c * CNN_POST_POOL0_LEN + i] = m;
        }
    }

    /* Layer 3: Conv1d(8 → 16, k=7, pad=3) + ReLU. Reads bufA (8, 128),
     * writes bufB (16, 128). */
    for (int co = 0; co < CNN_C3_OUT; co++) {
        const float bias = cnn_b3[co];
        for (int i = 0; i < CNN_POST_POOL0_LEN; i++) {
            float sum = bias;
            for (int ci = 0; ci < CNN_C3_IN; ci++) {
                for (int k = 0; k < CNN_C3_KERNEL; k++) {
                    int p = i + k - CNN_C3_PAD;
                    if (p >= 0 && p < CNN_POST_POOL0_LEN) {
                        sum += cnn_w3[(co * CNN_C3_IN + ci) * CNN_C3_KERNEL + k]
                             * bufA[ci * CNN_POST_POOL0_LEN + p];
                    }
                }
            }
            bufB[co * CNN_POST_POOL0_LEN + i] = relu(sum);
        }
    }

    /* MaxPool1d(k=4) on bufB, in-place → (16, 32). */
    for (int c = 0; c < CNN_C3_OUT; c++) {
        for (int i = 0; i < CNN_POST_POOL3_LEN; i++) {
            float m = bufB[c * CNN_POST_POOL0_LEN + i * CNN_POOL_K];
            for (int k = 1; k < CNN_POOL_K; k++) {
                m = fmaxf_(m, bufB[c * CNN_POST_POOL0_LEN + i * CNN_POOL_K + k]);
            }
            bufB[c * CNN_POST_POOL3_LEN + i] = m;
        }
    }

    /* AdaptiveAvgPool1d(1) — Global Average Pool over (16, 32) → (16,). */
    float gap[CNN_FC_IN];
    for (int c = 0; c < CNN_FC_IN; c++) {
        float sum = 0.0f;
        for (int i = 0; i < CNN_POST_POOL3_LEN; i++) {
            sum += bufB[c * CNN_POST_POOL3_LEN + i];
        }
        gap[c] = sum / (float)CNN_POST_POOL3_LEN;
    }

    /* Linear(16 → 1) + threshold compare (no sigmoid — equivalent). */
    float logit = cnn_b8[0];
    for (int i = 0; i < CNN_FC_IN; i++) {
        logit += cnn_w8[i] * gap[i];
    }
    *logit_out = logit;
    *accept_out = (logit >= CNN_LOGIT_THRESH) ? 1 : 0;
}
