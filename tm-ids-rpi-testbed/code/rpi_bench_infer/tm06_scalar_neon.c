/*
 * 6. "Scalar" NEON  -- one clause at a time, NEON across the clause's bytes.
 *
 * Rules are zero-padded to 48 bytes = 3 x 128-bit vectors.  For one clause:
 *     missing = OR_v (rule_v & ~input_v)       (vbicq)
 *     fires   = maxv(missing) == 0 && maxv(OR_v rule_v) != 0
 * Clauses are still visited sequentially (intra-clause SIMD only).
 */

#define IMPL_NAME "06 scalar NEON (intra-clause)"
#include "tm_common.h"
#include "neon_compat.h"

#define PAD_BYTES 48

static uint8_t rules48[NUM_CLASSES][CL_PAD][PAD_BYTES] __attribute__((aligned(16)));

static void impl_setup(void)
{
    memset(rules48, 0, sizeof(rules48));
    for (int c = 0; c < NUM_CLASSES; c++)
        for (int k = 0; k < ncl[c]; k++)
            memcpy(rules48[c][k], rules_p[c][k], RULE_BYTES);
}

static void impl_infer_batch(int32_t scores[NS][NUM_CLASSES])
{
    for (int s = 0; s < NS; s++) {
        uint8_t xb[PAD_BYTES] __attribute__((aligned(16))) = {0};
        memcpy(xb, lit_p[s], RULE_BYTES);
        uint8x16_t x0 = vld1q_u8(xb), x1 = vld1q_u8(xb + 16), x2 = vld1q_u8(xb + 32);

        for (int c = 0; c < NUM_CLASSES; c++) {
            int32_t sum = 0;
            for (int k = 0; k < ncl[c]; k++) {
                const uint8_t *r = rules48[c][k];
                uint8x16_t r0 = vld1q_u8(r), r1 = vld1q_u8(r + 16), r2 = vld1q_u8(r + 32);
                uint8x16_t missing = vorrq_u8(vorrq_u8(vbicq_u8(r0, x0), vbicq_u8(r1, x1)), vbicq_u8(r2, x2));
                uint8x16_t ne = vorrq_u8(vorrq_u8(r0, r1), r2);
                if (vmaxvq_u8(missing) == 0 && vmaxvq_u8(ne) != 0)
                    sum += weights[c][k];
            }
            scores[s][c] = sum;
        }
    }
}
