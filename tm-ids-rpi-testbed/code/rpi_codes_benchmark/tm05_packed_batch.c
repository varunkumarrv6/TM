/*
 * 5. Packed, array-at-a-time  -- C port of "Packed NumPy".
 *
 * Mirrors how the NumPy version works: whole-array passes with
 * materialised intermediates instead of a fused per-clause loop.
 *
 *   tmp   = rules64 & ~x            (36000 x 9 words)
 *   fire  = ~tmp.any(-1) & nonempty (36000 flags; nonempty precomputed)
 *   score = fire.reshape(6, 6000) @ weights
 */

#define IMPL_NAME "05 packed batch (numpy-style)"
#include "tm_common.h"

#define WORDS 9

static uint64_t rules64[TOTAL_CLAUSES][WORDS];
static uint8_t  nonempty[TOTAL_CLAUSES];
static int8_t   wflat[TOTAL_CLAUSES];

static uint64_t tmp[TOTAL_CLAUSES][WORDS];
static uint8_t  fire[TOTAL_CLAUSES];

static void impl_setup(void)
{
    memset(rules64, 0, sizeof(rules64));
    for (int c = 0; c < NUM_CLASSES; c++)
        for (int k = 0; k < CLAUSES; k++) {
            int i = c * CLAUSES + k;
            memcpy(rules64[i], rules_p[c][k], RULE_BYTES);
            uint8_t ne = 0;
            for (int b = 0; b < RULE_BYTES; b++) ne |= rules_p[c][k][b];
            nonempty[i] = ne != 0;
            wflat[i]    = weights[c][k];
        }
}

static void impl_infer_batch(int32_t scores[NS][NUM_CLASSES])
{
    for (int s = 0; s < NS; s++) {
        uint64_t nx[WORDS] = {0};
        memcpy(nx, lit_p[s], RULE_BYTES);
        for (int w = 0; w < WORDS; w++) nx[w] = ~nx[w];

        /* pass 1: elementwise AND-NOT */
        for (int i = 0; i < TOTAL_CLAUSES; i++)
            for (int w = 0; w < WORDS; w++)
                tmp[i][w] = rules64[i][w] & nx[w];

        /* pass 2: any() reduction + empty mask */
        for (int i = 0; i < TOTAL_CLAUSES; i++) {
            uint64_t m = 0;
            for (int w = 0; w < WORDS; w++) m |= tmp[i][w];
            fire[i] = (m == 0) & nonempty[i];
        }

        /* pass 3: per-class dot product with weights */
        for (int c = 0; c < NUM_CLASSES; c++) {
            int32_t sum = 0;
            for (int k = 0; k < CLAUSES; k++)
                sum += fire[c * CLAUSES + k] * wflat[c * CLAUSES + k];
            scores[s][c] = sum;
        }
    }
}
