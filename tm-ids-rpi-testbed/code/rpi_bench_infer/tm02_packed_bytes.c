/*
 * 2. Packed bytes  -- C port of the "Packed Python" version.
 *
 * Each clause is 45 bytes (356 literal bits).  Per byte:
 *     missing |= rule[b] & ~input[b]
 * All 45 bytes are always processed.
 */

#define IMPL_NAME "02 packed bytes"
#include "tm_common.h"

static void impl_setup(void) {}

static void impl_infer_batch(int32_t scores[NS][NUM_CLASSES])
{
    for (int s = 0; s < NS; s++) {
        const uint8_t *x = lit_p[s];
        for (int c = 0; c < NUM_CLASSES; c++) {
            int32_t sum = 0;
            for (int k = 0; k < ncl[c]; k++) {
                const uint8_t *r = rules_p[c][k];
                uint8_t missing = 0, nonempty = 0;
                for (int b = 0; b < RULE_BYTES; b++) {
                    missing  |= r[b] & (uint8_t)~x[b];
                    nonempty |= r[b];
                }
                if (nonempty && !missing) sum += weights[c][k];
            }
            scores[s][c] = sum;
        }
    }
}
