/*
 * 3. Packed bytes + early exit  -- C port of "Packed Python with Early Exit".
 *
 * Stops scanning a clause at the first byte with a missing required literal.
 * The empty-clause check is only needed for clauses that survive the scan.
 */

#define IMPL_NAME "03 packed early-exit"
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
                int b;
                for (b = 0; b < RULE_BYTES; b++)
                    if (r[b] & (uint8_t)~x[b]) break;
                if (b < RULE_BYTES) continue;

                uint8_t nonempty = 0;
                for (b = 0; b < RULE_BYTES; b++) nonempty |= r[b];
                if (nonempty) sum += weights[c][k];
            }
            scores[s][c] = sum;
        }
    }
}
