/*
 * 4. Packed 64-bit words  -- C port of "Packed 64-bit Python".
 *
 * 66 bytes are zero-padded to 72 bytes = 9 uint64 words per clause, so a
 * clause is checked with 9 AND-NOT/OR operations instead of 66.
 * Padding is zero in the rule, so ~input padding never creates a miss.
 */

#define IMPL_NAME "04 packed 64-bit"
#include "tm_common.h"

#define WORDS 9

static uint64_t rules64[NUM_CLASSES][CLAUSES][WORDS];

static void impl_setup(void)
{
    memset(rules64, 0, sizeof(rules64));
    for (int c = 0; c < NUM_CLASSES; c++)
        for (int k = 0; k < CLAUSES; k++)
            memcpy(rules64[c][k], rules_p[c][k], RULE_BYTES);
}

static void impl_infer_batch(int32_t scores[NS][NUM_CLASSES])
{
    for (int s = 0; s < NS; s++) {
        uint64_t x[WORDS] = {0};
        memcpy(x, lit_p[s], RULE_BYTES);

        for (int c = 0; c < NUM_CLASSES; c++) {
            int32_t sum = 0;
            for (int k = 0; k < CLAUSES; k++) {
                const uint64_t *r = rules64[c][k];
                uint64_t missing = 0, nonempty = 0;
                for (int w = 0; w < WORDS; w++) {
                    missing  |= r[w] & ~x[w];
                    nonempty |= r[w];
                }
                if (nonempty && !missing) sum += weights[c][k];
            }
            scores[s][c] = sum;
        }
    }
}
