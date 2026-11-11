/*
 * 1. Normal (unpacked) baseline  -- C port of the "Normal NumPy" version.
 *
 * Rules and literals are one byte per literal (526 bytes per clause).
 * Every clause is evaluated over all 526 literals, no early exit, exactly
 * like the vectorised  (rules & ~literals).any(-1)  reduction.
 */

#define IMPL_NAME "01 normal (unpacked)"
#include "tm_common.h"

static void impl_setup(void) {}

static void impl_infer_batch(int32_t scores[NS][NUM_CLASSES])
{
    for (int s = 0; s < NS; s++) {
        const uint8_t *l = lit_u[s];
        for (int c = 0; c < NUM_CLASSES; c++) {
            int32_t sum = 0;
            for (int k = 0; k < CLAUSES; k++) {
                const uint8_t *r = rules_u[c][k];
                uint8_t violated = 0, nonempty = 0;
                for (int i = 0; i < LITERALS; i++) {
                    violated |= r[i] & (uint8_t)(l[i] ^ 1);
                    nonempty |= r[i];
                }
                if (nonempty && !violated) sum += weights[c][k];
            }
            scores[s][c] = sum;
        }
    }
}
