/*
 * 10. Final optimised NEON (single core).
 *
 * Builds on the inter-clause layout of #8 and removes work from the hot loop:
 *   - clauses regrouped by (class, polarity): every 16-lane group has one
 *     weight sign, so a group only needs a fire COUNT (vsubq_u8 of the
 *     0xFF masks), no per-lane weights or multiplies;
 *   - empty clauses dropped at setup (they never fire), so no nonempty
 *     tracking; partial groups padded with a never-firing rule (padding
 *     bit 527 set, which is always 0 in the input);
 *   - the 66 broadcast input vectors are built once per sample;
 *   - two groups evaluated per iteration for instruction-level parallelism;
 *   - optional early exit: skip the rest of a group pair once all 32 lanes
 *     have already failed (EARLY_EXIT, checked every EXIT_STRIDE bytes).
 *
 * Score = count(pos fired) - count(neg fired); weights are +/-1.
 */

#define IMPL_NAME "10 final optimised NEON"
#include "tm_common.h"
#include "neon_compat.h"

#ifndef EARLY_EXIT
#define EARLY_EXIT 1
#endif
#ifndef EXIT_STRIDE
#define EXIT_STRIDE 8
#endif

#define LANES      16
#define MAX_GROUPS ((CLAUSES + LANES - 1) / LANES + 2)

typedef struct { uint8_t b[RULE_BYTES][LANES]; } Group;

static Group  groups[NUM_CLASSES][2][MAX_GROUPS] __attribute__((aligned(16)));
static int    ngroups[NUM_CLASSES][2];          /* [class][0 = +1, 1 = -1], always even */

static void impl_setup(void)
{
    memset(groups, 0, sizeof(groups));

    for (int c = 0; c < NUM_CLASSES; c++) {
        int count[2] = {0, 0};
        for (int k = 0; k < CLAUSES; k++) {
            uint8_t ne = 0;
            for (int b = 0; b < RULE_BYTES; b++) ne |= rules_p[c][k][b];
            if (!ne) continue;

            int w = weights[c][k];
            if (w != 1 && w != -1) { fprintf(stderr, "ERROR: weight %d not +/-1\n", w); exit(1); }
            int p = w > 0 ? 0 : 1;
            int g = count[p] / LANES, lane = count[p] % LANES;
            for (int b = 0; b < RULE_BYTES; b++) groups[c][p][g].b[b][lane] = rules_p[c][k][b];
            count[p]++;
        }

        for (int p = 0; p < 2; p++) {
            int n = (count[p] + LANES - 1) / LANES;
            if (n & 1) n++;                       /* pairs of groups */
            for (int i = count[p]; i < n * LANES; i++)
                groups[c][p][i / LANES].b[RULE_BYTES - 1][i % LANES] = 0x80;  /* never fires */
            if (n > 255) { fprintf(stderr, "ERROR: too many groups for u8 counters\n"); exit(1); }
            ngroups[c][p] = n;
        }
    }
}

static void impl_infer_batch(int32_t scores[NS][NUM_CLASSES])
{
    uint8x16_t xv[RULE_BYTES];

    for (int s = 0; s < NS; s++) {
        for (int b = 0; b < RULE_BYTES; b++) xv[b] = vdupq_n_u8(lit_p[s][b]);

        for (int c = 0; c < NUM_CLASSES; c++) {
            int32_t cnt[2];
            for (int p = 0; p < 2; p++) {
                uint8x16_t acc = vdupq_n_u8(0);
                const Group *G = groups[c][p];

                for (int g = 0; g < ngroups[c][p]; g += 2) {
                    uint8x16_t f0 = vdupq_n_u8(0), f1 = vdupq_n_u8(0);
                    int b = 0;
#if EARLY_EXIT
                    for (; b < RULE_BYTES - EXIT_STRIDE; b += EXIT_STRIDE) {
                        for (int j = 0; j < EXIT_STRIDE; j++) {
                            f0 = vorrq_u8(f0, vbicq_u8(vld1q_u8(G[g].b[b + j]),     xv[b + j]));
                            f1 = vorrq_u8(f1, vbicq_u8(vld1q_u8(G[g + 1].b[b + j]), xv[b + j]));
                        }
                        if (vmaxvq_u8(vorrq_u8(vceqzq_u8(f0), vceqzq_u8(f1))) == 0)
                            goto next_pair;               /* all 32 lanes already failed */
                    }
#endif
                    for (; b < RULE_BYTES; b++) {
                        f0 = vorrq_u8(f0, vbicq_u8(vld1q_u8(G[g].b[b]),     xv[b]));
                        f1 = vorrq_u8(f1, vbicq_u8(vld1q_u8(G[g + 1].b[b]), xv[b]));
                    }
                    acc = vsubq_u8(acc, vceqzq_u8(f0));   /* 0xFF == -1 -> +1 per fired lane */
                    acc = vsubq_u8(acc, vceqzq_u8(f1));
#if EARLY_EXIT
                next_pair:;
#endif
                }
                cnt[p] = vaddlvq_u8(acc);
            }
            scores[s][c] = cnt[0] - cnt[1];
        }
    }
}
