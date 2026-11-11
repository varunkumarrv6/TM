/*
 * 10. Final optimised NEON (single core, weighted clauses: 4-bit normalised weights).
 *
 * Builds on the inter-clause layout of #8 (16 clauses per vector) and removes
 * work from the hot loop:
 *   - empty clauses dropped at setup (they never fire), so no nonempty tracking;
 *     partial groups padded with a never-firing rule (padding literal bit 359 set,
 *     always 0 in the input) of weight 0;
 *   - weights applied with a mask instead of a multiply: fired lanes = 0xFF,
 *     widened to int16 -1, AND with the int16 weights, pairwise-accumulated into
 *     int32 (vpadalq_s16) -- no horizontal add inside the loop;
 *   - the 45 broadcast input vectors are built once per sample;
 *   - two groups evaluated per iteration for instruction-level parallelism;
 *   - early exit: skip the rest of a group pair once all 32 lanes have failed
 *     (EARLY_EXIT, checked every EXIT_STRIDE bytes).
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
#define MAX_GROUPS (CL_PAD / LANES + 2)

typedef struct { uint8_t b[RULE_BYTES][LANES]; int16_t w[LANES]; } Group;

static Group groups[NUM_CLASSES][MAX_GROUPS] __attribute__((aligned(16)));
static int   ngroups[NUM_CLASSES];                      /* always even */

static void impl_setup(void)
{
    memset(groups, 0, sizeof(groups));
    for (int c = 0; c < NUM_CLASSES; c++) {
        int n = 0;
        for (int k = 0; k < ncl[c]; k++) {
            uint8_t ne = 0;
            for (int b = 0; b < RULE_BYTES; b++) ne |= rules_p[c][k][b];
            if (!ne) continue;
            int g = n / LANES, lane = n % LANES;
            for (int b = 0; b < RULE_BYTES; b++) groups[c][g].b[b][lane] = rules_p[c][k][b];
            groups[c][g].w[lane] = weights[c][k];
            n++;
        }
        int ng = (n + LANES - 1) / LANES;
        if (ng & 1) ng++;                                 /* pairs of groups */
        for (int i = n; i < ng * LANES; i++)
            groups[c][i / LANES].b[RULE_BYTES - 1][i % LANES] = 0x80;   /* never fires, weight 0 */
        ngroups[c] = ng;
    }
}

static void impl_infer_batch(int32_t scores[NS][NUM_CLASSES])
{
    uint8x16_t xv[RULE_BYTES];

    for (int s = 0; s < NS; s++) {
        for (int b = 0; b < RULE_BYTES; b++) xv[b] = vdupq_n_u8(lit_p[s][b]);

        for (int c = 0; c < NUM_CLASSES; c++) {
            int32x4_t acc = vdupq_n_s32(0);
            const Group *G = groups[c];

            for (int g = 0; g < ngroups[c]; g += 2) {
                uint8x16_t f0 = vdupq_n_u8(0), f1 = vdupq_n_u8(0);
                int b = 0;
#if EARLY_EXIT
                for (; b < RULE_BYTES - EXIT_STRIDE; b += EXIT_STRIDE) {
                    for (int j = 0; j < EXIT_STRIDE; j++) {
                        f0 = vorrq_u8(f0, vbicq_u8(vld1q_u8(G[g].b[b + j]),     xv[b + j]));
                        f1 = vorrq_u8(f1, vbicq_u8(vld1q_u8(G[g + 1].b[b + j]), xv[b + j]));
                    }
                    if (vmaxvq_u8(vorrq_u8(vceqzq_u8(f0), vceqzq_u8(f1))) == 0)
                        goto next_pair;                   /* all 32 lanes already failed */
                }
#endif
                for (; b < RULE_BYTES; b++) {
                    f0 = vorrq_u8(f0, vbicq_u8(vld1q_u8(G[g].b[b]),     xv[b]));
                    f1 = vorrq_u8(f1, vbicq_u8(vld1q_u8(G[g + 1].b[b]), xv[b]));
                }
                {
                    int8x16_t m0 = vreinterpretq_s8_u8(vceqzq_u8(f0));   /* fired lane = -1 */
                    int8x16_t m1 = vreinterpretq_s8_u8(vceqzq_u8(f1));
                    acc = vpadalq_s16(acc, vandq_s16(vld1q_s16(G[g].w),         vmovl_s8(vget_low_s8(m0))));
                    acc = vpadalq_s16(acc, vandq_s16(vld1q_s16(G[g].w + 8),     vmovl_s8(vget_high_s8(m0))));
                    acc = vpadalq_s16(acc, vandq_s16(vld1q_s16(G[g + 1].w),     vmovl_s8(vget_low_s8(m1))));
                    acc = vpadalq_s16(acc, vandq_s16(vld1q_s16(G[g + 1].w + 8), vmovl_s8(vget_high_s8(m1))));
                }
#if EARLY_EXIT
            next_pair:;
#endif
            }
            scores[s][c] = vaddvq_s32(acc);
        }
    }
}
