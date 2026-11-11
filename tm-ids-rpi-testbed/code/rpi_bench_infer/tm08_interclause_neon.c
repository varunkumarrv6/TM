/*
 * 8. Inter-clause NEON (CLAUSE PARALLELISM)  -- kernel of
 *    rpi_codes_benchmark/original/tm_interclause_neon_single.c (eval_16_clauses_neon).
 *
 * Rules transposed to [class][group][byte][16 lanes]: one 128-bit vector holds
 * the same byte of 16 clauses, so 16 clauses are checked per instruction.
 * Weights int16, fire mask FF->1, vmull_s16, horizontal add.  Single core.
 */

#define IMPL_NAME "08 inter-clause NEON"
#include "tm_common.h"
#include "neon_compat.h"

#define SIMD_CLAUSES      16
#define GROUPS_PER_CLASS  (CL_PAD / SIMD_CLAUSES)

static uint8_t rules_t[NUM_CLASSES][GROUPS_PER_CLASS][RULE_BYTES][SIMD_CLAUSES] __attribute__((aligned(16)));

static void impl_setup(void)
{
    for (int c = 0; c < NUM_CLASSES; c++)
        for (int g = 0; g < GROUPS_PER_CLASS; g++)
            for (int b = 0; b < RULE_BYTES; b++)
                for (int lane = 0; lane < SIMD_CLAUSES; lane++)
                    rules_t[c][g][b][lane] = rules_p[c][g * SIMD_CLAUSES + lane][b];
}

static inline int32_t eval_16_clauses_neon(const uint8_t *rules, const int16_t *w, const uint8_t *input)
{
    uint8x16_t failed = vdupq_n_u8(0);
    uint8x16_t nonempty = vdupq_n_u8(0);
    for (int b = 0; b < RULE_BYTES; b++) {
        uint8x16_t rule = vld1q_u8(rules + (size_t)b * SIMD_CLAUSES);
        uint8x16_t in   = vdupq_n_u8(input[b]);
        failed   = vorrq_u8(failed, vandq_u8(rule, vmvnq_u8(in)));
        nonempty = vorrq_u8(nonempty, rule);
    }
    uint8x16_t fire_mask = vandq_u8(vceqzq_u8(failed), vmvnq_u8(vceqzq_u8(nonempty)));

    uint16x8_t fire_lo = vmovl_u8(vshr_n_u8(vget_low_u8(fire_mask), 7));
    uint16x8_t fire_hi = vmovl_u8(vshr_n_u8(vget_high_u8(fire_mask), 7));
    int16x8_t w_lo = vld1q_s16(w), w_hi = vld1q_s16(w + 8);
    int16x8_t f_lo = vreinterpretq_s16_u16(fire_lo), f_hi = vreinterpretq_s16_u16(fire_hi);

    int32x4_t sum0 = vmull_s16(vget_low_s16(w_lo),  vget_low_s16(f_lo));
    int32x4_t sum1 = vmull_s16(vget_high_s16(w_lo), vget_high_s16(f_lo));
    int32x4_t sum2 = vmull_s16(vget_low_s16(w_hi),  vget_low_s16(f_hi));
    int32x4_t sum3 = vmull_s16(vget_high_s16(w_hi), vget_high_s16(f_hi));
    return vaddvq_s32(sum0) + vaddvq_s32(sum1) + vaddvq_s32(sum2) + vaddvq_s32(sum3);
}

static void impl_infer_batch(int32_t scores[NS][NUM_CLASSES])
{
    for (int s = 0; s < NS; s++)
        for (int c = 0; c < NUM_CLASSES; c++) {
            int32_t score = 0;
            for (int g = 0; g < GROUPS_PER_CLASS; g++)
                score += eval_16_clauses_neon(&rules_t[c][g][0][0], &weights[c][g * SIMD_CLAUSES], lit_p[s]);
            scores[s][c] = score;
        }
}
