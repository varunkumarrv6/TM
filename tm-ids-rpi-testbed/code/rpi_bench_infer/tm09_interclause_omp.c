/*
 * 9. Inter-clause NEON + OpenMP (CLAUSE PARALLELISM + CORE PARALLELISM)
 *
 * Same 16-clause NEON kernel as #8; the clause groups of each inference are
 * split across TM_THREADS cores (default 4 = Pi 4).  The receiver also decodes
 * the datagrams on TM_THREADS cores (TM_USE_OMP in tm_common.h).
 *
 * One parallel region per inference, so the fork/join + reduction cost is paid
 * per sample -- compare against single-core #8 / #10.
 */

#define IMPL_NAME "09 inter-clause NEON + OpenMP"
#define TM_USE_OMP
#include "tm_common.h"
#include "neon_compat.h"

#define SIMD_CLAUSES      16
#define GROUPS_PER_CLASS  (CL_PAD / SIMD_CLAUSES)
#define TOTAL_GROUPS      (NUM_CLASSES * GROUPS_PER_CLASS)

static uint8_t rules_t[TOTAL_GROUPS][RULE_BYTES][SIMD_CLAUSES] __attribute__((aligned(16)));
static int16_t w16[TOTAL_GROUPS * SIMD_CLAUSES];

static void impl_setup(void)
{
    for (int c = 0; c < NUM_CLASSES; c++)
        for (int g = 0; g < GROUPS_PER_CLASS; g++) {
            for (int b = 0; b < RULE_BYTES; b++)
                for (int lane = 0; lane < SIMD_CLAUSES; lane++)
                    rules_t[c * GROUPS_PER_CLASS + g][b][lane] = rules_p[c][g * SIMD_CLAUSES + lane][b];
            for (int lane = 0; lane < SIMD_CLAUSES; lane++)
                w16[(c * GROUPS_PER_CLASS + g) * SIMD_CLAUSES + lane] = weights[c][g * SIMD_CLAUSES + lane];
        }
    static int told = 0;
    if (!told++) fprintf(stderr, "[09] OpenMP threads: %d\n", nthreads);
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
    for (int s = 0; s < NS; s++) {
        int32_t sc[NUM_CLASSES] = {0};
        const uint8_t *x = lit_p[s];

        #pragma omp parallel for num_threads(nthreads) schedule(static) reduction(+:sc[:NUM_CLASSES])
        for (int g = 0; g < TOTAL_GROUPS; g++)
            sc[g / GROUPS_PER_CLASS] += eval_16_clauses_neon(&rules_t[g][0][0], &w16[g * SIMD_CLAUSES], x);

        memcpy(scores[s], sc, sizeof(sc));
    }
}
