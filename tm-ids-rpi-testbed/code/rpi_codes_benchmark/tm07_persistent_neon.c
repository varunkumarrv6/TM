/*
 * 7. Persistent NEON  -- sample-parallel kernel from rpi_customarchi.c.
 *
 * The inputs are transposed once per batch into persistent vectors
 *     input_vectors[byte][4]   (4 x 16 lanes = 64 sample slots, 50 used)
 * and each rule byte is broadcast against all samples at once, i.e. one rule
 * is checked against 50 inputs per pass (clause_match_neon in
 * rpi_customarchi.c).  Empty-clause flags are computed once at setup instead
 * of per call.
 */

#define IMPL_NAME "07 persistent NEON (sample-par)"
#include "tm_common.h"
#include "neon_compat.h"

static uint8x16_t input_vectors[RULE_BYTES][4];
static uint8_t    nonempty[NUM_CLASSES][CLAUSES];

static void impl_setup(void)
{
    for (int c = 0; c < NUM_CLASSES; c++)
        for (int k = 0; k < CLAUSES; k++) {
            uint8_t ne = 0;
            for (int b = 0; b < RULE_BYTES; b++) ne |= rules_p[c][k][b];
            nonempty[c][k] = ne != 0;
        }
}

static void prepare_neon_inputs(void)
{
    uint8_t temp[RULE_BYTES][64];
    memset(temp, 0, sizeof(temp));
    for (int s = 0; s < NS; s++)
        for (int b = 0; b < RULE_BYTES; b++)
            temp[b][s] = lit_p[s][b];

    for (int b = 0; b < RULE_BYTES; b++)
        for (int v = 0; v < 4; v++)
            input_vectors[b][v] = vld1q_u8(&temp[b][16 * v]);
}

static inline void clause_match_neon(const uint8_t *rule, uint8_t match[64])
{
    uint8x16_t bad0 = vdupq_n_u8(0), bad1 = vdupq_n_u8(0),
               bad2 = vdupq_n_u8(0), bad3 = vdupq_n_u8(0);

    for (int b = 0; b < RULE_BYTES; b++) {
        uint8x16_t rule_vec = vdupq_n_u8(rule[b]);
        bad0 = vorrq_u8(bad0, vandq_u8(rule_vec, vmvnq_u8(input_vectors[b][0])));
        bad1 = vorrq_u8(bad1, vandq_u8(rule_vec, vmvnq_u8(input_vectors[b][1])));
        bad2 = vorrq_u8(bad2, vandq_u8(rule_vec, vmvnq_u8(input_vectors[b][2])));
        bad3 = vorrq_u8(bad3, vandq_u8(rule_vec, vmvnq_u8(input_vectors[b][3])));
    }

    uint8x16_t zero = vdupq_n_u8(0);
    vst1q_u8(&match[0],  vceqq_u8(bad0, zero));
    vst1q_u8(&match[16], vceqq_u8(bad1, zero));
    vst1q_u8(&match[32], vceqq_u8(bad2, zero));
    vst1q_u8(&match[48], vceqq_u8(bad3, zero));
}

static void impl_infer_batch(int32_t scores[NS][NUM_CLASSES])
{
    uint8_t matches[64];

    prepare_neon_inputs();
    memset(scores, 0, sizeof(int32_t) * NS * NUM_CLASSES);

    for (int c = 0; c < NUM_CLASSES; c++)
        for (int k = 0; k < CLAUSES; k++) {
            if (!nonempty[c][k]) continue;
            clause_match_neon(rules_p[c][k], matches);
            int w = weights[c][k];
            for (int s = 0; s < NS; s++)
                if (matches[s]) scores[s][c] += w;
        }
}
