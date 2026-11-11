/*
 * 7. Persistent NEON  -- sample-parallel kernel from rpi_customarchi.c.
 *
 * The inputs are transposed into persistent vectors input_vectors[byte][4]
 * (4 x 16 lanes = 64 samples) and each rule byte is broadcast against all 64
 * samples at once (clause_match_neon in rpi_customarchi.c).  The NS samples
 * are processed in chunks of 64.  Empty-clause flags are computed once at setup.
 */

#define IMPL_NAME "07 persistent NEON (sample-par)"
#include "tm_common.h"
#include "neon_compat.h"

#define CHUNK 64

static uint8x16_t input_vectors[RULE_BYTES][4];
static uint8_t    nonempty[NUM_CLASSES][CL_PAD];

static void impl_setup(void)
{
    for (int c = 0; c < NUM_CLASSES; c++)
        for (int k = 0; k < CL_PAD; k++) {
            uint8_t ne = 0;
            for (int b = 0; b < RULE_BYTES; b++) ne |= rules_p[c][k][b];
            nonempty[c][k] = ne != 0;
        }
}

static void prepare_neon_inputs(int s0, int n)
{
    uint8_t temp[RULE_BYTES][CHUNK];
    memset(temp, 0, sizeof(temp));
    for (int s = 0; s < n; s++)
        for (int b = 0; b < RULE_BYTES; b++)
            temp[b][s] = lit_p[s0 + s][b];
    for (int b = 0; b < RULE_BYTES; b++)
        for (int v = 0; v < 4; v++)
            input_vectors[b][v] = vld1q_u8(&temp[b][16 * v]);
}

static inline void clause_match_neon(const uint8_t *rule, uint8_t match[CHUNK])
{
    uint8x16_t bad0 = vdupq_n_u8(0), bad1 = vdupq_n_u8(0), bad2 = vdupq_n_u8(0), bad3 = vdupq_n_u8(0);
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
    uint8_t matches[CHUNK];
    memset(scores, 0, sizeof(int32_t) * NS * NUM_CLASSES);
    for (int s0 = 0; s0 < NS; s0 += CHUNK) {
        int n = NS - s0 < CHUNK ? NS - s0 : CHUNK;
        prepare_neon_inputs(s0, n);
        for (int c = 0; c < NUM_CLASSES; c++)
            for (int k = 0; k < ncl[c]; k++) {
                if (!nonempty[c][k]) continue;
                clause_match_neon(rules_p[c][k], matches);
                int w = weights[c][k];
                for (int s = 0; s < n; s++)
                    if (matches[s]) scores[s0 + s][c] += w;
            }
    }
}
