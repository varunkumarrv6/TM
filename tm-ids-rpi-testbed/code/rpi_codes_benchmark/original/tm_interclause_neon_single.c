#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <arm_neon.h>

#define NUM_CLASSES        6
#define CLAUSES_PER_CLASS  6000
#define TOTAL_CLAUSES      (NUM_CLASSES * CLAUSES_PER_CLASS)

#define RULE_BYTES         66
#define SIMD_CLAUSES       16

#define GROUPS_PER_CLASS   (CLAUSES_PER_CLASS / SIMD_CLAUSES)
#define TOTAL_GROUPS       (NUM_CLASSES * GROUPS_PER_CLASS)

#define ITERATIONS         1000

static uint8_t  *rules_raw;
static uint8_t  *rules_transposed;
static int16_t  *weights;

static uint8_t input_bits[RULE_BYTES];


/* =========================================================
 * Transposed index
 *
 * [class][group][byte][lane]
 * ========================================================= */

static inline size_t transposed_index(
    int class_id,
    int group_id,
    int byte_id,
    int lane)
{
    return
        (((size_t)class_id * GROUPS_PER_CLASS + group_id)
        * RULE_BYTES + byte_id)
        * SIMD_CLAUSES + lane;
}


/* =========================================================
 * Load data
 * ========================================================= */

static void load_data(void)
{
    FILE *f;

    rules_raw = malloc(
        (size_t)TOTAL_CLAUSES * RULE_BYTES
    );

    rules_transposed = malloc(
        (size_t)TOTAL_GROUPS *
        RULE_BYTES *
        SIMD_CLAUSES
    );

    weights = malloc(
        (size_t)TOTAL_CLAUSES * sizeof(int16_t)
    );

    if (!rules_raw ||
        !rules_transposed ||
        !weights)
    {
        fprintf(stderr,
                "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }


    /* ---------------- Rules ---------------- */

    f = fopen("rules_bitpacked.bin", "rb");

    if (!f)
    {
        perror("rules_bitpacked.bin");
        exit(EXIT_FAILURE);
    }

    size_t nr = fread(
        rules_raw,
        1,
        (size_t)TOTAL_CLAUSES * RULE_BYTES,
        f
    );

    fclose(f);

    if (nr !=
        (size_t)TOTAL_CLAUSES * RULE_BYTES)
    {
        fprintf(stderr,
                "Error reading rules\n");
        exit(EXIT_FAILURE);
    }

    printf("Rules loaded     : %zu bytes\n", nr);


    /* ---------------- Weights ---------------- */

    f = fopen("weights.bin", "rb");

    if (!f)
    {
        perror("weights.bin");
        exit(EXIT_FAILURE);
    }

    size_t nw = fread(
        weights,
        sizeof(int16_t),
        TOTAL_CLAUSES,
        f
    );

    fclose(f);

    if (nw != TOTAL_CLAUSES)
    {
        fprintf(stderr,
                "Error reading weights\n");
        exit(EXIT_FAILURE);
    }

    printf("Weights loaded   : %zu bytes\n",
           nw * sizeof(int16_t));


    /* ---------------- Input ---------------- */

    f = fopen("input.bin", "rb");

    if (!f)
    {
        perror("input.bin");
        exit(EXIT_FAILURE);
    }

    size_t ni = fread(
        input_bits,
        1,
        RULE_BYTES,
        f
    );

    fclose(f);

    if (ni != RULE_BYTES)
    {
        fprintf(stderr,
                "Error reading input\n");
        exit(EXIT_FAILURE);
    }

    printf("Input loaded     : %zu bytes\n", ni);
}


/* =========================================================
 * Transpose rules
 *
 * Original:
 *
 *   [clause][byte]
 *
 * New:
 *
 *   [group][byte][16 clauses]
 *
 * This allows one NEON vector to process
 * 16 clauses simultaneously.
 * ========================================================= */

static void transpose_rules(void)
{
    for (int c = 0;
         c < NUM_CLASSES;
         c++)
    {
        for (int g = 0;
             g < GROUPS_PER_CLASS;
             g++)
        {
            for (int b = 0;
                 b < RULE_BYTES;
                 b++)
            {
                for (int lane = 0;
                     lane < SIMD_CLAUSES;
                     lane++)
                {
                    int clause =
                        g * SIMD_CLAUSES + lane;

                    rules_transposed[
                        transposed_index(
                            c,
                            g,
                            b,
                            lane
                        )
                    ] =
                        rules_raw[
                            ((size_t)c *
                             CLAUSES_PER_CLASS +
                             clause)
                            * RULE_BYTES
                            + b
                        ];
                }
            }
        }
    }
}


/* =========================================================
 * FAST 16-CLAUSE NEON KERNEL
 *
 * 16 clauses are processed simultaneously.
 *
 * For every byte:
 *
 *     missing = rule & ~input
 *
 * If no missing bit exists after all 66 bytes:
 *
 *     clause fires
 *
 * ========================================================= */

static inline int32_t eval_16_clauses_neon(
    const uint8_t *rules,
    const int16_t *w,
    const uint8_t *input)
{
    uint8x16_t failed =
        vdupq_n_u8(0);

    uint8x16_t nonempty =
        vdupq_n_u8(0);


    /* -----------------------------------------------------
     * Scan 66 packed bytes
     * ----------------------------------------------------- */

    for (int b = 0;
         b < RULE_BYTES;
         b++)
    {
        /*
         * Load rule byte for
         * 16 clauses.
         */
        uint8x16_t rule =
            vld1q_u8(
                rules +
                (size_t)b * SIMD_CLAUSES
            );


        /*
         * Broadcast input byte
         * to all 16 lanes.
         */
        uint8x16_t in =
            vdupq_n_u8(input[b]);


        /*
         * Find required bits
         * that are missing.
         */
        uint8x16_t missing =
            vandq_u8(
                rule,
                vmvnq_u8(in)
            );


        /*
         * Accumulate failures.
         */
        failed =
            vorrq_u8(
                failed,
                missing
            );


        /*
         * Detect non-empty clauses.
         */
        nonempty =
            vorrq_u8(
                nonempty,
                rule
            );
    }


    /* -----------------------------------------------------
     * Determine firing clauses
     * ----------------------------------------------------- */

    uint8x16_t fire_mask =
        vceqzq_u8(failed);


    uint8x16_t nonempty_mask =
        vmvnq_u8(
            vceqzq_u8(nonempty)
        );


    /*
     * Empty clauses must not fire.
     */
    fire_mask =
        vandq_u8(
            fire_mask,
            nonempty_mask
        );


    /* -----------------------------------------------------
     * Convert FF -> 1
     * ----------------------------------------------------- */

    uint8x8_t fire_lo =
        vget_low_u8(fire_mask);

    uint8x8_t fire_hi =
        vget_high_u8(fire_mask);


    fire_lo =
        vshr_n_u8(
            fire_lo,
            7
        );

    fire_hi =
        vshr_n_u8(
            fire_hi,
            7
        );


    /* -----------------------------------------------------
     * 8-bit -> 16-bit
     * ----------------------------------------------------- */

    uint16x8_t fire_lo_u16 =
        vmovl_u8(fire_lo);

    uint16x8_t fire_hi_u16 =
        vmovl_u8(fire_hi);


    /* -----------------------------------------------------
     * Load 16 weights
     * ----------------------------------------------------- */

    int16x8_t w_lo =
        vld1q_s16(w);

    int16x8_t w_hi =
        vld1q_s16(w + 8);


    int16x8_t fire_lo_s16 =
        vreinterpretq_s16_u16(
            fire_lo_u16
        );

    int16x8_t fire_hi_s16 =
        vreinterpretq_s16_u16(
            fire_hi_u16
        );


    /* -----------------------------------------------------
     * Multiply weights × fire
     * ----------------------------------------------------- */

    int32x4_t sum0 =
        vmull_s16(
            vget_low_s16(w_lo),
            vget_low_s16(fire_lo_s16)
        );

    int32x4_t sum1 =
        vmull_s16(
            vget_high_s16(w_lo),
            vget_high_s16(fire_lo_s16)
        );

    int32x4_t sum2 =
        vmull_s16(
            vget_low_s16(w_hi),
            vget_low_s16(fire_hi_s16)
        );

    int32x4_t sum3 =
        vmull_s16(
            vget_high_s16(w_hi),
            vget_high_s16(fire_hi_s16)
        );


    /* -----------------------------------------------------
     * Horizontal reduction
     * ----------------------------------------------------- */

    return
        vaddvq_s32(sum0) +
        vaddvq_s32(sum1) +
        vaddvq_s32(sum2) +
        vaddvq_s32(sum3);
}


/* =========================================================
 * SINGLE-CORE INFERENCE
 * ========================================================= */

static void inference_single(
    int32_t scores[NUM_CLASSES])
{
    memset(
        scores,
        0,
        NUM_CLASSES * sizeof(int32_t)
    );


    for (int c = 0;
         c < NUM_CLASSES;
         c++)
    {
        const uint8_t *class_rules =
            rules_transposed +
            (size_t)c *
            GROUPS_PER_CLASS *
            RULE_BYTES *
            SIMD_CLAUSES;


        const int16_t *class_weights =
            weights +
            (size_t)c *
            CLAUSES_PER_CLASS;


        int32_t score = 0;


        for (int g = 0;
             g < GROUPS_PER_CLASS;
             g++)
        {
            const uint8_t *group_rules =
                class_rules +
                (size_t)g *
                RULE_BYTES *
                SIMD_CLAUSES;


            const int16_t *group_weights =
                class_weights +
                (size_t)g *
                SIMD_CLAUSES;


            score +=
                eval_16_clauses_neon(
                    group_rules,
                    group_weights,
                    input_bits
                );
        }


        scores[c] = score;
    }
}


/* =========================================================
 * Prediction
 * ========================================================= */

static int predict(
    const int32_t scores[NUM_CLASSES])
{
    int best = 0;

    for (int c = 1;
         c < NUM_CLASSES;
         c++)
    {
        if (scores[c] > scores[best])
            best = c;
    }

    return best;
}


/* =========================================================
 * Timer
 * ========================================================= */

static double time_ms(void)
{
    struct timespec ts;

    clock_gettime(
        CLOCK_MONOTONIC,
        &ts
    );

    return
        (double)ts.tv_sec * 1000.0 +
        (double)ts.tv_nsec / 1000000.0;
}


/* =========================================================
 * Main
 * ========================================================= */

int main(void)
{
    printf("\n");
    printf("====================================================\n");
    printf("TM INTER-CLAUSE NEON - SINGLE CORE\n");
    printf("====================================================\n");

    printf("Classes           : %d\n",
           NUM_CLASSES);

    printf("Clauses/class     : %d\n",
           CLAUSES_PER_CLASS);

    printf("Total clauses     : %d\n",
           TOTAL_CLAUSES);

    printf("Rule bytes        : %d\n",
           RULE_BYTES);

    printf("SIMD clauses      : %d\n",
           SIMD_CLAUSES);

    printf("Groups/class      : %d\n",
           GROUPS_PER_CLASS);

    printf("Total SIMD groups : %d\n",
           TOTAL_GROUPS);


    /* -----------------------------------------------------
     * Load
     * ----------------------------------------------------- */

    printf("\nLoading data...\n");

    load_data();


    /* -----------------------------------------------------
     * Transpose
     * ----------------------------------------------------- */

    printf("\nTransposing rules...\n");

    double t0 = time_ms();

    transpose_rules();

    double t1 = time_ms();

    printf("Transpose time    : %.3f ms\n",
           t1 - t0);

    printf("Transpose excluded from inference latency.\n");


    /* -----------------------------------------------------
     * Correctness
     * ----------------------------------------------------- */

    int32_t reference[NUM_CLASSES] =
    {
        -726,
         2590,
        -553,
        -856,
        -2769,
        -2759
    };


    int32_t scores[NUM_CLASSES];


    inference_single(scores);


    printf("\n====================================================\n");
    printf("CORRECTNESS TEST\n");
    printf("====================================================\n");


    printf("Scores: [");

    for (int c = 0;
         c < NUM_CLASSES;
         c++)
    {
        printf(
            "%d%s",
            scores[c],
            c == NUM_CLASSES - 1
                ? ""
                : ", "
        );
    }

    printf("]\n");


    int prediction =
        predict(scores);


    printf("Prediction : %d\n",
           prediction);


    int correct = 1;

    for (int c = 0;
         c < NUM_CLASSES;
         c++)
    {
        if (scores[c] != reference[c])
        {
            correct = 0;
            break;
        }
    }


    printf(
        "Reference  : %s\n",
        correct ? "MATCH" : "MISMATCH"
    );


    if (!correct)
    {
        printf(
            "\nERROR: correctness failed.\n"
        );

        free(rules_raw);
        free(rules_transposed);
        free(weights);

        return EXIT_FAILURE;
    }


    /* -----------------------------------------------------
     * Benchmark
     * ----------------------------------------------------- */

    printf("\n====================================================\n");
    printf("SINGLE-CORE LATENCY BENCHMARK\n");
    printf("====================================================\n");

    printf("Iterations        : %d\n",
           ITERATIONS);


    /*
     * Warm-up
     */
    for (int i = 0;
         i < 20;
         i++)
    {
        inference_single(scores);
    }


    double start =
        time_ms();


    for (int i = 0;
         i < ITERATIONS;
         i++)
    {
        inference_single(scores);
    }


    double end =
        time_ms();


    double avg =
        (end - start) /
        ITERATIONS;


    printf(
        "1-core avg        : %.6f ms\n",
        avg
    );

    printf(
        "Prediction        : %d\n",
        predict(scores)
    );

    printf(
        "Correctness       : PASS\n",
        correct
    );


    printf("\n====================================================\n");
    printf("FINAL RESULT\n");
    printf("====================================================\n");

    printf(
        "Inference latency : %.6f ms\n",
        avg
    );

    printf(
        "SIMD width        : %d clauses\n",
        SIMD_CLAUSES
    );

    printf(
        "CPU cores used    : 1\n"
    );

    printf(
        "Prediction        : %d\n",
        predict(scores)
    );


    free(rules_raw);
    free(rules_transposed);
    free(weights);

    return 0;
}
