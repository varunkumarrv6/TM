/*
 * Shared harness for the 10 reconstructed TM inference kernels.
 *
 * Model : rpi_tm_model/{rules,weights}.npy   (6 classes x 6000 clauses, 526 literals)
 * Packed: rules_bitpacked.npy                (6 x 6000 x 66 bytes, little bit order)
 * Input : rpi_50_test/X_50.npy               (50 x 263 boolean features)
 * Check : rpi_50_test/y_50_tm_predictions.npy (TMU predictions, must match 50/50)
 *
 * Clause semantics (same as the Python reference / rpi_customarchi.c):
 *   literal vector = [x, 1-x]                 (526 bits, packed into 66 bytes)
 *   clause fires   = rule != 0  &&  (rule & ~literals) == 0
 *   class score    = sum(weight[clause] for fired clauses)   (raw, no clipping)
 *   prediction     = argmax(score)  (first max wins, like numpy)
 *
 * Each implementation file defines IMPL_NAME, includes this header and
 * provides impl_setup() (untimed, model-side preparation) and
 * impl_infer_batch() (timed: all 50 samples -> 50 x 6 scores).
 */

#ifndef TM_COMMON_H
#define TM_COMMON_H

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#define NUM_CLASSES     6
#define CLAUSES         6000
#define TOTAL_CLAUSES   (NUM_CLASSES * CLAUSES)
#define FEATURES        263
#define LITERALS        526
#define RULE_BYTES      66
#define NS              50

static uint8_t  rules_u[NUM_CLASSES][CLAUSES][LITERALS];
static uint8_t  rules_p[NUM_CLASSES][CLAUSES][RULE_BYTES];
static int8_t   weights[NUM_CLASSES][CLAUSES];
static uint8_t  X[NS][FEATURES];
static uint8_t  lit_u[NS][LITERALS];
static uint8_t  lit_p[NS][RULE_BYTES];
static uint32_t y_tmu[NS];
static uint32_t y_true[NS];
static int32_t  ref_scores[NS][NUM_CLASSES];

#ifndef TM_NO_MAIN
static void impl_setup(void);
static void impl_infer_batch(int32_t scores[NS][NUM_CLASSES]);
#endif


/* ============================================================
   .NPY LOADER  (C-order, fixed dtype, exact size)
   ============================================================ */

/* Returns 1 if the array is stored Fortran-order (caller transposes). */
static int load_npy(const char *dir, const char *name, const char *descr,
                    size_t nbytes, void *dst)
{
    char path[4096];
    snprintf(path, sizeof(path), "%s/%s", dir, name);

    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }

    unsigned char pre[10];
    if (fread(pre, 1, 10, f) != 10 || memcmp(pre, "\x93NUMPY", 6) != 0) {
        fprintf(stderr, "ERROR: %s is not a .npy file\n", path);
        exit(1);
    }

    size_t hlen;
    long   data_off;
    if (pre[6] == 1) {
        hlen = pre[8] | (pre[9] << 8);
        data_off = 10 + (long)hlen;
    } else {
        unsigned char ext[2];
        if (fread(ext, 1, 2, f) != 2) { fprintf(stderr, "ERROR: %s header\n", path); exit(1); }
        hlen = pre[8] | (pre[9] << 8) | (ext[0] << 16) | ((size_t)ext[1] << 24);
        data_off = 12 + (long)hlen;
    }

    char *hdr = calloc(hlen + 1, 1);
    if (fread(hdr, 1, hlen, f) != hlen) { fprintf(stderr, "ERROR: %s header\n", path); exit(1); }

    int fortran = strstr(hdr, "'fortran_order': True") != NULL;
    if (!strstr(hdr, descr)) {
        fprintf(stderr, "ERROR: %s header mismatch (want %s): %s\n", path, descr, hdr);
        exit(1);
    }
    free(hdr);

    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    if ((size_t)(fsize - data_off) != nbytes) {
        fprintf(stderr, "ERROR: %s has %ld data bytes, expected %zu\n",
                path, fsize - data_off, nbytes);
        exit(1);
    }

    fseek(f, data_off, SEEK_SET);
    if (fread(dst, 1, nbytes, f) != nbytes) { fprintf(stderr, "ERROR: %s read\n", path); exit(1); }
    fclose(f);
    return fortran;
}


/* ============================================================
   DATA PREPARATION  (untimed, common to all kernels)
   ============================================================ */

static void load_all(const char *dir)
{
    load_npy(dir, "rpi_tm_model/rules.npy",   "'|u1'", sizeof(rules_u), rules_u);
    load_npy(dir, "rpi_tm_model/weights.npy", "'|i1'", sizeof(weights), weights);
    load_npy(dir, "rules_bitpacked.npy",      "'|u1'", sizeof(rules_p), rules_p);
    if (load_npy(dir, "rpi_50_test/X_50.npy", "'|u1'", sizeof(X), X)) {
        static uint8_t t[sizeof(X)];          /* stored column-major: X[f][s] */
        memcpy(t, X, sizeof(X));
        for (int s = 0; s < NS; s++)
            for (int f = 0; f < FEATURES; f++) X[s][f] = t[f * NS + s];
    }
    load_npy(dir, "rpi_50_test/y_50_tm_predictions.npy", "'<u4'", sizeof(y_tmu), y_tmu);
    load_npy(dir, "rpi_50_test/y_50_actual.npy",         "'<u4'", sizeof(y_true), y_true);

    /* Packed rules must be packbits(rules, bitorder="little"). */
    for (int c = 0; c < NUM_CLASSES; c++)
        for (int k = 0; k < CLAUSES; k++)
            for (int i = 0; i < LITERALS; i++) {
                int bit = (rules_p[c][k][i / 8] >> (i % 8)) & 1;
                if (bit != rules_u[c][k][i]) {
                    fprintf(stderr, "ERROR: rules_bitpacked.npy != packbits(rules.npy)\n");
                    exit(1);
                }
            }

    /* Literals [x, 1-x], unpacked and packed (bits 526..527 stay 0). */
    memset(lit_p, 0, sizeof(lit_p));
    for (int s = 0; s < NS; s++)
        for (int f = 0; f < FEATURES; f++) {
            int x = X[s][f] ? 1 : 0;
            lit_u[s][f]            = (uint8_t)x;
            lit_u[s][FEATURES + f] = (uint8_t)(1 - x);
            int lit = x ? f : FEATURES + f;
            lit_p[s][lit / 8] |= (uint8_t)(1u << (lit % 8));
        }
}

/* Independent plain-C reference used to check every kernel's scores. */
static void compute_reference(void)
{
    for (int s = 0; s < NS; s++)
        for (int c = 0; c < NUM_CLASSES; c++) {
            int32_t sum = 0;
            for (int k = 0; k < CLAUSES; k++) {
                int nonempty = 0, fires = 1;
                for (int i = 0; i < LITERALS; i++) {
                    if (rules_u[c][k][i]) {
                        nonempty = 1;
                        if (!lit_u[s][i]) { fires = 0; break; }
                    }
                }
                if (nonempty && fires) sum += weights[c][k];
            }
            ref_scores[s][c] = sum;
        }
}

static int argmax6(const int32_t *v)
{
    int best = 0;
    for (int c = 1; c < NUM_CLASSES; c++)
        if (v[c] > v[best]) best = c;
    return best;
}


/* ============================================================
   TIMING HARNESS
   ============================================================ */

static double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

#ifndef TM_NO_MAIN

static int32_t out_scores[NS][NUM_CLASSES];

int main(int argc, char **argv)
{
    const char *dir = argc > 1 ? argv[1] : "..";
    double budget  = argc > 2 ? atof(argv[2]) : 1.0;   /* seconds of timed runs */

    load_all(dir);
    compute_reference();
    impl_setup();

    /* Warm-up + correctness run. */
    memset(out_scores, 0, sizeof(out_scores));
    impl_infer_batch(out_scores);

    int scores_ok = 1, tmu_match = 0, correct = 0;
    for (int s = 0; s < NS; s++) {
        if (memcmp(out_scores[s], ref_scores[s], sizeof(out_scores[s])) != 0) scores_ok = 0;
        int p = argmax6(out_scores[s]);
        tmu_match += (p == (int)y_tmu[s]);
        correct   += (p == (int)y_true[s]);
    }

    /* Timed runs: at least 3, until the time budget is used (max 10000). */
    enum { MAX_REPS = 10000 };
    static double t[MAX_REPS];
    int reps = 0;
    double start = now_seconds();
    while (reps < MAX_REPS && (reps < 3 || now_seconds() - start < budget)) {
        double t0 = now_seconds();
        impl_infer_batch(out_scores);
        t[reps++] = now_seconds() - t0;
    }
    qsort(t, reps, sizeof(double), cmp_double);
    double batch_ms = t[reps / 2] * 1e3;

    printf("%-28s per-sample %10.4f ms | 50-sample batch %10.3f ms | reps %5d | "
           "scores==ref %s | TMU match %2d/50 | acc %.2f\n",
           IMPL_NAME, batch_ms / NS, batch_ms, reps,
           scores_ok ? "YES" : "NO ", tmu_match, correct / (double)NS);
    printf("RESULT|%s|%.6f|%.6f|%d|%d|%d|%.4f\n",
           IMPL_NAME, batch_ms / NS, batch_ms, reps, scores_ok, tmu_match, correct / (double)NS);

    return (scores_ok && tmu_match == NS) ? 0 : 2;
}

#endif /* TM_NO_MAIN */

#endif
