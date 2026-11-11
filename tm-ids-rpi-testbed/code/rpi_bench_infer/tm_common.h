/*
 * Shared harness for the 10 TM inference kernels -- MedSec-25 single TM (5 classes x 900 clauses,
 * 356 literals, weighted), model received as a stream of UDP-style datagrams.
 *
 * Pipeline measured by every kernel (same receiver code for all 10):
 *   1. RECEIVE (timed)  : for every datagram of the stream file
 *        clear header [class | n_real | seq lo | seq hi | norm flag (bit 7) + 4-bit weight | scale lo | scale hi |
 *                      node (bits 7..4) + layout (bit 0)]
 *        weights are 4-bit two's complement (-8..7), normalised by s = max|w| / 7 (Q8.8 in the header);
 *        the receiver checks the flag and keeps w4 (w ~ w4 * s; the common scale does not change the argmax)
 *        Hamming(7,4) decode of the 548-byte payload with the formula-based syndrome of rpi_customarchi.c
 *            s1 = P1^D1^D2^D4, s2 = P2^D1^D3^D4, s4 = P4^D2^D3^D4, flip bit (7 - syndrome)
 *        header copy == clear header, seq == packet index, then the n_real rules go to rules_p[class] with
 *        the packet's 4-bit weight (padding rules dropped)
 *   2. PREP (timed once): kernel-specific layout of the rebuilt model (transposes, padding, grouping)
 *   3. INFER (timed)    : NS test samples -> NS x 5 class sums; latency = median batch time / NS
 * Checks: model hash == MODEL_HASH (FNV-1a over rules + 4-bit weights), a plain-C reference and every kernel
 * reproduce the class sums of the 4-bit model for all NS samples exactly (scores_ref.npy, computed
 * independently in Python), predictions == y_ref.npy.
 *
 * Clause semantics: literals = [x, 1-x] (356 bits, packed LSB first into 45 bytes);
 *   clause fires = rule != 0 && (rule & ~literals) == 0;  score = sum of weights of fired clauses (no clipping);
 *   prediction = argmax (first maximum wins, like numpy).
 *
 * Each kernel file defines IMPL_NAME, includes this header and provides impl_setup() and impl_infer_batch().
 * Defining TM_USE_OMP before the include (kernel 09) also decodes the datagrams on TM_THREADS cores.
 */

#ifndef TM_COMMON_H
#define TM_COMMON_H

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#ifdef TM_USE_OMP
#include <omp.h>
#endif
#include "model_dims.h"

#define TOTAL_SLOTS   (NUM_CLASSES * CL_PAD)
#define RULE_AREA     (RULES_PER_PACKET * RULE_BYTES)        /* 540 */

static uint8_t  rules_p[NUM_CLASSES][CL_PAD][RULE_BYTES];   /* rebuilt model, packed (padding slots = 0) */
static uint8_t  rules_u[NUM_CLASSES][CL_PAD][LITERALS];     /* same, one byte per literal (kernel 01)     */
static int16_t  weights[NUM_CLASSES][CL_PAD];
static int      ncl[NUM_CLASSES];                           /* received clauses per class                 */

static uint8_t  X[NS][FEATURES];
static uint8_t  lit_u[NS][LITERALS];
static uint8_t  lit_p[NS][RULE_BYTES];
static uint32_t y_ref[NS], y_true[NS];
static int32_t  ref_expected[NS][NUM_CLASSES];             /* class sums of the 4-bit model (Python) */
static int32_t  ref_scores[NS][NUM_CLASSES];

static uint8_t *stream;           /* the datagram stream file, read once (untimed) */
static size_t   stream_len;
static int      layout;           /* 0 = raw, 1 = Hamming(7,4) */
static int      corrected;        /* Hamming codewords corrected in the last receive */
static int      normalised;       /* header flag: weights are 4-bit normalised */
static double   weight_scale;     /* s from the header (Q8.8 / 256): w ~ w4 * s */
static int      nthreads __attribute__((unused)) = 1;

#ifndef TM_NO_MAIN
static void impl_setup(void);
static void impl_infer_batch(int32_t scores[NS][NUM_CLASSES]);
#endif


/* ============================================================
   .NPY LOADER  (C-order, fixed dtype, exact size)
   ============================================================ */

static void load_npy(const char *dir, const char *name, const char *descr, size_t nbytes, void *dst)
{
    char path[4096];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    unsigned char pre[10];
    if (fread(pre, 1, 10, f) != 10 || memcmp(pre, "\x93NUMPY", 6) != 0) {
        fprintf(stderr, "ERROR: %s is not a .npy file\n", path); exit(1);
    }
    size_t hlen; long off;
    if (pre[6] == 1) { hlen = pre[8] | (pre[9] << 8); off = 10 + (long)hlen; }
    else {
        unsigned char ext[2];
        if (fread(ext, 1, 2, f) != 2) { fprintf(stderr, "ERROR: %s header\n", path); exit(1); }
        hlen = pre[8] | (pre[9] << 8) | (ext[0] << 16) | ((size_t)ext[1] << 24); off = 12 + (long)hlen;
    }
    char *hdr = calloc(hlen + 1, 1);
    if (fread(hdr, 1, hlen, f) != hlen) { fprintf(stderr, "ERROR: %s header\n", path); exit(1); }
    if (!strstr(hdr, descr) || strstr(hdr, "'fortran_order': True")) {
        fprintf(stderr, "ERROR: %s header mismatch (want %s, C order): %s\n", path, descr, hdr); exit(1);
    }
    free(hdr);
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    if ((size_t)(fsize - off) != nbytes) {
        fprintf(stderr, "ERROR: %s has %ld data bytes, expected %zu\n", path, fsize - off, nbytes); exit(1);
    }
    fseek(f, off, SEEK_SET);
    if (fread(dst, 1, nbytes, f) != nbytes) { fprintf(stderr, "ERROR: %s read\n", path); exit(1); }
    fclose(f);
}

static void load_stream(const char *dir, const char *name)
{
    char path[4096];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    stream_len = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    stream = malloc(stream_len);
    if (fread(stream, 1, stream_len, f) != stream_len) { fprintf(stderr, "ERROR: %s read\n", path); exit(1); }
    fclose(f);
    layout = stream_len ? (stream[7] & 1) : -1;
    size_t dg = HDR_BYTES + (layout == 1 ? ENC_BYTES : PAYLOAD_BYTES);
    if ((layout != 0 && layout != 1) || stream_len != (size_t)N_PACKETS * dg) {
        fprintf(stderr, "ERROR: %s: layout %d, %zu bytes, expected %d datagrams of %zu bytes\n",
                path, layout, stream_len, N_PACKETS, dg);
        exit(1);
    }
}


/* ============================================================
   HAMMING(7,4) DECODER  (formula-based syndrome, rpi_customarchi.c layout)
   codeword bits 6..0 = P1 P2 D1 P4 D2 D3 D4; 7 encoded bytes = 8 codewords = 4 decoded bytes
   ============================================================ */

static inline uint8_t calculate_syndrome(uint8_t cw)
{
    int p1 = (cw >> 6) & 1, p2 = (cw >> 5) & 1, d1 = (cw >> 4) & 1, p4 = (cw >> 3) & 1;
    int d2 = (cw >> 2) & 1, d3 = (cw >> 1) & 1, d4 = cw & 1;
    int s1 = p1 ^ d1 ^ d2 ^ d4;
    int s2 = p2 ^ d1 ^ d3 ^ d4;
    int s4 = p4 ^ d2 ^ d3 ^ d4;
    return (uint8_t)(s1 | (s2 << 1) | (s4 << 2));
}

static inline uint8_t nibble_of(uint8_t cw)                  /* D1 D2 D3 D4 -> nibble bits 3..0 */
{
    return (uint8_t)((((cw >> 4) & 1) << 3) | (((cw >> 2) & 1) << 2) | (((cw >> 1) & 1) << 1) | (cw & 1));
}

/* DECODE (detection): the 1096 codewords of one payload and their syndromes */
static void detect_packet(const uint8_t *enc, uint8_t *cw, uint8_t *syn)
{
    for (int g = 0; g < PAYLOAD_BYTES / 4; g++) {
        const uint8_t *e = enc + 7 * g;
        uint64_t v = 0;
        for (int i = 0; i < 7; i++) v = (v << 8) | e[i];              /* 56 bits, MSB first */
        for (int i = 0; i < 8; i++) {                                  /* codeword 2j = high nibble, 2j+1 = low */
            uint8_t c = (uint8_t)((v >> (49 - 7 * i)) & 0x7F);
            cw[8 * g + i] = c;
            syn[8 * g + i] = calculate_syndrome(c);
        }
    }
}

/* ECC (correction): flip the bit named by a non-zero syndrome (7 - s), rebuild the payload bytes */
static int correct_packet(const uint8_t *cw, const uint8_t *syn, uint8_t *out)
{
    int fixed = 0;
    for (int j = 0; j < PAYLOAD_BYTES; j++) {
        uint8_t hi = cw[2 * j], lo = cw[2 * j + 1];
        if (syn[2 * j])     { hi ^= (uint8_t)(1u << (7 - syn[2 * j]));     fixed++; }
        if (syn[2 * j + 1]) { lo ^= (uint8_t)(1u << (7 - syn[2 * j + 1])); fixed++; }
        out[j] = (uint8_t)((nibble_of(hi) << 4) | nibble_of(lo));
    }
    return fixed;
}


/* ============================================================
   RECEIVER  (timed): datagrams -> decoded payloads -> rebuilt clause bank
   ============================================================ */

static int     slot_of[N_PACKETS];   /* first clause slot of each packet in its class (from the clear headers) */
static uint8_t cwbuf[N_PACKETS][PAYLOAD_BYTES * 2], synbuf[N_PACKETS][PAYLOAD_BYTES * 2];
static double  detect_ms_last, correct_ms_last;   /* stage times of the last receive_model() */

static double rx_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static void receive_model(void)
{
    const size_t dg = HDR_BYTES + (layout == 1 ? ENC_BYTES : PAYLOAD_BYTES);

    /* pass 1 -- DECODE: codewords + formula-based syndromes of every datagram */
    double t0 = rx_now();
    if (layout == 1) {
#ifdef TM_USE_OMP
        #pragma omp parallel for num_threads(nthreads) schedule(static)
#endif
        for (int s = 0; s < N_PACKETS; s++)
            detect_packet(stream + (size_t)s * dg + HDR_BYTES, cwbuf[s], synbuf[s]);
    }
    double t1 = rx_now();

    /* pass 2 -- ECC: correct, rebuild payload, header / sequence checks, place the rules */
    memset(rules_p, 0, sizeof(rules_p));
    memset(weights, 0, sizeof(weights));
    int next[NUM_CLASSES] = {0};                      /* slot assignment from the clear headers */
    for (int s = 0; s < N_PACKETS; s++) {
        const uint8_t *h = stream + (size_t)s * dg;
        if (h[0] >= NUM_CLASSES || h[1] == 0 || h[1] > RULES_PER_PACKET || next[h[0]] + h[1] > CL_PAD ||
            (h[4] & 0x70) != 0) {
            fprintf(stderr, "ERROR: bad clear header in packet %d\n", s); exit(1);
        }
        slot_of[s] = next[h[0]];
        next[h[0]] += h[1];
    }
    memcpy(ncl, next, sizeof(ncl));

    int fixed = 0, bad = -1;
#ifdef TM_USE_OMP
    #pragma omp parallel for num_threads(nthreads) schedule(static) reduction(+:fixed) reduction(max:bad)
#endif
    for (int s = 0; s < N_PACKETS; s++) {
        const uint8_t *h = stream + (size_t)s * dg;
        uint8_t pay[PAYLOAD_BYTES];
        if (layout == 1) fixed += correct_packet(cwbuf[s], synbuf[s], pay);
        else memcpy(pay, h + HDR_BYTES, PAYLOAD_BYTES);

        int seq = h[2] | (h[3] << 8);
        if (memcmp(pay + RULE_AREA, h, HDR_BYTES) != 0 || seq != s || (h[7] & 1) != layout) { bad = s; continue; }

        int c = h[0];
        int16_t w = (int16_t)((int8_t)(h[4] << 4) >> 4);      /* sign-extend the 4-bit weight */
        for (int r = 0; r < h[1]; r++) {
            memcpy(rules_p[c][slot_of[s] + r], pay + r * RULE_BYTES, RULE_BYTES);
            weights[c][slot_of[s] + r] = w;
        }
    }
    if (bad >= 0) { fprintf(stderr, "ERROR: packet %d failed the header / sequence check\n", bad); exit(1); }
    double t2 = rx_now();
    detect_ms_last = (t1 - t0) * 1e3;
    correct_ms_last = (t2 - t1) * 1e3;
    corrected = fixed;
    normalised = (stream[4] & 0x80) != 0;                     /* every header carries the flag and the scale */
    weight_scale = (stream[5] | (stream[6] << 8)) / 256.0;
}

static uint64_t model_hash(void)
{
    uint64_t h = 0xcbf29ce484222325ULL;
    for (int c = 0; c < NUM_CLASSES; c++)
        for (int k = 0; k < ncl[c]; k++) {
            for (int b = 0; b < RULE_BYTES; b++) h = (h ^ rules_p[c][k][b]) * 0x100000001b3ULL;
            uint16_t w = (uint16_t)weights[c][k];
            h = (h ^ (w & 0xFF)) * 0x100000001b3ULL;
            h = (h ^ (w >> 8)) * 0x100000001b3ULL;
        }
    return h;
}


/* ============================================================
   SAMPLES, UNPACKED RULES, PLAIN-C REFERENCE  (untimed)
   ============================================================ */

static void load_samples(const char *dir)
{
    load_npy(dir, "X_test.npy",     "'|u1'", sizeof(X), X);
    load_npy(dir, "y_ref.npy",      "'<u4'", sizeof(y_ref), y_ref);
    load_npy(dir, "y_true.npy",     "'<u4'", sizeof(y_true), y_true);
    load_npy(dir, "scores_ref.npy", "'<i4'", sizeof(ref_expected), ref_expected);
    memset(lit_p, 0, sizeof(lit_p));
    for (int s = 0; s < NS; s++)
        for (int f = 0; f < FEATURES; f++) {
            int x = X[s][f] ? 1 : 0;
            lit_u[s][f] = (uint8_t)x;
            lit_u[s][FEATURES + f] = (uint8_t)(1 - x);
            int lit = x ? f : FEATURES + f;
            lit_p[s][lit / 8] |= (uint8_t)(1u << (lit % 8));
        }
}

static void unpack_rules(void)
{
    for (int c = 0; c < NUM_CLASSES; c++)
        for (int k = 0; k < CL_PAD; k++)
            for (int i = 0; i < LITERALS; i++) rules_u[c][k][i] = (rules_p[c][k][i / 8] >> (i % 8)) & 1;
}

static void compute_reference(void)
{
    for (int s = 0; s < NS; s++)
        for (int c = 0; c < NUM_CLASSES; c++) {
            int32_t sum = 0;
            for (int k = 0; k < ncl[c]; k++) {
                int nonempty = 0, fires = 1;
                for (int i = 0; i < LITERALS; i++)
                    if (rules_u[c][k][i]) { nonempty = 1; if (!lit_u[s][i]) { fires = 0; break; } }
                if (nonempty && fires) sum += weights[c][k];
            }
            ref_scores[s][c] = sum;
        }
}

static int argmax_c(const int32_t *v)
{
    int best = 0;
    for (int c = 1; c < NUM_CLASSES; c++) if (v[c] > v[best]) best = c;
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
    const char *dir   = argc > 1 ? argv[1] : "data";
    double budget     = argc > 2 ? atof(argv[2]) : 1.0;        /* seconds of timed inference runs */
    const char *sname = argc > 3 ? argv[3] : "medsec_tm_hamming.bin";
#ifdef TM_USE_OMP
    nthreads = 4;                                              /* Pi 4 cores; override with TM_THREADS */
    const char *e = getenv("TM_THREADS");
    if (e && atoi(e) > 0) nthreads = atoi(e);
#endif
    enum { MAX_REPS = 20000 };
    static double t[MAX_REPS];

    load_samples(dir);
    load_stream(dir, sname);

    /* 1. receive: Hamming decode + header checks + rebuild (median of repeated runs) */
    static double td[MAX_REPS], tc[MAX_REPS];
    int reps = 0;
    double start = now_seconds();
    while (reps < MAX_REPS && (reps < 5 || now_seconds() - start < 0.5)) {
        double t0 = now_seconds();
        receive_model();
        t[reps] = now_seconds() - t0;
        td[reps] = detect_ms_last; tc[reps] = correct_ms_last;
        reps++;
    }
    qsort(t, reps, sizeof(double), cmp_double);
    qsort(td, reps, sizeof(double), cmp_double);
    qsort(tc, reps, sizeof(double), cmp_double);
    double receive_ms = t[reps / 2] * 1e3, detect_ms = td[reps / 2], correct_ms = tc[reps / 2];
    uint64_t hsh = model_hash();
    int hash_ok = hsh == MODEL_HASH;

    unpack_rules();
    compute_reference();
    int ref_ok = memcmp(ref_scores, ref_expected, sizeof(ref_scores)) == 0;

    /* 2. kernel-specific preparation (once) */
    double t0 = now_seconds();
    impl_setup();
    double prep_ms = (now_seconds() - t0) * 1e3;

    /* 3. inference: warm-up + correctness, then timed batches */
    memset(out_scores, 0, sizeof(out_scores));
    impl_infer_batch(out_scores);
    int scores_ok = memcmp(out_scores, ref_expected, sizeof(out_scores)) == 0, tmu_match = 0, correct = 0;
    for (int s = 0; s < NS; s++) {
        int p = argmax_c(out_scores[s]);
        tmu_match += p == (int)y_ref[s];
        correct   += p == (int)y_true[s];
    }
    reps = 0;
    start = now_seconds();
    while (reps < MAX_REPS && (reps < 3 || now_seconds() - start < budget)) {
        double t1 = now_seconds();
        impl_infer_batch(out_scores);
        t[reps++] = now_seconds() - t1;
    }
    qsort(t, reps, sizeof(double), cmp_double);
    double batch_ms = t[reps / 2] * 1e3;

    /* end to end: first datagram -> predictions of all NS flows (decode + ECC + kernel layout + inference + argmax) */
    static int pred_e2e[NS];
    reps = 0;
    start = now_seconds();
    while (reps < MAX_REPS && (reps < 5 || now_seconds() - start < budget)) {
        double t1 = now_seconds();
        receive_model();
        impl_setup();
        impl_infer_batch(out_scores);
        for (int s = 0; s < NS; s++) pred_e2e[s] = argmax_c(out_scores[s]);
        t[reps++] = now_seconds() - t1;
    }
    qsort(t, reps, sizeof(double), cmp_double);
    double e2e_ms = t[reps / 2] * 1e3;
    for (int s = 0; s < NS; s++) tmu_match -= (pred_e2e[s] != (int)y_ref[s]);   /* end-to-end run must agree */

    int norm_ok = normalised && (int)(weight_scale * 256.0 + 0.5) == WEIGHT_SCALE_Q88;
    int ok = hash_ok && ref_ok && scores_ok && tmu_match == NS && norm_ok;
    printf("%-30s %dT | receive %8.3f ms = decode %7.3f + ECC %7.3f (%s, %d corrected, 4-bit weights %s s=%.4f) | "
           "prep %7.2f ms | infer %8.4f ms/sample | end-to-end %8.3f ms | hash %s | scores==ref %s | pred %d/%d | acc %.4f\n",
           IMPL_NAME, nthreads, receive_ms, detect_ms, correct_ms, layout ? "Hamming" : "raw", corrected,
           normalised ? "normalised," : "NOT normalised!", weight_scale, prep_ms, batch_ms / NS, e2e_ms,
           hash_ok ? "OK" : "BAD", scores_ok ? "YES" : "NO", tmu_match, NS, correct / (double)NS);
    ok = ok && tmu_match == NS;
    printf("RESULT|%s|%d|%s|%d|%.6f|%d|%.6f|%.6f|%.6f|%d|%d|%d|%d|%.6f|%d|%.6f|%.6f|%.6f\n",
           IMPL_NAME, nthreads, sname, layout, receive_ms, corrected, prep_ms, batch_ms / NS, batch_ms, reps,
           hash_ok, scores_ok, tmu_match, correct / (double)NS, ok, detect_ms, correct_ms, e2e_ms);
    return ok ? 0 : 2;
}

#endif /* TM_NO_MAIN */
#endif
