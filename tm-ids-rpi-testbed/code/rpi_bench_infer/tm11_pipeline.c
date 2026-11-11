/*
 * 11. Pipelined receive + inference (end-to-end latency).
 *
 * Every datagram goes through three stages (as rpi_customarchi.c):
 *   A  DETECT  : extract the 1096 Hamming(7,4) codewords, formula-based syndrome of each
 *   B  CORRECT : flip the bit named by a non-zero syndrome (7 - s), rebuild the payload bytes,
 *                check header copy / sequence / normalisation flag, read class, n_real, 4-bit weight
 *   C  INFER   : the packet's 12 rules (one class, one weight) form exactly one 16-lane NEON group;
 *                every test sample is checked against all 12 rules at once and
 *                sums[sample][class] += w4 * (number of fired rules)
 * A packet can be scored as soon as it is corrected, so inference does not wait for the whole model.
 *
 * Modes (all produce the same class sums; latency = first packet available -> sums of all NS samples):
 *   seq    : A for all packets, then B for all, then C for all            (1 core, no overlap)
 *   stream : A -> B -> C for packet n, then packet n+1                    (1 core)
 *   swp    : software pipeline, one loop iteration does A(n+2), B(n+1), C(n) on three independent
 *            packets, so the out-of-order core always has independent work (1 core)
 *   pipe   : three persistent threads pinned to cores 1, 2, 3 (A / B / C), ring of RING packet buffers,
 *            lock-free progress counters: while C scores packet n, B corrects n+1 and A decodes n+2
 *   hyb2ph : hybrid, two phases: A then B for all packets with OpenMP on TM_THREADS cores (default 4),
 *            then C for all packets on one core
 *   hybpipe: hybrid pipeline: HYB_WORKERS (3) worker threads on cores 1..3 each do A + B for every 3rd packet,
 *            the main thread does C for each packet as soon as it is corrected (4 cores)
 * Worker threads of pipe / hybpipe sleep while idle and are woken (outside the timed region) before each run.
 *
 * usage: ./bin/tm11_pipeline [data dir=data] [seconds per mode=1.0] [stream=medsec_tm_hamming.bin]
 */

#define TM_NO_MAIN
#define IMPL_NAME "11 pipeline"
#include "tm_common.h"
#include "neon_compat.h"
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <omp.h>

#define CW_PER_PKT (PAYLOAD_BYTES * 2)                 /* 1096 codewords per datagram */
#define LANES      16
#define RING       16

#if defined(__aarch64__) || defined(__ARM_NEON)
#define CPU_RELAX() __asm__ __volatile__("yield")
#else
#define CPU_RELAX() __builtin_ia32_pause()
#endif

typedef struct {
    const uint8_t *dg;                                 /* the datagram inside the stream */
    int     seq;
    uint8_t cw[CW_PER_PKT], syn[CW_PER_PKT];           /* stage A output */
    uint8_t pay[PAYLOAD_BYTES];                        /* stage B output */
    int     cls, n, w4, fixed, bad;
} Slot;

static size_t     DG;
static Slot      *all_slots;                           /* seq mode: one slot per packet */
static Slot       ring[RING] __attribute__((aligned(64)));
static uint8x16_t xv[NS][RULE_BYTES];                  /* broadcast input bytes, prepared once */
static int32_t    sums[NS][NUM_CLASSES];


/* ------------------------------------------------------------ stages */

static inline void stage_detect(Slot *sl)                /* DECODE: shared detect_packet() of tm_common.h */
{
    if (layout == 1) detect_packet(sl->dg + HDR_BYTES, sl->cw, sl->syn);
}

static inline void stage_correct(Slot *sl)               /* ECC: shared correct_packet() + header checks */
{
    const uint8_t *h = sl->dg;
    if (layout == 1) sl->fixed = correct_packet(sl->cw, sl->syn, sl->pay);
    else { memcpy(sl->pay, h + HDR_BYTES, PAYLOAD_BYTES); sl->fixed = 0; }
    int seq = h[2] | (h[3] << 8);
    sl->bad = memcmp(sl->pay + RULE_AREA, h, HDR_BYTES) != 0 || seq != sl->seq || (h[7] & 1) != layout ||
              !(h[4] & 0x80) || h[0] >= NUM_CLASSES || h[1] == 0 || h[1] > RULES_PER_PACKET;
    sl->cls = h[0];
    sl->n = h[1];
    sl->w4 = (int8_t)(h[4] << 4) >> 4;                  /* sign-extend the 4-bit weight */
}

static inline void stage_infer(const Slot *sl)
{
    if (sl->bad || sl->w4 == 0) return;                 /* weight 0 contributes nothing */
    uint8_t g[RULE_BYTES][LANES] __attribute__((aligned(16)));
    memset(g, 0, sizeof(g));
    for (int r = 0; r < sl->n; r++)
        for (int b = 0; b < RULE_BYTES; b++) g[b][r] = sl->pay[r * RULE_BYTES + b];

    uint8x16_t ne = vdupq_n_u8(0);
    for (int b = 0; b < RULE_BYTES; b++) ne = vorrq_u8(ne, vld1q_u8(g[b]));
    uint8x16_t live = vandq_u8(vmvnq_u8(vceqzq_u8(ne)), vdupq_n_u8(1));   /* 1 = non-empty rule lane */

    const int c = sl->cls, w = sl->w4;
    for (int s = 0; s < NS; s++) {
        uint8x16_t f0 = vdupq_n_u8(0), f1 = vdupq_n_u8(0);                  /* two chains for ILP */
        int b = 0;
        for (; b + 1 < RULE_BYTES; b += 2) {
            f0 = vorrq_u8(f0, vbicq_u8(vld1q_u8(g[b]),     xv[s][b]));
            f1 = vorrq_u8(f1, vbicq_u8(vld1q_u8(g[b + 1]), xv[s][b + 1]));
        }
        if (b < RULE_BYTES) f0 = vorrq_u8(f0, vbicq_u8(vld1q_u8(g[b]), xv[s][b]));
        uint8x16_t fired = vandq_u8(vceqzq_u8(vorrq_u8(f0, f1)), live);
        sums[s][c] += w * (int)vaddlvq_u8(fired);
    }
}


/* ------------------------------------------------------------ stage busy-time instrumentation */
/* Only in separate instrumented runs (instr = 1): end-to-end times come from runs without timers.
 * busy[k] = time spent inside stage k (0 decode, 1 ECC, 2 infer), excluding waits; one writer per entry. */
static int    instr;
static double busy[3];
#define STAGE(k, call) do { if (instr) { double t_ = now_seconds(); call; busy[k] += now_seconds() - t_; } \
                            else { call; } } while (0)

/* ------------------------------------------------------------ single-core modes */

static void slot_init(Slot *sl, int s) { sl->dg = stream + (size_t)s * DG; sl->seq = s; }

static void run_seq(void)
{
    for (int s = 0; s < N_PACKETS; s++) { slot_init(&all_slots[s], s); STAGE(0, stage_detect(&all_slots[s])); }
    for (int s = 0; s < N_PACKETS; s++) STAGE(1, stage_correct(&all_slots[s]));
    for (int s = 0; s < N_PACKETS; s++) STAGE(2, stage_infer(&all_slots[s]));
}

static void run_stream(void)
{
    for (int s = 0; s < N_PACKETS; s++) {
        Slot *sl = &ring[s % RING];
        slot_init(sl, s);
        STAGE(0, stage_detect(sl)); STAGE(1, stage_correct(sl)); STAGE(2, stage_infer(sl));
    }
}

static void run_swp(void)
{
    for (int i = 0; i < N_PACKETS + 2; i++) {
        if (i < N_PACKETS)              { Slot *a = &ring[i % RING]; slot_init(a, i); STAGE(0, stage_detect(a)); }
        if (i >= 1 && i - 1 < N_PACKETS) STAGE(1, stage_correct(&ring[(i - 1) % RING]));
        if (i >= 2)                      STAGE(2, stage_infer(&ring[(i - 2) % RING]));
    }
}


/* ------------------------------------------------------------ 3-thread pipeline */

static atomic_int doneA, doneB, doneC, quit;

/* A group of persistent worker threads.  Idle threads SLEEP on a condition variable (no busy-waiting that would
 * steal cores from other modes).  group_arm() wakes them before a timed run and waits until all are spinning on
 * `go` (outside the timed region); group_go() starts the run. */
typedef struct {
    pthread_mutex_t mx;
    pthread_cond_t  cv;
    int             arm_gen, n;
    atomic_int      ready, go;
} Group;
static Group grp_pipe = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0, 3, 0, 0};
static Group grp_hyb  = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0, 3, 0, 0};

static void group_arm(Group *g)
{
    atomic_store(&g->ready, 0);
    pthread_mutex_lock(&g->mx); g->arm_gen++; pthread_cond_broadcast(&g->cv); pthread_mutex_unlock(&g->mx);
    while (atomic_load_explicit(&g->ready, memory_order_acquire) < g->n) CPU_RELAX();
}

static void group_go(Group *g) { atomic_fetch_add_explicit(&g->go, 1, memory_order_release); }

static int group_wait(Group *g, int *seen_arm)      /* worker side: sleep until armed, then spin until go */
{
    pthread_mutex_lock(&g->mx);
    while (g->arm_gen == *seen_arm && !atomic_load(&quit)) pthread_cond_wait(&g->cv, &g->mx);
    *seen_arm = g->arm_gen;
    pthread_mutex_unlock(&g->mx);
    if (atomic_load(&quit)) return 0;
    int go0 = atomic_load_explicit(&g->go, memory_order_acquire);
    atomic_fetch_add_explicit(&g->ready, 1, memory_order_release);
    while (atomic_load_explicit(&g->go, memory_order_acquire) == go0) CPU_RELAX();
    return 1;
}

static void group_quit(Group *g)
{
    pthread_mutex_lock(&g->mx); pthread_cond_broadcast(&g->cv); pthread_mutex_unlock(&g->mx);
}

static void pin_to_core(int core)
{
#ifdef __linux__
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);   /* best effort */
#else
    (void)core;
#endif
}

static void *stage_thread(void *arg)
{
    int stage = (int)(intptr_t)arg;                    /* 0 = A, 1 = B, 2 = C */
    pin_to_core(stage + 1);
    int seen_arm = 0;
    while (group_wait(&grp_pipe, &seen_arm)) {
        for (int i = 0; i < N_PACKETS; i++) {
            Slot *sl = &ring[i % RING];
            if (stage == 0) {
                while (i - atomic_load_explicit(&doneC, memory_order_acquire) >= RING) CPU_RELAX();
                slot_init(sl, i); STAGE(0, stage_detect(sl));
                atomic_store_explicit(&doneA, i + 1, memory_order_release);
            } else if (stage == 1) {
                while (atomic_load_explicit(&doneA, memory_order_acquire) <= i) CPU_RELAX();
                STAGE(1, stage_correct(sl));
                atomic_store_explicit(&doneB, i + 1, memory_order_release);
            } else {
                while (atomic_load_explicit(&doneB, memory_order_acquire) <= i) CPU_RELAX();
                STAGE(2, stage_infer(sl));
                atomic_store_explicit(&doneC, i + 1, memory_order_release);
            }
        }
    }
    return NULL;
}

static int pipe_fixed, pipe_bad;

static void run_pipe(void)
{
    atomic_store(&doneA, 0); atomic_store(&doneB, 0); atomic_store(&doneC, 0);
    group_go(&grp_pipe);
    while (atomic_load_explicit(&doneC, memory_order_acquire) < N_PACKETS) CPU_RELAX();
}


/* ------------------------------------------------------------ hybrid modes */

static int omp_threads = 4;                         /* hyb2ph: decode + ECC cores (TM_THREADS) */

static void run_hyb2ph(void)
{
    double t0 = instr ? now_seconds() : 0;
    #pragma omp parallel for num_threads(omp_threads) schedule(static)
    for (int s = 0; s < N_PACKETS; s++) { slot_init(&all_slots[s], s); stage_detect(&all_slots[s]); }
    double t1 = instr ? now_seconds() : 0;
    #pragma omp parallel for num_threads(omp_threads) schedule(static)
    for (int s = 0; s < N_PACKETS; s++) stage_correct(&all_slots[s]);
    double t2 = instr ? now_seconds() : 0;
    for (int s = 0; s < N_PACKETS; s++) stage_infer(&all_slots[s]);        /* one core */
    if (instr) { busy[0] += t1 - t0; busy[1] += t2 - t1; busy[2] += now_seconds() - t2; }
}

#define HYB_WORKERS 3
static Slot       hring[RING] __attribute__((aligned(64)));
static atomic_int hyb_done, hyb_ready[RING];  /* ready[slot] = packet index + 1 once corrected */
static double     wbusy[HYB_WORKERS][2];               /* per-worker decode / ECC busy time */

static void *hyb_worker(void *arg)
{
    int k = (int)(intptr_t)arg;
    pin_to_core(k + 1);
    int seen_arm = 0;
    while (group_wait(&grp_hyb, &seen_arm)) {
        wbusy[k][0] = wbusy[k][1] = 0;
        for (int i = k; i < N_PACKETS; i += HYB_WORKERS) {
            while (i - atomic_load_explicit(&hyb_done, memory_order_acquire) >= RING) CPU_RELAX();   /* slot free */
            Slot *sl = &hring[i % RING];
            slot_init(sl, i);
            double t0 = instr ? now_seconds() : 0;
            stage_detect(sl);
            double t1 = instr ? now_seconds() : 0;
            stage_correct(sl);
            if (instr) { wbusy[k][0] += t1 - t0; wbusy[k][1] += now_seconds() - t1; }
            atomic_store_explicit(&hyb_ready[i % RING], i + 1, memory_order_release);
        }
    }
    return NULL;
}

static void run_hybpipe(void)
{
    atomic_store(&hyb_done, 0);
    for (int r = 0; r < RING; r++) atomic_store(&hyb_ready[r], 0);
    group_go(&grp_hyb);
    for (int i = 0; i < N_PACKETS; i++) {                  /* inference on the main thread (core 0), in order */
        while (atomic_load_explicit(&hyb_ready[i % RING], memory_order_acquire) != i + 1) CPU_RELAX();
        STAGE(2, stage_infer(&hring[i % RING]));
        atomic_store_explicit(&hyb_done, i + 1, memory_order_release);
    }
    if (instr)                                             /* workers finished before the last packet was ready */
        for (int k = 0; k < HYB_WORKERS; k++) { busy[0] += wbusy[k][0]; busy[1] += wbusy[k][1]; }
}


/* ------------------------------------------------------------ driver */

typedef void (*mode_fn)(void);

static int count_fixed_bad(const Slot *slots, int n, int *bad)
{
    int f = 0; *bad = 0;
    for (int i = 0; i < n; i++) { f += slots[i].fixed; *bad |= slots[i].bad; }
    return f;
}

int main(int argc, char **argv)
{
    const char *dir   = argc > 1 ? argv[1] : "data";
    double budget     = argc > 2 ? atof(argv[2]) : 1.0;
    const char *sname = argc > 3 ? argv[3] : "medsec_tm_hamming.bin";

    load_samples(dir);
    load_stream(dir, sname);
    DG = HDR_BYTES + (layout == 1 ? ENC_BYTES : PAYLOAD_BYTES);
    all_slots = aligned_alloc(64, sizeof(Slot) * N_PACKETS);
    for (int s = 0; s < NS; s++)
        for (int b = 0; b < RULE_BYTES; b++) xv[s][b] = vdupq_n_u8(lit_p[s][b]);

    const char *e = getenv("TM_THREADS");
    if (e && atoi(e) > 0) omp_threads = atoi(e);
    pthread_t th[3], hw[HYB_WORKERS];
    for (int k = 0; k < 3; k++) pthread_create(&th[k], NULL, stage_thread, (void *)(intptr_t)k);
    for (int k = 0; k < HYB_WORKERS; k++) pthread_create(&hw[k], NULL, hyb_worker, (void *)(intptr_t)k);

    enum { NMODES = 6 };
    const char *names[NMODES] = {"seq", "stream", "swp", "pipe", "hyb2ph", "hybpipe"};
    const int cores[NMODES] = {1, 1, 1, 3, omp_threads, HYB_WORKERS + 1};
    mode_fn fns[NMODES] = {run_seq, run_stream, run_swp, run_pipe, run_hyb2ph, run_hybpipe};
    Group *grp_of[NMODES] = {NULL, NULL, NULL, &grp_pipe, NULL, &grp_hyb};   /* threads to wake before a run */
    enum { MAX_REPS = 20000 };
    static double t[MAX_REPS];
    double base = 0;
    int all_ok = 1, fixed_seq = -1;

    printf("== stream %s (%s), %d datagrams, %d test samples\n", sname, layout ? "Hamming" : "raw", N_PACKETS, NS);
    for (int m = 0; m < NMODES; m++) {
        int reps = 0;
        double start = now_seconds();
        static int pred[NS];
        while (reps < MAX_REPS && (reps < 5 || now_seconds() - start < budget)) {
            memset(sums, 0, sizeof(sums));
            if (grp_of[m]) group_arm(grp_of[m]);                         /* untimed: wake the workers */
            double t0 = now_seconds();
            fns[m]();
            for (int s = 0; s < NS; s++) pred[s] = argmax_c(sums[s]);     /* predictions of all flows */
            t[reps++] = now_seconds() - t0;
        }
        qsort(t, reps, sizeof(double), cmp_double);
        double ms = t[reps / 2] * 1e3;
        if (m == 0) base = ms;

        /* stage breakdown: median busy time per stage over instrumented runs */
        enum { IREPS = 15 };
        double bz[3][IREPS];
        instr = 1;
        for (int r = 0; r < IREPS; r++) {
            busy[0] = busy[1] = busy[2] = 0;
            memset(sums, 0, sizeof(sums));
            if (grp_of[m]) group_arm(grp_of[m]);
            fns[m]();
            for (int k = 0; k < 3; k++) bz[k][r] = busy[k];
        }
        instr = 0;
        double stage_ms[3];
        for (int k = 0; k < 3; k++) { qsort(bz[k], IREPS, sizeof(double), cmp_double); stage_ms[k] = bz[k][IREPS / 2] * 1e3; }

        /* correctness of the last run */
        int bad = 0, fixed;
        if (m == 0 || m == 4) fixed = count_fixed_bad(all_slots, N_PACKETS, &bad);   /* seq, hyb2ph keep all slots */
        else {
            /* re-run once with per-packet bookkeeping for the corrected count */
            memset(sums, 0, sizeof(sums));
            fixed = 0;
            if (m == 3 || m == 5) { group_arm(grp_of[m]); fns[m](); fixed = -1; }   /* no per-packet record */
            else {
                for (int s = 0; s < N_PACKETS; s++) {
                    Slot *sl = &ring[s % RING];
                    slot_init(sl, s); stage_detect(sl); stage_correct(sl); stage_infer(sl);
                    fixed += sl->fixed; bad |= sl->bad;
                }
            }
        }
        if (m == 0) fixed_seq = fixed;
        int scores_ok = memcmp(sums, ref_expected, sizeof(sums)) == 0, npred = 0;
        for (int s = 0; s < NS; s++) npred += argmax_c(sums[s]) == (int)y_ref[s] && pred[s] == (int)y_ref[s];
        int ok = scores_ok && npred == NS && !bad && (fixed < 0 || fixed == fixed_seq);
        all_ok &= ok;
        printf("%-7s %dcore%s end-to-end %9.4f ms  (%.2fx vs seq) | busy: decode %7.3f  ECC %7.3f  infer %7.3f ms | "
               "corrected %5d  scores==ref %s  pred %d/%d\n",
               names[m], cores[m], cores[m] > 1 ? "s" : " ", ms, base / ms, stage_ms[0], stage_ms[1], stage_ms[2], fixed < 0 ? fixed_seq : fixed,
               scores_ok ? "YES" : "NO", npred, NS);
        printf("RESULT|%s|%d|%s|%d|%.6f|%.4f|%d|%d|%d|%d|%.6f|%.6f|%.6f\n", names[m], cores[m], sname, layout, ms,
               base / ms, reps, fixed < 0 ? fixed_seq : fixed, scores_ok, ok, stage_ms[0], stage_ms[1], stage_ms[2]);
    }
    atomic_store(&quit, 1);
    group_quit(&grp_pipe); group_quit(&grp_hyb);
    for (int k = 0; k < 3; k++) pthread_join(th[k], NULL);
    for (int k = 0; k < HYB_WORKERS; k++) pthread_join(hw[k], NULL);
    (void)pipe_fixed; (void)pipe_bad; (void)receive_model; (void)model_hash;
    return all_ok ? 0 : 2;
}
