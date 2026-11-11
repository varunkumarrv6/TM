/*
 * Central RPi of the physical 3-TM testbed (MedSec-25, 70/15/15, 4-bit weights): receives the datagrams of the
 * 3 TMs (TM VIA_LAPTOP_NODE through the laptop forwarder), Hamming-decodes and corrects them, scores the 50 test flows with NEON
 * as every packet arrives, and when a transmission is complete prints, per TM and fused:
 *     accuracy on the 50 flows (and the clean value), mean true-class sum (and clean), flows whose 5 sums are all 0
 * plus the class sums of the first --show flows. It keeps running: every new transmission (after the senders and the
 * forwarder are restarted, e.g. with another K) is a new run.
 *
 * Per packet (same pipeline as rpi_bench_infer, stream mode):
 *   DECODE  codewords + syndrome s = s1 + 2 s2 + 4 s4 (formula, rpi_customarchi.c)     [Hamming packets]
 *   ECC     flip bit (7 - s), rebuild the payload; the header used is the repaired protected copy (raw: the clear
 *           header, which must equal its copy); format from the datagram size; bad / duplicate headers dropped
 *   INFER   12 rules of one class / one 4-bit weight = one 16-lane NEON group, all 50 flows
 * Fusion: sum of the 3 TMs' raw class sums (no clipping), argmax (first maximum wins).
 * A run ends when all datagrams of the 3 TMs arrived (used or dropped), or IDLE_MS after the last packet
 * (then marked incomplete).
 * The forwarder's label packet ("TMLABEL attack=.. K=..") is stored with the run.
 *
 * Output (results/):  runs.csv  (one row per run, appended)   run_<id>_class_sums.csv  (50 flows x 3 TMs + fused)
 *
 * build:  gcc -O3 -mcpu=native -o receiver receiver.c                  (Raspberry Pi 4)
 *         gcc -O3 -march=native -msse4.1 -o receiver receiver.c        (x86 check)
 * run:    ./receiver [--show 5] [--idle-ms 3000] [--out results]
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <poll.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include "net_config.h"
#include "dist3_config.h"
#include "neon_compat.h"

#define LANES       16
#define RULE_AREA   (RULES_PER_PACKET * RULE_BYTES)
#define CW_PER_PKT  (PAYLOAD_BYTES * 2)
#define DG_HAM      (HDR_BYTES + ENC_BYTES)
#define DG_RAW      (HDR_BYTES + PAYLOAD_BYTES)

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ------------------------------------------------------------ Hamming(7,4) */
static inline uint8_t calculate_syndrome(uint8_t cw)
{
    int p1 = (cw >> 6) & 1, p2 = (cw >> 5) & 1, d1 = (cw >> 4) & 1, p4 = (cw >> 3) & 1;
    int d2 = (cw >> 2) & 1, d3 = (cw >> 1) & 1, d4 = cw & 1;
    int s1 = p1 ^ d1 ^ d2 ^ d4;
    int s2 = p2 ^ d1 ^ d3 ^ d4;
    int s4 = p4 ^ d2 ^ d3 ^ d4;
    return (uint8_t)(s1 | (s2 << 1) | (s4 << 2));
}

static inline uint8_t nibble_of(uint8_t cw)
{
    return (uint8_t)((((cw >> 4) & 1) << 3) | (((cw >> 2) & 1) << 2) | (((cw >> 1) & 1) << 1) | (cw & 1));
}

static void detect_packet(const uint8_t *enc, uint8_t *cw, uint8_t *syn)
{
    for (int g = 0; g < PAYLOAD_BYTES / 4; g++) {
        const uint8_t *e = enc + 7 * g;
        uint64_t v = 0;
        for (int i = 0; i < 7; i++) v = (v << 8) | e[i];
        for (int i = 0; i < 8; i++) {
            uint8_t c = (uint8_t)((v >> (49 - 7 * i)) & 0x7F);
            cw[8 * g + i] = c;
            syn[8 * g + i] = calculate_syndrome(c);
        }
    }
}

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

/* ------------------------------------------------------------ test flows and clean references */
static uint8x16_t xv[NS][RULE_BYTES];
static uint8_t    y_act[NS], y_ref[NS];
static int32_t    sums_ref[N_NODES][NS][NUM_CLASSES];

static void load_file(const char *path, void *dst, size_t n)
{
    FILE *f = fopen(path, "rb");
    if (!f || fread(dst, 1, n, f) != n) { fprintf(stderr, "cannot read %zu bytes from %s\n", n, path); exit(1); }
    fclose(f);
}

static void load_flows(void)
{
    static uint8_t X[NS][FEATURES];
    load_file("X_50.bin", X, sizeof(X));
    load_file("y_50_actual.bin", y_act, sizeof(y_act));
    load_file("y_50_ref.bin", y_ref, sizeof(y_ref));
    load_file("sums_ref.bin", sums_ref, sizeof(sums_ref));
    for (int s = 0; s < NS; s++) {
        uint8_t lit[RULE_BYTES] = {0};
        for (int l = 0; l < LITERALS; l++) {
            int v = l < FEATURES ? X[s][l] : 1 - X[s][l - FEATURES];
            if (v) lit[l / 8] |= (uint8_t)(1u << (l % 8));
        }
        for (int b = 0; b < RULE_BYTES; b++) xv[s][b] = vdupq_n_u8(lit[b]);
    }
}

static int argmax_c(const int32_t *v)
{
    int b = 0;
    for (int c = 1; c < NUM_CLASSES; c++) if (v[c] > v[b]) b = c;
    return b;
}

/* ------------------------------------------------------------ one run = one transmission of the 3 models */
static struct {
    int     active, got[N_NODES], layout_mask, arrived, dropped, clear_damaged, repeats;
    uint8_t seen[N_NODES][MAX_PACKETS];
    int32_t sums[N_NODES][NS][NUM_CLASSES];
    long    corrected;
    double  t_first, t_last, t_decode, t_ecc, t_infer;
    char    label[4096];
} run;
static char  pending_label[4096] = "no label (forwarder not used?)";
static int   run_id, show = 5, total_packets;
static const char *out = "results";

static void infer_packet(int32_t sums[NS][NUM_CLASSES], const uint8_t *pay, int cls, int n, int w4)
{
    if (w4 == 0) return;
    uint8_t g[RULE_BYTES][LANES] __attribute__((aligned(16)));
    memset(g, 0, sizeof(g));
    for (int r = 0; r < n; r++)
        for (int b = 0; b < RULE_BYTES; b++) g[b][r] = pay[r * RULE_BYTES + b];
    uint8x16_t ne = vdupq_n_u8(0);
    for (int b = 0; b < RULE_BYTES; b++) ne = vorrq_u8(ne, vld1q_u8(g[b]));
    uint8x16_t live = vandq_u8(vmvnq_u8(vceqzq_u8(ne)), vdupq_n_u8(1));
    for (int s = 0; s < NS; s++) {
        uint8x16_t f0 = vdupq_n_u8(0), f1 = vdupq_n_u8(0);
        int b = 0;
        for (; b + 1 < RULE_BYTES; b += 2) {
            f0 = vorrq_u8(f0, vbicq_u8(vld1q_u8(g[b]),     xv[s][b]));
            f1 = vorrq_u8(f1, vbicq_u8(vld1q_u8(g[b + 1]), xv[s][b + 1]));
        }
        if (b < RULE_BYTES) f0 = vorrq_u8(f0, vbicq_u8(vld1q_u8(g[b]), xv[s][b]));
        uint8x16_t fired = vandq_u8(vceqzq_u8(vorrq_u8(f0, f1)), live);
        sums[s][cls] += w4 * (int)vaddlvq_u8(fired);
    }
}

/* label "TMLABEL attack=include K=4 seed=1 literals=3 77 ..." -> field value (or "") */
static void label_field(const char *lab, const char *key, char *dst, size_t n)
{
    const char *p = strstr(lab, key);
    dst[0] = 0;
    if (!p) return;
    p += strlen(key);
    size_t k = strcmp(key, "literals=") ? strcspn(p, " ") : strlen(p);
    snprintf(dst, n, "%.*s", (int)k, p);
}

static void finish_run(void)
{
    double t_done;
    int32_t fused[NS][NUM_CLASSES];
    int pred[N_NODES + 1][NS], cor[N_NODES + 1] = {0}, cor_ref[N_NODES + 1] = {0}, zero[N_NODES + 1] = {0};
    double tsum[N_NODES + 1] = {0}, tsum_ref[N_NODES + 1] = {0};
    int complete = 1, agree = 0;
    if (run.arrived < total_packets) complete = 0;                 /* datagrams missing on the network */
    for (int s = 0; s < NS; s++) {
        int32_t fref[NUM_CLASSES] = {0};
        for (int c = 0; c < NUM_CLASSES; c++) {
            fused[s][c] = 0;
            for (int n = 0; n < N_NODES; n++) { fused[s][c] += run.sums[n][s][c]; fref[c] += sums_ref[n][s][c]; }
        }
        for (int n = 0; n <= N_NODES; n++) {
            const int32_t *v = n < N_NODES ? run.sums[n][s] : fused[s];
            const int32_t *r = n < N_NODES ? sums_ref[n][s] : fref;
            pred[n][s] = argmax_c(v);
            cor[n] += pred[n][s] == y_act[s];
            cor_ref[n] += argmax_c(r) == y_act[s];
            tsum[n] += v[y_act[s]];
            tsum_ref[n] += r[y_act[s]];
            int z = 1;
            for (int c = 0; c < NUM_CLASSES; c++) if (v[c]) z = 0;
            zero[n] += z;
        }
        agree += pred[N_NODES][s] == y_ref[s];
    }
    t_done = now_s();
    run_id++;

    char attack[32], K[16], seed[16], target[16], lits[4000];
    label_field(run.label, "attack=", attack, sizeof attack);
    label_field(run.label, "target=", target, sizeof target);
    label_field(run.label, "K=", K, sizeof K);
    label_field(run.label, "seed=", seed, sizeof seed);
    label_field(run.label, "literals=", lits, sizeof lits);
    const char *lay = run.layout_mask == 2 ? "hamming" : run.layout_mask == 1 ? "raw" : "mixed";
    char name[N_NODES + 1][24];
    for (int n = 0; n < N_NODES; n++)
        snprintf(name[n], sizeof name[n], "TM %d%s", n, n == VIA_LAPTOP_NODE ? " (via laptop)" : "");
    snprintf(name[N_NODES], sizeof name[N_NODES], "fused");
    double proc = (run.t_decode + run.t_ecc + run.t_infer) * 1e3;

    printf("\n=== run %d   %s   layout %s%s\n", run_id, run.label, lay, complete ? "" : "   ** INCOMPLETE **");
    printf("datagrams arrived %d/%d   used: TM0 %d/%d  TM1 %d/%d  TM2 %d/%d   dropped %d (unusable header)\n",
           run.arrived, total_packets, run.got[0], PACKETS_PER_NODE[0], run.got[1], PACKETS_PER_NODE[1], run.got[2],
           PACKETS_PER_NODE[2], run.dropped);
    if (run.repeats)
        printf("** %d repeated packets ignored: a sender was started twice in this run -- check the result **\n",
               run.repeats);
    if (run.layout_mask == 1)                                       /* raw only: nothing can be corrected */
        printf("no ECC (raw): nothing corrected   clear headers damaged %d (such packets are dropped)\n",
               run.clear_damaged);
    else
        printf("corrected codewords %ld   clear headers damaged %d (repaired from the Hamming-protected copy)%s\n",
               run.corrected, run.clear_damaged,
               run.layout_mask == 3 ? "   ** MIXED: some senders raw, some Hamming **" : "");
    printf("%-18s %10s %10s %16s %10s %14s\n", "", "accuracy", "clean", "mean true sum", "clean", "all sums = 0");
    for (int n = 0; n <= N_NODES; n++)
        printf("%-18s %9.2f%% %9.2f%% %16.2f %10.2f %11d/%d\n", name[n], 100.0 * cor[n] / NS,
               100.0 * cor_ref[n] / NS, tsum[n] / NS, tsum_ref[n] / NS, zero[n], NS);
    printf("fused prediction equal to the clean model for %d/%d flows\n", agree, NS);
    if (show > 0) {
        printf("class sums of the first %d flows (TM0 | TM1 | TM2 | fused -> prediction, true class):\n", show);
        for (int s = 0; s < show && s < NS; s++) {
            printf("  flow %2d:", s);
            for (int n = 0; n <= N_NODES; n++) {
                const int32_t *v = n < N_NODES ? run.sums[n][s] : fused[s];
                printf(" [");
                for (int c = 0; c < NUM_CLASSES; c++) printf("%5d", v[c]);
                printf(" ]");
            }
            printf(" -> %d, true %d\n", pred[N_NODES][s], y_act[s]);
        }
    }
    printf("time: decode %.2f ms, ECC %.2f ms, inference %.2f ms (processing %.2f ms); network first->last packet "
           "%.1f ms; last packet -> predictions %.3f ms\n", run.t_decode * 1e3, run.t_ecc * 1e3, run.t_infer * 1e3,
           proc, (run.t_last - run.t_first) * 1e3, (t_done - run.t_last) * 1e3);
    fflush(stdout);

    char path[512];
    snprintf(path, sizeof path, "%s/runs.csv", out);
    int fresh = access(path, F_OK) != 0;
    FILE *f = fopen(path, "a");
    if (f) {
        if (fresh)
            fprintf(f, "run,attack,K,seed,target,literals,attacked_node,layout,complete,arrived,pkts_tm0,pkts_tm1,pkts_tm2,dropped,"
                       "clear_header_damaged,corrected,"
                       "acc_tm0,acc_tm1,acc_tm2,acc_fused,clean_acc_tm0,clean_acc_tm1,clean_acc_tm2,clean_acc_fused,"
                       "true_sum_tm0,true_sum_tm1,true_sum_tm2,true_sum_fused,clean_true_sum_tm0,clean_true_sum_tm1,"
                       "clean_true_sum_tm2,clean_true_sum_fused,all_zero_tm0,all_zero_tm1,all_zero_tm2,agree_clean,"
                       "decode_ms,ecc_ms,infer_ms,processing_ms,net_ms,tail_ms\n");
        fprintf(f, "%d,%s,%s,%s,%s,%s,%d,%s,%d,%d,%d,%d,%d,%d,%d,%ld", run_id, attack, K, seed, target[0] ? target : "all",
                lits, VIA_LAPTOP_NODE, lay, complete, run.arrived, run.got[0], run.got[1], run.got[2], run.dropped,
                run.clear_damaged, run.corrected);
        for (int n = 0; n <= N_NODES; n++) fprintf(f, ",%.2f", 100.0 * cor[n] / NS);
        for (int n = 0; n <= N_NODES; n++) fprintf(f, ",%.2f", 100.0 * cor_ref[n] / NS);
        for (int n = 0; n <= N_NODES; n++) fprintf(f, ",%.3f", tsum[n] / NS);
        for (int n = 0; n <= N_NODES; n++) fprintf(f, ",%.3f", tsum_ref[n] / NS);
        for (int n = 0; n < N_NODES; n++) fprintf(f, ",%d", zero[n]);
        fprintf(f, ",%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n", agree, run.t_decode * 1e3, run.t_ecc * 1e3,
                run.t_infer * 1e3, proc, (run.t_last - run.t_first) * 1e3, (t_done - run.t_last) * 1e3);
        fclose(f);
    }
    snprintf(path, sizeof path, "%s/run_%03d_class_sums.csv", out, run_id);
    f = fopen(path, "w");
    if (f) {
        fprintf(f, "run,attack,K,flow,actual,clean_pred,source,pred,c0,c1,c2,c3,c4,clean_c0,clean_c1,clean_c2,"
                   "clean_c3,clean_c4\n");
        for (int s = 0; s < NS; s++)
            for (int n = 0; n <= N_NODES; n++) {
                const int32_t *v = n < N_NODES ? run.sums[n][s] : fused[s];
                int32_t r[NUM_CLASSES] = {0};
                for (int c = 0; c < NUM_CLASSES; c++)
                    for (int m = 0; m < N_NODES; m++) if (n == N_NODES || m == n) r[c] += sums_ref[m][s][c];
                fprintf(f, "%d,%s,%s,%d,%d,%d,%s,%d", run_id, attack, K, s, y_act[s], y_ref[s],
                        (const char *[]){"tm0", "tm1", "tm2", "fused"}[n], pred[n][s]);
                for (int c = 0; c < NUM_CLASSES; c++) fprintf(f, ",%d", v[c]);
                for (int c = 0; c < NUM_CLASSES; c++) fprintf(f, ",%d", r[c]);
                fputc('\n', f);
            }
        fclose(f);
    }
    printf("saved: %s/runs.csv (row %d), %s\n", out, run_id, path);
    printf("waiting for the next transmission ...\n");
    fflush(stdout);
    memset(&run, 0, sizeof run);
}

/*
 * One datagram. The format comes from the SIZE (967 B Hamming, 556 B raw), not from the header, because the attacker
 * may change any bit. With Hamming the header used is the protected copy inside the payload after correction (a
 * damaged clear header is repaired this way); a raw datagram has no protection, so a clear header that differs from
 * its copy makes the packet unusable. Packets whose header is still invalid, or that name a (TM, packet) already
 * received, are dropped. The run ends when all datagrams of the 3 TMs have arrived (used or dropped).
 */
static void process(const uint8_t *dg, int len, double t_rx)
{
    static uint8_t cw[CW_PER_PKT], syn[CW_PER_PKT], pay[PAYLOAD_BYTES];
    if (len > 8 && !memcmp(dg, "TMLABEL ", 8)) {                       /* label from the laptop forwarder */
        snprintf(pending_label, sizeof pending_label, "%.*s", len - 8, (const char *)dg + 8);
        if (run.active) snprintf(run.label, sizeof run.label, "%s", pending_label);
        return;
    }
    int layout = len == DG_HAM ? 1 : len == DG_RAW ? 0 : -1;
    if (layout < 0) return;
    if (!run.active) {
        run.active = 1; run.t_first = t_rx;
        snprintf(run.label, sizeof run.label, "%s", pending_label);
    }
    run.arrived++;

    double t0 = now_s();
    if (layout) detect_packet(dg + HDR_BYTES, cw, syn);
    double t1 = now_s();
    if (layout) run.corrected += correct_packet(cw, syn, pay);
    else memcpy(pay, dg + HDR_BYTES, PAYLOAD_BYTES);
    const uint8_t *h = layout ? pay + RULE_AREA : dg;                  /* header actually used */
    int clear_ok = memcmp(pay + RULE_AREA, dg, HDR_BYTES) == 0;
    if (!clear_ok) run.clear_damaged++;
    int node = h[7] >> 4, idx = h[2] | (h[3] << 8), scale = h[5] | (h[6] << 8);
    int ok = (layout || clear_ok) && node < N_NODES && (h[7] & 1) == layout && idx < PACKETS_PER_NODE[node] &&
             (h[4] & 0x80) && scale == WEIGHT_SCALE_Q88 && h[0] < NUM_CLASSES && h[1] >= 1 &&
             h[1] <= RULES_PER_PACKET;
    if (ok && run.seen[node][idx]) {                               /* same packet again (sender restarted?) */
        run.repeats++; run.arrived--;                              /* does not count towards the run */
        return;
    }
    double t2 = now_s();
    if (ok) {
        infer_packet(run.sums[node], pay, h[0], h[1], (int8_t)(h[4] << 4) >> 4);
        run.seen[node][idx] = 1;
        run.got[node]++;
    } else {
        run.dropped++;
    }
    double t3 = now_s();
    run.t_decode += t1 - t0; run.t_ecc += t2 - t1; run.t_infer += t3 - t2;
    run.layout_mask |= 1 << layout;
    run.t_last = t_rx;
    if (run.arrived >= total_packets) finish_run();
}

int main(int argc, char **argv)
{
    int idle_ms = 3000;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--show") && i + 1 < argc) show = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--idle-ms") && i + 1 < argc) idle_ms = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
        else { fprintf(stderr, "usage: %s [--show 5] [--idle-ms 3000] [--out results]\n", argv[0]); return 1; }
    }
    load_flows();
    for (int n = 0; n < N_NODES; n++) total_packets += PACKETS_PER_NODE[n];
    mkdir(out, 0755);
    char path[512];
    snprintf(path, sizeof path, "%s/runs.csv", out);
    FILE *f = fopen(path, "r");                                  /* continue the run numbering */
    if (f) { int ch, lines = 0; while ((ch = fgetc(f)) != EOF) lines += ch == '\n'; fclose(f); run_id = lines > 0 ? lines - 1 : 0; }

    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    int rcvbuf = 16 << 20;
    setsockopt(sock, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof rcvbuf);
    struct sockaddr_in a = {0};
    a.sin_family = AF_INET;
    a.sin_port = htons(CENTRAL_PORT);
    a.sin_addr.s_addr = htonl(INADDR_ANY);                       /* all interfaces (CENTRAL_IP is one of them) */
    if (bind(sock, (struct sockaddr *)&a, sizeof a) < 0) { perror("bind"); return 1; }
    printf("central RPi %s: listening on UDP %d; expecting %d + %d + %d datagrams per transmission, %d test flows\n",
           CENTRAL_IP, CENTRAL_PORT, PACKETS_PER_NODE[0], PACKETS_PER_NODE[1], PACKETS_PER_NODE[2], NS);
    printf("waiting for the first transmission ... (Ctrl-C to quit)\n");
    fflush(stdout);

    static uint8_t buf[2048];
    struct pollfd pfd = {sock, POLLIN, 0};
    for (;;) {
        int pr = poll(&pfd, 1, run.active ? idle_ms : -1);
        if (pr < 0) { if (errno == EINTR) continue; perror("poll"); break; }
        if (pr == 0) { finish_run(); continue; }                 /* senders stopped before everything arrived */
        int len = (int)recv(sock, buf, sizeof buf, 0);
        if (len > 0) process(buf, len, now_s());
    }
    return 0;
}
