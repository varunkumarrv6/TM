/*
 * Central RPi of the distributed 3-TM MedSec-25 testbed: UDP receive -> Hamming(7,4) decode -> ECC -> TM inference
 * of 50 test flows, fused over the 3 TMs, once per round of the schedule. Every round is logged to CSV.
 *
 * Per UDP packet (2-byte round prefix + datagram, see make_inputs.py), streamed as it arrives (the "stream" mode of
 * rpi_bench_infer/tm11_pipeline.c):
 *   DECODE  codewords of the 548-byte payload + formula syndrome s = s1 + 2 s2 + 4 s4 (Hamming layout only)
 *   ECC     flip bit (7 - s) of every codeword with s != 0, rebuild the payload, protected header copy == clear
 *           header, weights flagged as 4-bit normalised with the expected scale
 *   INFER   the packet's 12 rules (one class, one 4-bit weight) are one 16-lane NEON group; for each of the 50 flows
 *           fired = rule != 0 && (rule & ~literals) == 0;  sums[node][flow][class] += w4 * #fired
 * When all packets of the 3 TMs for a round have arrived: fused sum = sum over the 3 TMs (raw, no clipping),
 * prediction = argmax (first maximum wins), accuracy vs the true class, agreement with the clean reference.
 *
 * Output (directory --out, default results/):
 *   rounds.csv      one row per round: schedule fields, packets per TM, corrected codewords, header errors,
 *                   accuracy, agreement with the clean model, clean class sums reproduced exactly (clean rounds),
 *                   times: network span (first -> last packet), decode / ECC / inference busy time, tail
 *                   (last packet -> predictions) and end-to-end (first packet -> predictions)
 *   class_sums.csv  per round and flow: true class, clean prediction, prediction, and the 5 class sums of TM 0, 1, 2
 *                   and of the fused model
 *
 * build : gcc -O3 -mcpu=native -o receiver receiver.c            (Raspberry Pi 4, real NEON)
 *         gcc -O3 -march=native -msse4.1 -o receiver receiver.c   (x86 check, NEON mapped to SSE)
 * run   : ./receiver [--port 5001] [--schedule schedule.csv] [--out results] [--idle-ms 8000]
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
#include <netinet/in.h>
#include "dist3_config.h"
#include "neon_compat.h"

#define LANES       16
#define RULE_AREA   (RULES_PER_PACKET * RULE_BYTES)          /* 540 */
#define CW_PER_PKT  (PAYLOAD_BYTES * 2)                      /* 1096 */
#define DG_HAM      (HDR_BYTES + ENC_BYTES)                  /* 967 */
#define DG_RAW      (HDR_BYTES + PAYLOAD_BYTES)              /* 556 */
#define WINDOW      8                                        /* rounds that can be open at the same time */
#define MAX_ROUNDS  65536

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ------------------------------------------------------------ Hamming(7,4), as rpi_bench_infer/tm_common.h */
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

/* ------------------------------------------------------------ test flows and references */
static uint8x16_t xv[NS][RULE_BYTES];                         /* broadcast literal bytes of every flow */
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

/* ------------------------------------------------------------ schedule (for logging only) */
typedef struct { char layout[8], attack[8], nodes[32], literals[256]; int K, iter; } Sched;
static Sched *sched;
static int    n_sched;

static void load_schedule(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    sched = calloc(MAX_ROUNDS, sizeof(Sched));
    char line[1024];
    if (!fgets(line, sizeof line, f)) { fprintf(stderr, "empty schedule\n"); exit(1); }   /* header */
    while (fgets(line, sizeof line, f)) {
        line[strcspn(line, "\r\n")] = 0;
        char *p = line, *fld[7];
        for (int i = 0; i < 7; i++) fld[i] = strsep(&p, ",");
        if (!fld[6]) continue;
        int r = atoi(fld[0]);
        if (r < 0 || r >= MAX_ROUNDS) continue;
        Sched *s = &sched[r];
        snprintf(s->layout, sizeof s->layout, "%s", fld[1]);
        snprintf(s->attack, sizeof s->attack, "%s", fld[2]);
        s->K = atoi(fld[3]);
        snprintf(s->nodes, sizeof s->nodes, "%s", fld[4]);
        s->iter = atoi(fld[5]);
        snprintf(s->literals, sizeof s->literals, "%s", fld[6]);
        if (r + 1 > n_sched) n_sched = r + 1;
    }
    fclose(f);
}

/* ------------------------------------------------------------ rounds */
typedef struct {
    int     round, used;
    int     got[N_NODES];
    uint8_t seen[N_NODES][MAX_PACKETS];
    int32_t sums[N_NODES][NS][NUM_CLASSES];
    long    corrected;
    int     bad_header, layout_seen;
    double  t_first, t_last, t_decode, t_ecc, t_infer;
} Round;

static Round   win[WINDOW];
static uint8_t done_round[MAX_ROUNDS];
static int     n_done, total_packets;
static FILE   *f_rounds, *f_sums;
static long    late_packets, bad_size;

static int argmax_c(const int32_t *v)
{
    int b = 0;
    for (int c = 1; c < NUM_CLASSES; c++) if (v[c] > v[b]) b = c;
    return b;
}

static void finalize(Round *R)
{
    double t_done;
    int32_t fused[NS][NUM_CLASSES];
    int pred[NS], correct = 0, agree = 0, exact = 1, complete = 1;
    for (int n = 0; n < N_NODES; n++) if (R->got[n] != PACKETS_PER_NODE[n]) complete = 0;
    for (int s = 0; s < NS; s++) {
        for (int c = 0; c < NUM_CLASSES; c++) {
            fused[s][c] = 0;
            for (int n = 0; n < N_NODES; n++) {
                fused[s][c] += R->sums[n][s][c];
                if (R->sums[n][s][c] != sums_ref[n][s][c]) exact = 0;
            }
        }
        pred[s] = argmax_c(fused[s]);
    }
    t_done = now_s();
    for (int s = 0; s < NS; s++) { correct += pred[s] == y_act[s]; agree += pred[s] == y_ref[s]; }

    const Sched *S = &sched[R->round];
    double e2e = (t_done - R->t_first) * 1e3, net = (R->t_last - R->t_first) * 1e3, tail = (t_done - R->t_last) * 1e3;
    fprintf(f_rounds, "%d,%s,%s,%d,%s,%d,%s,%d,%d,%d,%d,%ld,%d,%.4f,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
            R->round, S->layout, S->attack, S->K, S->nodes, S->iter, S->literals, R->got[0], R->got[1], R->got[2],
            complete, R->corrected, R->bad_header, 100.0 * correct / NS, correct, agree, exact,
            net, R->t_decode * 1e3, R->t_ecc * 1e3, R->t_infer * 1e3, tail, e2e);
    for (int s = 0; s < NS; s++) {
        for (int n = 0; n <= N_NODES; n++) {
            const int32_t *v = n < N_NODES ? R->sums[n][s] : fused[s];
            fprintf(f_sums, "%d,%d,%d,%d,%d,%s", R->round, s, y_act[s], y_ref[s], pred[s],
                    n < N_NODES ? (const char *[]){"tm0", "tm1", "tm2"}[n] : "fused");
            for (int c = 0; c < NUM_CLASSES; c++) fprintf(f_sums, ",%d", v[c]);
            fputc('\n', f_sums);
        }
    }
    fflush(f_rounds); fflush(f_sums);
    printf("round %4d  %-7s %-7s K=%2d  packets %d/%d/%d%s  corrected %6ld  acc %6.2f %%  agree %2d/%d%s  "
           "decode %.2f  ECC %.2f  infer %.2f  tail %.2f ms\n",
           R->round, S->layout, S->attack, S->K, R->got[0], R->got[1], R->got[2], complete ? "" : " INCOMPLETE",
           R->corrected, 100.0 * correct / NS, agree, NS, exact ? "  (clean sums exact)" : "",
           R->t_decode * 1e3, R->t_ecc * 1e3, R->t_infer * 1e3, tail);
    fflush(stdout);
    done_round[R->round] = 1;
    n_done++;
    R->used = 0;
}

static Round *get_round(int r)
{
    Round *R = &win[r % WINDOW];
    if (R->used && R->round == r) return R;
    if (R->used) {
        if (R->round > r) return NULL;          /* a much older round: its slot was reused */
        finalize(R);                             /* an older round never completed -> log it as incomplete */
    }
    memset(R, 0, sizeof(*R));
    R->used = 1; R->round = r;
    return R;
}

/* one packet of 12 rules (one class, one 4-bit weight) = one 16-lane group, scored for all flows */
static void infer_packet(int32_t sums[NS][NUM_CLASSES], const uint8_t *pay, int cls, int n, int w4)
{
    if (w4 == 0) return;                                     /* weight 0 contributes nothing */
    uint8_t g[RULE_BYTES][LANES] __attribute__((aligned(16)));
    memset(g, 0, sizeof(g));
    for (int r = 0; r < n; r++)
        for (int b = 0; b < RULE_BYTES; b++) g[b][r] = pay[r * RULE_BYTES + b];
    uint8x16_t ne = vdupq_n_u8(0);
    for (int b = 0; b < RULE_BYTES; b++) ne = vorrq_u8(ne, vld1q_u8(g[b]));
    uint8x16_t live = vandq_u8(vmvnq_u8(vceqzq_u8(ne)), vdupq_n_u8(1));       /* 1 = non-empty rule */
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

static void process(const uint8_t *pkt, int len, double t_rx)
{
    static uint8_t cw[CW_PER_PKT], syn[CW_PER_PKT], pay[PAYLOAD_BYTES];
    if (len < PREFIX_BYTES + HDR_BYTES) { bad_size++; return; }
    int r = pkt[0] | (pkt[1] << 8);
    const uint8_t *h = pkt + PREFIX_BYTES;
    int layout = h[7] & 1, node = h[7] >> 4;
    if (len - PREFIX_BYTES != (layout ? DG_HAM : DG_RAW) || node >= N_NODES) { bad_size++; return; }
    if (done_round[r]) { late_packets++; return; }
    int idx = h[2] | (h[3] << 8);
    if (idx >= PACKETS_PER_NODE[node]) { bad_size++; return; }
    Round *R = get_round(r);
    if (!R) { late_packets++; return; }
    if (R->seen[node][idx]) return;                          /* duplicate */
    if (R->got[0] + R->got[1] + R->got[2] == 0) R->t_first = t_rx;

    double t0 = now_s();
    if (layout) detect_packet(h + HDR_BYTES, cw, syn);
    double t1 = now_s();
    if (layout) R->corrected += correct_packet(cw, syn, pay);
    else memcpy(pay, h + HDR_BYTES, PAYLOAD_BYTES);
    int scale = h[5] | (h[6] << 8);
    int ok = memcmp(pay + RULE_AREA, h, HDR_BYTES) == 0 && (h[4] & 0x80) && scale == WEIGHT_SCALE_Q88 &&
             h[0] < NUM_CLASSES && h[1] >= 1 && h[1] <= RULES_PER_PACKET;
    double t2 = now_s();
    if (ok) infer_packet(R->sums[node], pay, h[0], h[1], (int8_t)(h[4] << 4) >> 4);
    else R->bad_header++;
    double t3 = now_s();

    R->t_decode += t1 - t0; R->t_ecc += t2 - t1; R->t_infer += t3 - t2;
    R->layout_seen |= 1 << layout;
    R->seen[node][idx] = 1;
    R->got[node]++;
    R->t_last = t_rx;
    int complete = 1;
    for (int n = 0; n < N_NODES; n++) if (R->got[n] != PACKETS_PER_NODE[n]) complete = 0;
    if (complete) finalize(R);
}

int main(int argc, char **argv)
{
    int port = 5001, idle_ms = 8000;
    const char *sched_path = "schedule.csv", *out = "results";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--port") && i + 1 < argc) port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--schedule") && i + 1 < argc) sched_path = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
        else if (!strcmp(argv[i], "--idle-ms") && i + 1 < argc) idle_ms = atoi(argv[++i]);
        else { fprintf(stderr, "usage: %s [--port 5001] [--schedule schedule.csv] [--out results] [--idle-ms 8000]\n",
                       argv[0]); return 1; }
    }
    load_flows();
    load_schedule(sched_path);
    for (int n = 0; n < N_NODES; n++) total_packets += PACKETS_PER_NODE[n];
    mkdir(out, 0755);
    char path[512];
    snprintf(path, sizeof path, "%s/rounds.csv", out);
    f_rounds = fopen(path, "w");
    snprintf(path, sizeof path, "%s/class_sums.csv", out);
    f_sums = fopen(path, "w");
    if (!f_rounds || !f_sums) { fprintf(stderr, "cannot write to %s/\n", out); return 1; }
    fprintf(f_rounds, "round,layout,attack,K,nodes,iteration,literals,pkts_tm0,pkts_tm1,pkts_tm2,complete,corrected,"
                      "bad_headers,acc_pct,correct,agree_ref,clean_sums_exact,net_ms,decode_ms,ecc_ms,infer_ms,"
                      "tail_ms,e2e_ms\n");
    fprintf(f_sums, "round,flow,actual,ref_pred,pred,source,c0,c1,c2,c3,c4\n");

    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    int rcvbuf = 16 * 1024 * 1024;
    setsockopt(sock, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof rcvbuf);
    struct sockaddr_in a = {0};
    a.sin_family = AF_INET; a.sin_port = htons((uint16_t)port); a.sin_addr.s_addr = htonl(INADDR_ANY);
    if (bind(sock, (struct sockaddr *)&a, sizeof a) < 0) { perror("bind"); return 1; }
    printf("central node: UDP %d, %d scheduled rounds, %d packets per round (TM %d/%d/%d), %d test flows\n",
           port, n_sched, total_packets, PACKETS_PER_NODE[0], PACKETS_PER_NODE[1], PACKETS_PER_NODE[2], NS);
    fflush(stdout);

    static uint8_t buf[2048];
    struct pollfd pfd = {sock, POLLIN, 0};
    int started = 0;
    while (n_done < n_sched) {
        int pr = poll(&pfd, 1, started ? idle_ms : -1);
        if (pr < 0) { if (errno == EINTR) continue; perror("poll"); break; }
        if (pr == 0) { printf("no packets for %d ms -> stop\n", idle_ms); break; }
        int len = (int)recv(sock, buf, sizeof buf, 0);
        if (len <= 0) continue;
        started = 1;
        process(buf, len, now_s());
    }
    for (int w = 0; w < WINDOW; w++) if (win[w].used) finalize(&win[w]);     /* rounds that never completed */
    printf("%d rounds logged to %s/ (late packets %ld, wrong-size packets %ld)\n", n_done, out, late_packets, bad_size);
    fclose(f_rounds); fclose(f_sums);
    return 0;
}
