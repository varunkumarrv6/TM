/* ============================================================
   CENTRAL NODE: UDP receive -> Hamming(7,4) decode -> TM inference
   MedSec-25, 3 weighted TMs, 4-bit weight nibbles
   Packet format = make_rpi_files.py / medsciot ecc_pipeline.py
   (derived from iot_dataset/rpi_customarchi.c)

   build (RPi, NEON):  gcc -O3 -march=native -o receiver rpi_medsciot_receiver.c
   build (x86 test):   gcc -O3 -o receiver rpi_medsciot_receiver.c
   run (UDP)       :   ./receiver                       (listens on 6000 direct + 5001 laptop)
   run (files)     :   ./receiver --files tm0_hamming.bin tm1_hamming.bin tm2_hamming.bin
   options         :   --raw  (548-byte datagrams, no ECC)   --dir <path to X_50.bin etc>   --idle-ms 3000
   ============================================================ */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <sys/resource.h>
#include <time.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/select.h>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#include "rpi_medsciot_config.h"

#define DIRECT_PORT   6000
#define LAPTOP_PORT   5001
#define RULE_AREA     (RULES_PER_PACKET * RULE_BYTES)        /* 540 */
#define MAX_PACKETS   2048

/* ------------------------------------------------------------ globals */
static uint8_t input_raw[NUM_INPUTS][INPUT_BYTES];
static uint8_t input_packed[NUM_INPUTS][RULE_BYTES];       /* 356 literals, LSB-first */
static uint8_t actual[NUM_INPUTS], tmu_pred[NUM_INPUTS];
static char class_names[NUM_CLASSES][64];
static int32_t scores[NUM_NODES][NUM_INPUTS][NUM_CLASSES];  /* per-TM class sums */
static uint8_t seen[NUM_NODES][MAX_PACKETS];

static long n_rx = 0, n_dup = 0, n_bad = 0, n_hdr_mismatch[NUM_NODES], n_pk[NUM_NODES], n_corr[NUM_NODES];
static double t_decode = 0, t_infer = 0;
static int raw_mode = 0;

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

/* ------------------------------------------------------------ RAM self-sampler */
#define RAM_MAX 60000
static double  ram_t[RAM_MAX];                 /* seconds since program start */
static float   ram_mb[RAM_MAX];                /* resident RAM (VmRSS) in MB   */
static int     ram_n = 0;
static double  ram_last = -1.0, ram_interval = 0.050, prog_start = 0.0;   /* default 50 ms */
static const char *ram_out = "ram_usage.csv";

static long read_vmrss_kb(void)
{
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return 0;
    char line[128]; long v = 0;
    while (fgets(line, sizeof(line), f))
        if (sscanf(line, "VmRSS: %ld kB", &v) == 1) break;
    fclose(f);
    return v;
}

/* record one RAM sample, throttled to ram_interval so it does not distort timing */
static void ram_sample(void)
{
    double now = now_s();
    if (ram_last < 0.0 || now - ram_last >= ram_interval) {
        if (ram_n < RAM_MAX) {
            ram_t[ram_n] = now - prog_start;
            ram_mb[ram_n] = read_vmrss_kb() / 1024.0f;
            ram_n++;
        }
        ram_last = now;
    }
}

static void ram_dump(void)
{
    FILE *f = fopen(ram_out, "w");
    if (!f) { perror(ram_out); return; }
    fprintf(f, "t_seconds,rss_mb\n");
    for (int i = 0; i < ram_n; i++) fprintf(f, "%.4f,%.4f\n", ram_t[i], ram_mb[i]);
    fclose(f);
    float pk = 0; for (int i = 0; i < ram_n; i++) if (ram_mb[i] > pk) pk = ram_mb[i];
    printf("RAM samples: %d over %.2f s -> %s (peak sampled %.2f MB)\n",
           ram_n, ram_n ? ram_t[ram_n - 1] : 0.0, ram_out, pk);
}

/* ------------------------------------------------------------ input files */
static int read_file(const char *dir, const char *name, void *buf, size_t n)
{
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }
    size_t got = fread(buf, 1, n, f);
    fclose(f);
    if (got != n) { fprintf(stderr, "%s: expected %zu bytes, got %zu\n", path, n, got); return -1; }
    return 0;
}

static int load_inputs(const char *dir)
{
    if (read_file(dir, "X_50.bin", input_raw, sizeof(input_raw)) ||
        read_file(dir, "actual.bin", actual, sizeof(actual)) ||
        read_file(dir, "tmu_pred.bin", tmu_pred, sizeof(tmu_pred)))
        return -1;
    char path[512];
    snprintf(path, sizeof(path), "%s/class_names.txt", dir);
    FILE *f = fopen(path, "r");
    for (int c = 0; c < NUM_CLASSES; c++) snprintf(class_names[c], 64, "class%d", c);
    if (f) {
        int id; char name[64];
        while (fscanf(f, "%d %63[^\n]", &id, name) == 2)
            if (id >= 0 && id < NUM_CLASSES) snprintf(class_names[id], 64, "%s", name);
        fclose(f);
    }
    /* literal k < 178: feature k ; literal 178 + k: NOT feature k ; packed LSB-first like the rules */
    memset(input_packed, 0, sizeof(input_packed));
    for (int s = 0; s < NUM_INPUTS; s++)
        for (int f2 = 0; f2 < INPUT_FEATURES; f2++) {
            int bit = (input_raw[s][f2 / 8] >> (f2 % 8)) & 1;
            int lit = bit ? f2 : INPUT_FEATURES + f2;
            input_packed[s][lit / 8] |= (uint8_t)(1u << (lit % 8));
        }
    return 0;
}

/* ------------------------------------------------------------ Hamming(7,4) */
/* codeword bit 6..0 = P1 P2 D1 P4 D2 D3 D4 ; D1 = nibble bit 3 ; codewords packed MSB-first */
static inline uint8_t get_codeword(const uint8_t *enc, int i)
{
    int bitpos = i * 7, byte = bitpos >> 3, off = bitpos & 7;
    uint16_t w = (uint16_t)enc[byte] << 8;
    if (byte + 1 < ENCODED_PAYLOAD) w |= enc[byte + 1];
    return (uint8_t)((w >> (9 - off)) & 0x7F);
}

static inline uint8_t decode_nibble(uint8_t cw, int *corrected)
{
    int p1 = (cw >> 6) & 1, p2 = (cw >> 5) & 1, d1 = (cw >> 4) & 1, p4 = (cw >> 3) & 1;
    int d2 = (cw >> 2) & 1, d3 = (cw >> 1) & 1, d4 = cw & 1;
    int syn = (p1 ^ d1 ^ d2 ^ d4) | ((p2 ^ d1 ^ d3 ^ d4) << 1) | ((p4 ^ d2 ^ d3 ^ d4) << 2);
    if (syn) {
        cw ^= (uint8_t)(1u << (7 - syn));           /* position 1..7 -> bit 6..0 */
        (*corrected)++;
    }
    return (uint8_t)((((cw >> 4) & 1) << 3) | (((cw >> 2) & 1) << 2) | (((cw >> 1) & 1) << 1) | (cw & 1));
}

static int hamming_decode_payload(const uint8_t *enc, uint8_t *out)
{
    int corrected = 0;
    for (int b = 0; b < PAYLOAD_BYTES; b++) {
        uint8_t hi = decode_nibble(get_codeword(enc, 2 * b), &corrected);
        uint8_t lo = decode_nibble(get_codeword(enc, 2 * b + 1), &corrected);
        out[b] = (uint8_t)((hi << 4) | lo);
    }
    return corrected;
}

/* ------------------------------------------------------------ TM inference for 12 rules */
static void infer_packet(int node, int cls, int weight, const uint8_t *rules)
{
    for (int r = 0; r < RULES_PER_PACKET; r++) {
        const uint8_t *rule = rules + r * RULE_BYTES;
        uint8_t nonempty = 0;
        for (int b = 0; b < RULE_BYTES; b++) nonempty |= rule[b];
        if (!nonempty || weight == 0) continue;     /* empty clause never fires (padding rules) */
        for (int s = 0; s < NUM_INPUTS; s++) {
            const uint8_t *x = input_packed[s];
            int fires = 1;
#ifdef __ARM_NEON
            uint8x16_t acc = vdupq_n_u8(0);
            int b = 0;
            for (; b + 16 <= RULE_BYTES; b += 16)
                acc = vorrq_u8(acc, vbicq_u8(vld1q_u8(rule + b), vld1q_u8(x + b)));   /* rule & ~x */
            if (vmaxvq_u8(acc)) fires = 0;
            for (; fires && b < RULE_BYTES; b++) if (rule[b] & ~x[b]) fires = 0;
#else
            for (int b = 0; b < RULE_BYTES; b++) if (rule[b] & ~x[b]) { fires = 0; break; }
#endif
            if (fires) scores[node][s][cls] += weight;
        }
    }
}

/* ------------------------------------------------------------ one datagram */
static void handle_datagram(const uint8_t *dg, ssize_t len)
{
    uint8_t pay[PAYLOAD_BYTES];
    int expected = raw_mode ? DATAGRAM_RAW : DATAGRAM_ENC;
    if (len != expected) { n_bad++; return; }
    double t0 = now_s();
    int corr = 0;
    if (raw_mode) memcpy(pay, dg + HEADER_BYTES, PAYLOAD_BYTES);
    else corr = hamming_decode_payload(dg + HEADER_BYTES, pay);
    /* the Hamming-protected header copy is authoritative; the clear header is only compared */
    const uint8_t *h = pay + RULE_AREA;
    int wfield = h[0] >> 4, weight = wfield > 7 ? wfield - 16 : wfield, cls = h[0] & 0x0F;
    int seq = h[1] | (h[2] << 8), node = h[3];
    if (cls >= NUM_CLASSES || node >= NUM_NODES || seq >= PACKETS_PER_NODE[node]) { n_bad++; return; }
    if (memcmp(h, dg, HEADER_BYTES) != 0) n_hdr_mismatch[node]++;
    if (seen[node][seq]) { n_dup++; return; }
    seen[node][seq] = 1;
    n_pk[node]++; n_corr[node] += corr; n_rx++;
    double t1 = now_s();
    infer_packet(node, cls, weight, pay);
    double t2 = now_s();
    t_decode += t1 - t0; t_infer += t2 - t1;
}

static long total_expected(void)
{
    long t = 0;
    for (int n = 0; n < NUM_NODES; n++) t += PACKETS_PER_NODE[n];
    return t;
}

/* ------------------------------------------------------------ receive paths */
static int udp_socket(int port)
{
    int fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) { perror("socket"); exit(1); }
    int yes = 1, rcvbuf = 8 * 1024 * 1024;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));
    struct sockaddr_in a = {0};
    a.sin_family = AF_INET; a.sin_addr.s_addr = htonl(INADDR_ANY); a.sin_port = htons(port);
    if (bind(fd, (struct sockaddr *)&a, sizeof(a)) < 0) { perror("bind"); exit(1); }
    return fd;
}

static void run_udp(int idle_ms)
{
    int fds[2] = {udp_socket(DIRECT_PORT), udp_socket(LAPTOP_PORT)};
    uint8_t buf[2048];
    long want = total_expected();
    printf("listening on UDP %d (direct) and %d (laptop), waiting for ALL %ld datagrams "
           "(TM0=%d TM1=%d TM2=%d)...\n", DIRECT_PORT, LAPTOP_PORT, want,
           PACKETS_PER_NODE[0], PACKETS_PER_NODE[1], PACKETS_PER_NODE[2]);
    printf("(will keep waiting; stops only after %d ms of total silence as a safety net)\n", idle_ms);
    long last_print = 0;
    while (n_rx < want) {
        fd_set rd; FD_ZERO(&rd); FD_SET(fds[0], &rd); FD_SET(fds[1], &rd);
        struct timeval tv = {idle_ms / 1000, (idle_ms % 1000) * 1000};
        int r = select((fds[0] > fds[1] ? fds[0] : fds[1]) + 1, &rd, NULL, NULL, n_rx ? &tv : NULL);
        if (r < 0) { if (errno == EINTR) continue; perror("select"); break; }
        if (r == 0) {
            printf("\nno packets for %d ms: giving up with %ld / %ld (TM0=%ld/%d TM1=%ld/%d TM2=%ld/%d). "
                   "Missing packets were lost in transit - lower the sender rate with --gap-us.\n", idle_ms,
                   n_rx, want, n_pk[0], PACKETS_PER_NODE[0], n_pk[1], PACKETS_PER_NODE[1], n_pk[2], PACKETS_PER_NODE[2]);
            break;
        }
        for (int i = 0; i < 2; i++)
            if (FD_ISSET(fds[i], &rd)) {
                ssize_t len = recv(fds[i], buf, sizeof(buf), 0);
                if (len > 0) handle_datagram(buf, len);
            }
        ram_sample();
        if (n_rx - last_print >= 100 || n_rx == want) {       /* live progress */
            printf("\rreceived %ld / %ld   (TM0 %ld/%d  TM1 %ld/%d  TM2 %ld/%d)", n_rx, want,
                   n_pk[0], PACKETS_PER_NODE[0], n_pk[1], PACKETS_PER_NODE[1], n_pk[2], PACKETS_PER_NODE[2]);
            fflush(stdout);
            last_print = n_rx;
        }
    }
    if (n_rx >= want) printf("\nall %ld datagrams received -> decoding + inference\n", want);
    close(fds[0]); close(fds[1]);
}

static void run_files(char **files, int n)
{
    int size = raw_mode ? DATAGRAM_RAW : DATAGRAM_ENC;
    uint8_t buf[2048];
    for (int i = 0; i < n; i++) {
        FILE *f = fopen(files[i], "rb");
        if (!f) { perror(files[i]); continue; }
        size_t got;
        while ((got = fread(buf, 1, size, f)) == (size_t)size) { handle_datagram(buf, (ssize_t)got); ram_sample(); }
        fclose(f);
    }
}

/* ------------------------------------------------------------ report */
static void report(double t_total)
{
    ram_sample();
    int correct_actual = 0, match_ref = 0;
    printf("\n%3s  %-18s %-18s %-18s %s\n", "#", "actual", "reference", "decoded", "match");
    for (int s = 0; s < NUM_INPUTS; s++) {
        int best = 0; int32_t best_v = 0;
        for (int c = 0; c < NUM_CLASSES; c++) {
            int32_t v = 0;
            for (int n = 0; n < NUM_NODES; n++) {           /* cap every TM's class sum at +-T, then add */
                int32_t x = scores[n][s][c];
                v += x > CLASS_SUM_T ? CLASS_SUM_T : (x < -CLASS_SUM_T ? -CLASS_SUM_T : x);
            }
            if (c == 0 || v > best_v) { best = c; best_v = v; }
        }
        correct_actual += best == actual[s];
        match_ref += best == tmu_pred[s];
        printf("%3d  %-18s %-18s %-18s %s\n", s, class_names[actual[s]], class_names[tmu_pred[s]], class_names[best],
               best == tmu_pred[s] ? "YES" : "NO");
    }
    printf("\n");
    for (int n = 0; n < NUM_NODES; n++)
        printf("TM %d: %ld / %d packets, %ld codewords corrected, %ld clear headers differ from protected copy\n",
               n, n_pk[n], PACKETS_PER_NODE[n], n_corr[n], n_hdr_mismatch[n]);
    printf("duplicates %ld, rejected %ld\n", n_dup, n_bad);
    printf("\ndecoded == reference : %d / %d\n", match_ref, NUM_INPUTS);
    printf("accuracy (decoded)   : %.2f %%\n", 100.0 * correct_actual / NUM_INPUTS);
    printf("RESULT               : %s\n", match_ref == NUM_INPUTS ? "PASS (identical to the no-attack model)" : "CHECK");
    printf("\ntime: decode %.3f ms, inference %.3f ms, total %.3f ms\n", t_decode * 1e3, t_infer * 1e3, t_total * 1e3);

    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);                 /* ru_maxrss is in kilobytes on Linux */
    long rss_kb = 0;
    FILE *st = fopen("/proc/self/status", "r");  /* also read current + peak RSS from /proc */
    char line[128];
    long vmrss = 0, vmhwm = 0;
    if (st) {
        while (fgets(line, sizeof(line), st))
            if (sscanf(line, "VmRSS: %ld kB", &vmrss) == 1 || sscanf(line, "VmHWM: %ld kB", &vmhwm) == 1) {}
        fclose(st);
    }
    rss_kb = ru.ru_maxrss;
    printf("RAM : peak RSS %.2f MB (getrusage) | current RSS %.2f MB | peak VmHWM %.2f MB\n",
           rss_kb / 1024.0, vmrss / 1024.0, vmhwm / 1024.0);
}

int main(int argc, char **argv)
{
    const char *dir = ".";
    int idle_ms = 60000, file_mode = 0, first_file = 0, n_files = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--raw")) raw_mode = 1;
        else if (!strcmp(argv[i], "--dir") && i + 1 < argc) dir = argv[++i];
        else if (!strcmp(argv[i], "--idle-ms") && i + 1 < argc) idle_ms = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ram-ms") && i + 1 < argc) ram_interval = atoi(argv[++i]) / 1000.0;
        else if (!strcmp(argv[i], "--ram-out") && i + 1 < argc) ram_out = argv[++i];
        else if (!strcmp(argv[i], "--files")) { file_mode = 1; first_file = i + 1; }
    }
    if (file_mode)
        for (int i = first_file; i < argc && argv[i][0] != '-'; i++) n_files++;
    if (load_inputs(dir)) return 1;
    printf("central node: %d TMs, %d classes, %d literals, %s datagrams of %d bytes\n", NUM_NODES, NUM_CLASSES,
           NUM_LITERALS, raw_mode ? "raw (no ECC)" : "Hamming(7,4)", raw_mode ? DATAGRAM_RAW : DATAGRAM_ENC);
    prog_start = now_s();
    ram_sample();                       /* first sample at startup */
    double t0 = now_s();
    if (file_mode) run_files(argv + first_file, n_files);
    else run_udp(idle_ms);
    report(now_s() - t0);
    ram_sample();
    ram_dump();                         /* write ram_usage.csv */
    return 0;
}
