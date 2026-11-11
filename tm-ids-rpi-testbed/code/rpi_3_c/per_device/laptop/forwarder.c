/*
 * Laptop forwarder (man in the middle), in C, for the TM routed through the laptop (VIA_LAPTOP_NODE): receives its
 * datagrams on LAPTOP_PORT, corrupts them and forwards them to CENTRAL_IP:CENTRAL_PORT (net_config.h).
 * Same behaviour and same random positions per seed as forwarder.py. Change K by hand between runs.
 *
 *   -t all (default): K random BIT positions anywhere in the datagram (clear header, rule data, Hamming parity,
 *                     protected header copy), the same K positions in every packet; drawn from 967 x 8 bits for a
 *                     Hamming datagram, 556 x 8 for a raw one
 *   -t literals     : K literal positions (0..355) changed in every real rule (the earlier experiment)
 *   -a include : bit -> 1    exclude : bit -> 0    toggle : flip    none : forward unchanged
 * Positions: splitmix64 + partial Fisher-Yates from the seed (-s), sorted; or explicit with -l.
 * Before the first packet of every transmission a label packet goes to the central RPi.
 *
 * build : gcc -O2 -o forwarder forwarder.c
 * run   : ./forwarder -a toggle -k 4            4 random bits anywhere in every packet flipped
 *         ./forwarder -a include -k 8 -s 3      another random set of 8 bit positions set to 1
 *         ./forwarder -a none                   clean pass-through
 *         ./forwarder -a include -k 8 -t literals
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <signal.h>
#include <time.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include "net_config.h"

#define LITERALS          356
#define HDR               8
#define RULE_BYTES        45
#define RULES_PER_PACKET  12
#define DG_HAM            (HDR + 959)
#define DG_RAW            (HDR + 548)
#define MAXPOS            (DG_HAM * 8)
#define NEW_RUN_GAP_S     2.0

static volatile sig_atomic_t stop;
static void on_signal(int s) { (void)s; stop = 1; }

static uint64_t splitmix64(uint64_t *x)
{
    uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static int cmp_int(const void *a, const void *b) { return *(const int *)a - *(const int *)b; }

/* K distinct positions of 0..n-1, sorted (same as forwarder.py) */
static void choose(int k, uint64_t seed, int n, int *out)
{
    static int pool[MAXPOS];
    for (int i = 0; i < n; i++) pool[i] = i;
    for (int i = 0; i < k; i++) {
        int j = i + (int)(splitmix64(&seed) % (uint64_t)(n - i));
        int t = pool[i]; pool[i] = pool[j]; pool[j] = t;
        out[i] = pool[i];
    }
    qsort(out, (size_t)k, sizeof(int), cmp_int);
}

/* byte index in the datagram and bit mask of literal `lit` of rule `rule` */
static void literal_bit(int rule, int lit, int layout, int *byte, uint8_t *mask)
{
    int B = RULE_BYTES * rule + lit / 8, b = lit % 8;
    if (!layout) { *byte = HDR + B; *mask = (uint8_t)(1u << b); return; }
    static const int pos[4] = {6, 5, 4, 2};                         /* nibble bit 0..3 -> D4 D3 D2 D1 */
    int cw = b >= 4 ? 2 * B : 2 * B + 1;
    int enc = 7 * cw + pos[b % 4];
    *byte = HDR + enc / 8;
    *mask = (uint8_t)(0x80u >> (enc % 8));
}

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int append_list(char *dst, size_t cap, int p, const char *pre, const int *v, int n)
{
    p += snprintf(dst + p, cap - (size_t)p, "%s", pre);
    for (int i = 0; i < n && (size_t)p < cap; i++) p += snprintf(dst + p, cap - (size_t)p, i ? " %d" : "%d", v[i]);
    return p;
}

int main(int argc, char **argv)
{
    const char *attack = "none", *list_arg = NULL, *target = "all";
    int k = 0;
    uint64_t seed = 1;
    int opt;
    while ((opt = getopt(argc, argv, "a:k:s:l:t:")) != -1) {
        if (opt == 'a') attack = optarg;
        else if (opt == 'k') k = atoi(optarg);
        else if (opt == 's') seed = strtoull(optarg, NULL, 10);
        else if (opt == 'l') list_arg = optarg;
        else if (opt == 't') target = optarg;
        else { fprintf(stderr, "usage: %s -a none|include|exclude|toggle [-k K] [-s seed] [-t all|literals] "
                               "[-l 3,17,...]\n", argv[0]);
               return 1; }
    }
    int mode = !strcmp(attack, "include") ? 1 : !strcmp(attack, "exclude") ? 2 : !strcmp(attack, "toggle") ? 3 :
               !strcmp(attack, "none") ? 0 : -1;
    int lits_mode = !strcmp(target, "literals");
    if (mode < 0 || (!lits_mode && strcmp(target, "all"))) { fprintf(stderr, "bad -a or -t\n"); return 1; }

    static int pos[2][MAXPOS];                                      /* [layout][i]: layout 0 raw, 1 Hamming */
    int npos = 0;
    if (mode && list_arg) {
        char *s = strdup(list_arg);
        for (char *t = strtok(s, ","); t && npos < MAXPOS; t = strtok(NULL, ",")) pos[0][npos] = pos[1][npos] = atoi(t), npos++;
        free(s);
    } else if (mode) {
        int nmax = lits_mode ? LITERALS : DG_RAW * 8;
        if (k < 0 || k > nmax) { fprintf(stderr, "K must be 0..%d\n", nmax); return 1; }
        npos = k;
        if (lits_mode) { choose(k, seed, LITERALS, pos[1]); memcpy(pos[0], pos[1], sizeof(int) * (size_t)k); }
        else { choose(k, seed, DG_RAW * 8, pos[0]); choose(k, seed, DG_HAM * 8, pos[1]); }
    }

    static char label[65536];
    int p = snprintf(label, sizeof label, "TMLABEL attack=%s K=%d seed=%lld target=%s literals=", attack, npos,
                     mode && list_arg ? -1LL : (long long)seed, target);
    if (npos) {
        if (lits_mode || list_arg) p = append_list(label, sizeof label, p, "", pos[1], npos);
        else {
            p = append_list(label, sizeof label, p, "hamming:", pos[1], npos);
            p = append_list(label, sizeof label, p, " raw:", pos[0], npos);
        }
    }

    /* (byte, mask) targets: target all -> per layout; target literals -> per layout and number of real rules */
    static int tb[2][RULES_PER_PACKET + 1][RULES_PER_PACKET * LITERALS];
    static uint8_t tm[2][RULES_PER_PACKET + 1][RULES_PER_PACKET * LITERALS];
    static int ab[2][MAXPOS];
    static uint8_t am[2][MAXPOS];
    for (int lay = 0; lay < 2; lay++) {
        if (lits_mode) {
            for (int n = 1; n <= RULES_PER_PACKET; n++)
                for (int r = 0, t = 0; r < n; r++)
                    for (int i = 0; i < npos; i++, t++) literal_bit(r, pos[lay][i], lay, &tb[lay][n][t], &tm[lay][n][t]);
        } else {
            for (int i = 0; i < npos; i++) { ab[lay][i] = pos[lay][i] / 8; am[lay][i] = (uint8_t)(0x80u >> (pos[lay][i] % 8)); }
        }
    }

    int rx = socket(AF_INET, SOCK_DGRAM, 0), tx = socket(AF_INET, SOCK_DGRAM, 0);
    int one = 1, rcvbuf = 8 << 20;
    setsockopt(rx, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    setsockopt(rx, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof rcvbuf);
    struct sockaddr_in in = {0}, dst = {0};
    in.sin_family = AF_INET; in.sin_port = htons(LAPTOP_PORT); in.sin_addr.s_addr = htonl(INADDR_ANY);
    if (bind(rx, (struct sockaddr *)&in, sizeof in) < 0) { perror("bind"); return 1; }
    dst.sin_family = AF_INET; dst.sin_port = htons(CENTRAL_PORT);
    if (inet_pton(AF_INET, CENTRAL_IP, &dst.sin_addr) != 1) { fprintf(stderr, "bad CENTRAL_IP\n"); return 1; }
    struct sigaction sa = {0};
    sa.sa_handler = on_signal;                                      /* no SA_RESTART: Ctrl-C ends recv */
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    printf("forwarder: UDP %d -> %s:%d   attack %s, K = %d, target %s\n", LAPTOP_PORT, CENTRAL_IP, CENTRAL_PORT,
           attack, npos, target);
    if (npos) printf("  %s\n", strstr(label, "literals=") + 9);
    fflush(stdout);

    uint8_t buf[2048];
    double last = -1e9;
    long n_run = 0, n_fwd = 0, n_hit = 0;
    while (!stop) {
        ssize_t len = recv(rx, buf, sizeof buf, 0);
        if (len < 0) continue;
        double t = now_s();
        if (t - last > NEW_RUN_GAP_S) {
            if (n_fwd) printf("  transmission %ld: %ld datagrams forwarded, %ld changed\n", n_run, n_fwd, n_hit);
            n_run++; n_fwd = n_hit = 0;
            sendto(tx, label, strlen(label), 0, (struct sockaddr *)&dst, sizeof dst);
            fflush(stdout);
        }
        last = t;
        int lay = len == DG_HAM ? 1 : len == DG_RAW ? 0 : -1;
        if (mode && lay >= 0) {
            int cnt = 0;
            const int *bi = NULL;
            const uint8_t *mi = NULL;
            if (!lits_mode) { cnt = npos; bi = ab[lay]; mi = am[lay]; }
            else if (buf[1] >= 1 && buf[1] <= RULES_PER_PACKET) { cnt = buf[1] * npos; bi = tb[lay][buf[1]]; mi = tm[lay][buf[1]]; }
            for (int i = 0; i < cnt; i++) {
                if (mode == 1) buf[bi[i]] |= mi[i];
                else if (mode == 2) buf[bi[i]] &= (uint8_t)~mi[i];
                else buf[bi[i]] ^= mi[i];
            }
            if (cnt) n_hit++;
        }
        sendto(tx, buf, (size_t)len, 0, (struct sockaddr *)&dst, sizeof dst);
        n_fwd++;
    }
    printf("\nstopped; last transmission: %ld datagrams forwarded, %ld changed\n", n_fwd, n_hit);
    return 0;
}
