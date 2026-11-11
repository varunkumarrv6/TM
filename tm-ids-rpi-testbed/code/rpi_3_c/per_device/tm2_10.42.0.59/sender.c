/*
 * Sending RPi of the physical 3-TM testbed: sends its TM's model (tm<n>_hamming.bin, or tm<n>_raw.bin with the
 * argument "raw") as UDP datagrams, one datagram every GAP_US microseconds. All addresses come from net_config.h:
 *   TM n == VIA_LAPTOP_NODE : RPIn_IP -> LAPTOP_IP:LAPTOP_PORT   (laptop forwarder attacks it, passes it on)
 *   any other TM n          : RPIn_IP -> CENTRAL_IP:CENTRAL_PORT
 * Which TM this RPi is: the one whose tm<n>_hamming.bin lies in this folder (or compile with -DNODE=n).
 * If RPIn_IP is not an address of this RPi, the OS picks the source address (a warning is printed).
 *
 * build : gcc -O2 -o sender sender.c
 * run   : ./sender          Hamming(7,4) stream (967-byte datagrams)
 *         ./sender raw      no ECC (556-byte datagrams)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include "net_config.h"

#define DG_HAM (8 + 959)
#define DG_RAW (8 + 548)

static const char *SRC_IPS[3] = {RPI0_IP, RPI1_IP, RPI2_IP};

static int find_node(void)
{
#ifdef NODE
    return NODE;
#else
    int node = -1, found = 0;
    for (int n = 0; n < 3; n++) {
        char path[32];
        snprintf(path, sizeof path, "tm%d_hamming.bin", n);
        if (access(path, R_OK) == 0) { node = n; found++; }
    }
    if (found != 1) {
        fprintf(stderr, found ? "more than one tm<n>_hamming.bin here: keep only this RPi's file or build with "
                                "-DNODE=n\n"
                              : "no tm<n>_hamming.bin in this folder\n");
        exit(1);
    }
    return node;
#endif
}

int main(int argc, char **argv)
{
    int raw = argc > 1 && !strcmp(argv[1], "raw");
    const int dg = raw ? DG_RAW : DG_HAM;
    const int node = find_node();
    const char *src_ip = SRC_IPS[node];
    const char *dst_ip = node == VIA_LAPTOP_NODE ? LAPTOP_IP : CENTRAL_IP;
    const int dst_port = node == VIA_LAPTOP_NODE ? LAPTOP_PORT : CENTRAL_PORT;

    char path[64];
    snprintf(path, sizeof path, "tm%d_%s.bin", node, raw ? "raw" : "hamming");
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return 1; }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *data = malloc((size_t)len);
    if (!data || fread(data, 1, (size_t)len, f) != (size_t)len || len % dg) {
        fprintf(stderr, "%s: %ld bytes is not a whole number of %d-byte datagrams\n", path, len, dg);
        return 1;
    }
    fclose(f);
    if ((data[7] >> 4) != node) { fprintf(stderr, "%s holds TM %d, not %d\n", path, data[7] >> 4, node); return 1; }

    int s = socket(AF_INET, SOCK_DGRAM, 0);
    int sndbuf = 4 << 20;
    setsockopt(s, SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof sndbuf);
    struct sockaddr_in src = {0};
    src.sin_family = AF_INET;
    if (inet_pton(AF_INET, src_ip, &src.sin_addr) != 1 || bind(s, (struct sockaddr *)&src, sizeof src) < 0) {
        printf("note: %s is not an address of this RPi -> the OS chooses the source address\n", src_ip);
        src_ip = "(auto)";
    }
    struct sockaddr_in dst = {0};
    dst.sin_family = AF_INET;
    dst.sin_port = htons(dst_port);
    if (inet_pton(AF_INET, dst_ip, &dst.sin_addr) != 1) { fprintf(stderr, "bad destination %s\n", dst_ip); return 1; }

    int n = (int)(len / dg);
    printf("TM %d: %s, %d datagrams of %d B, %s -> %s:%d%s, gap %d us\n", node, path, n, dg, src_ip, dst_ip, dst_port,
           node == VIA_LAPTOP_NODE ? " (laptop)" : " (central RPi)", GAP_US);
    struct timespec t0, t;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < n; i++) {
        long long due = (long long)t0.tv_sec * 1000000000LL + t0.tv_nsec + (long long)i * GAP_US * 1000LL;
        struct timespec ts = {(time_t)(due / 1000000000LL), (long)(due % 1000000000LL)};
        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, NULL);          /* fixed pace, no drift */
        if (sendto(s, data + (size_t)i * dg, dg, 0, (struct sockaddr *)&dst, sizeof dst) != dg) {
            perror("sendto");
            return 1;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t);
    printf("TM %d: sent %d datagrams in %.3f s\n", node, n, (t.tv_sec - t0.tv_sec) + (t.tv_nsec - t0.tv_nsec) * 1e-9);
    return 0;
}
