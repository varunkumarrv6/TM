/*
 * Writes the raw .bin files the original Pi program expects into the
 * current directory:
 *
 *   rules_bitpacked.bin  36000 x 66 uint8  (= rules_bitpacked.npy payload)
 *   weights.bin          36000 int16       (= weights.npy widened)
 *   input.bin            66 uint8          (packed literals of X_50 sample N, default 0)
 *
 * Usage: export_bins <iot_dataset dir> [sample]
 */

#define IMPL_NAME "export_bins"
#define TM_NO_MAIN
#pragma GCC diagnostic ignored "-Wunused-function"
#include "tm_common.h"

int main(int argc, char **argv)
{
    const char *dir = argc > 1 ? argv[1] : "..";
    int sample      = argc > 2 ? atoi(argv[2]) : 0;
    if (sample < 0 || sample >= NS) { fprintf(stderr, "sample must be 0..%d\n", NS - 1); return 1; }

    load_all(dir);

    FILE *f = fopen("rules_bitpacked.bin", "wb");
    fwrite(rules_p, 1, sizeof(rules_p), f);
    fclose(f);

    static int16_t w16[TOTAL_CLAUSES];
    for (int i = 0; i < TOTAL_CLAUSES; i++) w16[i] = ((int8_t *)weights)[i];
    f = fopen("weights.bin", "wb");
    fwrite(w16, sizeof(int16_t), TOTAL_CLAUSES, f);
    fclose(f);

    f = fopen("input.bin", "wb");
    fwrite(lit_p[sample], 1, RULE_BYTES, f);
    fclose(f);

    printf("wrote rules_bitpacked.bin, weights.bin, input.bin (sample %d)\n", sample);
    return 0;
}
