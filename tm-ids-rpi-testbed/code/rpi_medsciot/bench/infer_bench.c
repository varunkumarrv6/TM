/* Inter-clause NEON vs per-clause inference on the MedSec-25 3-TM model.
 * Times the one-time transpose (preprocessing) separately from inference, so we can see whether the
 * transpose cost is paid back by the faster inter-clause kernel.
 *
 * build (Pi 4):  gcc -O3 -march=native -o infer_bench infer_bench.c
 * build (x86) :  gcc -O3 -o infer_bench infer_bench.c      (scalar fallback, correctness only)
 * run        :  ./infer_bench        (reads tm{n}_rules.bin, tm{n}_weights.bin, inputs_lit.bin, expected.bin)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#include "infer_config.h"

#define CLAUSES_CLASS  PAD_CLAUSES
#define GROUPS         GROUPS_PER_CLASS
#define ITER           200

static uint8_t *raw[NUM_TMS];          /* [class][clause][45]              */
static uint8_t *trn[NUM_TMS];          /* [class][group][45][16] transposed */
static int16_t *wt[NUM_TMS];           /* [class][clause]                  */
static uint8_t inputs[NUM_INPUTS][RULE_BYTES];
static uint8_t expected[NUM_INPUTS];

static double ms(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3+t.tv_nsec/1e6; }

static void *xload(const char *name, size_t n){
    FILE *f=fopen(name,"rb"); if(!f){perror(name);exit(1);}
    void *p=malloc(n); if(fread(p,1,n,f)!=n){fprintf(stderr,"short read %s\n",name);exit(1);} fclose(f); return p;
}

static void load(void){
    char nm[64];
    for(int n=0;n<NUM_TMS;n++){
        sprintf(nm,"tm%d_rules.bin",n);   raw[n]=xload(nm,(size_t)NUM_CLASSES*CLAUSES_CLASS*RULE_BYTES);
        sprintf(nm,"tm%d_weights.bin",n); wt[n]=xload(nm,(size_t)NUM_CLASSES*CLAUSES_CLASS*sizeof(int16_t));
        trn[n]=malloc((size_t)NUM_CLASSES*GROUPS*RULE_BYTES*16);
    }
    uint8_t *in=xload("inputs_lit.bin",(size_t)NUM_INPUTS*RULE_BYTES); memcpy(inputs,in,sizeof inputs); free(in);
    uint8_t *ex=xload("expected.bin",NUM_INPUTS); memcpy(expected,ex,NUM_INPUTS); free(ex);
}

/* [class][clause][byte] -> [class][group][byte][16 lanes] */
static void transpose(void){
    for(int n=0;n<NUM_TMS;n++)
        for(int c=0;c<NUM_CLASSES;c++)
            for(int g=0;g<GROUPS;g++)
                for(int b=0;b<RULE_BYTES;b++)
                    for(int l=0;l<16;l++)
                        trn[n][(((size_t)(c*GROUPS+g))*RULE_BYTES+b)*16+l]
                            = raw[n][(((size_t)c*CLAUSES_CLASS)+(g*16+l))*RULE_BYTES+b];
}

/* ---- inter-clause kernel: 16 clauses at once (your transposed NEON kernel, 45 bytes, int16 weights) ---- */
static inline int32_t eval16(const uint8_t *r, const int16_t *w, const uint8_t *x){
#ifdef __ARM_NEON
    uint8x16_t failed=vdupq_n_u8(0), ne=vdupq_n_u8(0);
    for(int b=0;b<RULE_BYTES;b++){
        uint8x16_t rule=vld1q_u8(r+(size_t)b*16);
        failed=vorrq_u8(failed, vandq_u8(rule, vmvnq_u8(vdupq_n_u8(x[b]))));
        ne=vorrq_u8(ne, rule);
    }
    uint8x16_t fire=vandq_u8(vceqzq_u8(failed), vmvnq_u8(vceqzq_u8(ne)));
    uint8x8_t flo=vshr_n_u8(vget_low_u8(fire),7), fhi=vshr_n_u8(vget_high_u8(fire),7);
    int16x8_t fl=vreinterpretq_s16_u16(vmovl_u8(flo)), fh=vreinterpretq_s16_u16(vmovl_u8(fhi));
    int16x8_t wl=vld1q_s16(w), wh=vld1q_s16(w+8);
    int32x4_t s0=vmull_s16(vget_low_s16(wl),vget_low_s16(fl)),  s1=vmull_s16(vget_high_s16(wl),vget_high_s16(fl));
    int32x4_t s2=vmull_s16(vget_low_s16(wh),vget_low_s16(fh)),  s3=vmull_s16(vget_high_s16(wh),vget_high_s16(fh));
    return vaddvq_s32(s0)+vaddvq_s32(s1)+vaddvq_s32(s2)+vaddvq_s32(s3);
#else
    int32_t acc=0;
    for(int l=0;l<16;l++){ int fires=1,ne=0;
        for(int b=0;b<RULE_BYTES;b++){ uint8_t rb=r[b*16+l]; ne|=rb; if(rb & ~x[b]) fires=0; }
        if(fires&&ne) acc+=w[l]; }
    return acc;
#endif
}

/* ---- per-clause kernel: one clause vectorised across its 45 bytes (the receiver's current approach) ---- */
static inline int clause_fires(const uint8_t *rule, const uint8_t *x){
    int ne=0, fail=0;
#ifdef __ARM_NEON
    uint8x16_t acc=vdupq_n_u8(0), neacc=vdupq_n_u8(0); int b=0;
    for(;b+16<=RULE_BYTES;b+=16){ uint8x16_t rr=vld1q_u8(rule+b); acc=vorrq_u8(acc,vbicq_u8(rr,vld1q_u8(x+b))); neacc=vorrq_u8(neacc,rr); }
    if(vmaxvq_u8(acc)) fail=1; ne=vmaxvq_u8(neacc);
    for(;b<RULE_BYTES;b++){ if(rule[b]&~x[b]) fail=1; ne|=rule[b]; }
#else
    for(int b=0;b<RULE_BYTES;b++){ if(rule[b]&~x[b]) fail=1; ne|=rule[b]; }
#endif
    return !fail && ne;
}

static int fuse_argmax(int32_t t[NUM_CLASSES]){ int b=0; for(int c=1;c<NUM_CLASSES;c++) if(t[c]>t[b]) b=c; return b; }
static inline int32_t clip(int32_t v){ return v>CLASS_SUM_T?CLASS_SUM_T:(v<-CLASS_SUM_T?-CLASS_SUM_T:v); }

static void infer_inter(uint8_t pred[NUM_INPUTS]){
    for(int s=0;s<NUM_INPUTS;s++){
        int32_t tot[NUM_CLASSES]={0};
        for(int n=0;n<NUM_TMS;n++) for(int c=0;c<NUM_CLASSES;c++){
            int32_t sc=0;
            for(int g=0;g<GROUPS;g++)
                sc+=eval16(trn[n]+(((size_t)(c*GROUPS+g))*RULE_BYTES)*16, wt[n]+(size_t)c*CLAUSES_CLASS+g*16, inputs[s]);
            tot[c]+=clip(sc);
        }
        pred[s]=fuse_argmax(tot);
    }
}

static void infer_perclause(uint8_t pred[NUM_INPUTS]){
    for(int s=0;s<NUM_INPUTS;s++){
        int32_t tot[NUM_CLASSES]={0};
        for(int n=0;n<NUM_TMS;n++) for(int c=0;c<NUM_CLASSES;c++){
            int32_t sc=0;
            for(int j=0;j<CLAUSES_CLASS;j++){
                const uint8_t *rule=raw[n]+(((size_t)c*CLAUSES_CLASS)+j)*RULE_BYTES;
                if(clause_fires(rule,inputs[s])) sc+=wt[n][(size_t)c*CLAUSES_CLASS+j];
            }
            tot[c]+=clip(sc);
        }
        pred[s]=fuse_argmax(tot);
    }
}

int main(void){
    load();

    double t0=ms(); transpose(); double t_tr=ms()-t0;   /* one-time preprocessing */

    uint8_t pa[NUM_INPUTS], pb[NUM_INPUTS];
    infer_inter(pa); infer_perclause(pb);
    int ok_exp=0, ok_eq=0;
    for(int s=0;s<NUM_INPUTS;s++){ ok_exp+=pa[s]==expected[s]; ok_eq+=pa[s]==pb[s]; }

    for(int i=0;i<10;i++){ infer_inter(pa); infer_perclause(pb); }   /* warm-up */
    t0=ms(); for(int i=0;i<ITER;i++) infer_inter(pa);  double t_in=(ms()-t0)/ITER;
    t0=ms(); for(int i=0;i<ITER;i++) infer_perclause(pb); double t_pc=(ms()-t0)/ITER;

    printf("model: %d TMs x %d classes x %d clauses (padded), %d groups/class, %d inputs\n",
           NUM_TMS,NUM_CLASSES,CLAUSES_CLASS,GROUPS,NUM_INPUTS);
    printf("inter-clause == expected (tmu): %d/%d   inter == per-clause: %d/%d\n",
           ok_exp,NUM_INPUTS,ok_eq,NUM_INPUTS);
    printf("\n--- timings (for all %d inputs) ---\n",NUM_INPUTS);
    printf("transpose (preprocessing, once): %.4f ms\n", t_tr);
    printf("inter-clause inference         : %.4f ms   (%.5f ms/input)\n", t_in, t_in/NUM_INPUTS);
    printf("per-clause  inference          : %.4f ms   (%.5f ms/input)\n", t_pc, t_pc/NUM_INPUTS);

    double save = t_pc - t_in;
    printf("\n--- amortization ---\n");
    printf("inference saved per 50-input batch (per-clause - inter): %.4f ms\n", save);
    if(save>0){
        printf("transpose is paid back after %.2f batches of %d inputs\n", t_tr/save, NUM_INPUTS);
        printf("single batch total: inter+transpose %.4f ms  vs  per-clause %.4f ms  -> %s\n",
               t_tr+t_in, t_pc, (t_tr+t_in)<t_pc ? "inter WINS even with transpose" : "per-clause wins for ONE batch");
    } else {
        printf("inter-clause not faster here (%.4f ms slower) -> transpose not worth it on this build\n", -save);
    }
    for(int n=0;n<NUM_TMS;n++){ free(raw[n]); free(trn[n]); free(wt[n]); }
    return 0;
}
