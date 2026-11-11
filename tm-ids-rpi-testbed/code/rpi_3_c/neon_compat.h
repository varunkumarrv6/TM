/*
 * On ARM (Raspberry Pi 4, aarch64) this is just <arm_neon.h> (real NEON).
 *
 * On x86-64 it maps the handful of NEON intrinsics used by the kernels onto
 * SSE2/SSE4.1 so the same source can be built and checked on a PC.  All
 * NEON vector types collapse to __m128i here; only the 128-bit semantics the
 * kernels rely on are implemented.
 */

#ifndef NEON_COMPAT_H
#define NEON_COMPAT_H

#if defined(__ARM_NEON) || defined(__aarch64__)

#include <arm_neon.h>

#else

#include <stdint.h>
#include <smmintrin.h>

typedef __m128i uint8x16_t;
typedef __m128i int8x16_t;
typedef __m128i uint8x8_t;     /* 64-bit types: low 8 bytes significant */
typedef __m128i int8x8_t;
typedef __m128i int16x4_t;
typedef __m128i int16x8_t;
typedef __m128i uint16x8_t;
typedef __m128i int32x4_t;

static inline uint8x16_t vld1q_u8(const uint8_t *p)        { return _mm_loadu_si128((const __m128i *)p); }
static inline int8x16_t  vld1q_s8(const int8_t *p)         { return _mm_loadu_si128((const __m128i *)p); }
static inline void       vst1q_u8(uint8_t *p, uint8x16_t v) { _mm_storeu_si128((__m128i *)p, v); }

static inline uint8x16_t vdupq_n_u8(uint8_t x)  { return _mm_set1_epi8((char)x); }
static inline int16x8_t  vdupq_n_s16(int16_t x) { return _mm_set1_epi16(x); }

static inline uint8x16_t vandq_u8(uint8x16_t a, uint8x16_t b) { return _mm_and_si128(a, b); }
static inline int8x16_t  vandq_s8(int8x16_t a, int8x16_t b)   { return _mm_and_si128(a, b); }
static inline uint8x16_t vorrq_u8(uint8x16_t a, uint8x16_t b) { return _mm_or_si128(a, b); }
static inline uint8x16_t vmvnq_u8(uint8x16_t a)               { return _mm_xor_si128(a, _mm_set1_epi32(-1)); }
static inline uint8x16_t vbicq_u8(uint8x16_t a, uint8x16_t b) { return _mm_andnot_si128(b, a); }  /* a & ~b */
static inline uint8x16_t vceqq_u8(uint8x16_t a, uint8x16_t b) { return _mm_cmpeq_epi8(a, b); }
static inline uint8x16_t vsubq_u8(uint8x16_t a, uint8x16_t b) { return _mm_sub_epi8(a, b); }
static inline int16x8_t  vaddq_s16(int16x8_t a, int16x8_t b)  { return _mm_add_epi16(a, b); }

static inline int8x16_t  vreinterpretq_s8_u8(uint8x16_t a) { return a; }

static inline int8x8_t   vget_low_s8(int8x16_t a)  { return a; }
static inline int8x8_t   vget_high_s8(int8x16_t a) { return _mm_srli_si128(a, 8); }
static inline int16x8_t  vmovl_s8(int8x8_t a)      { return _mm_cvtepi8_epi16(a); }

/* Extra intrinsics used by the original inter-clause kernel. */
static inline uint8x16_t vceqzq_u8(uint8x16_t a)          { return _mm_cmpeq_epi8(a, _mm_setzero_si128()); }
static inline uint8x8_t  vget_low_u8(uint8x16_t a)        { return a; }
static inline uint8x8_t  vget_high_u8(uint8x16_t a)       { return _mm_srli_si128(a, 8); }
static inline uint8x8_t  vshr_n_u8(uint8x8_t a, int n)
{
    return _mm_and_si128(_mm_srl_epi16(a, _mm_cvtsi32_si128(n)), _mm_set1_epi8((char)(0xFF >> n)));
}
static inline uint16x8_t vmovl_u8(uint8x8_t a)            { return _mm_cvtepu8_epi16(a); }
static inline int16x8_t  vld1q_s16(const int16_t *p)      { return _mm_loadu_si128((const __m128i *)p); }
static inline int16x8_t  vreinterpretq_s16_u16(uint16x8_t a) { return a; }
static inline int16x4_t  vget_low_s16(int16x8_t a)        { return a; }
static inline int16x4_t  vget_high_s16(int16x8_t a)       { return _mm_srli_si128(a, 8); }
static inline int32x4_t  vmull_s16(int16x4_t a, int16x4_t b)
{
    return _mm_mullo_epi32(_mm_cvtepi16_epi32(a), _mm_cvtepi16_epi32(b));
}
static inline int32_t vaddvq_s32(int32x4_t v)
{
    v = _mm_add_epi32(v, _mm_srli_si128(v, 8));
    v = _mm_add_epi32(v, _mm_srli_si128(v, 4));
    return _mm_cvtsi128_si32(v);
}

static inline uint8_t vmaxvq_u8(uint8x16_t v)
{
    v = _mm_max_epu8(v, _mm_srli_si128(v, 8));
    v = _mm_max_epu8(v, _mm_srli_si128(v, 4));
    v = _mm_max_epu8(v, _mm_srli_si128(v, 2));
    v = _mm_max_epu8(v, _mm_srli_si128(v, 1));
    return (uint8_t)_mm_cvtsi128_si32(v);
}

static inline uint8_t vminvq_u8(uint8x16_t v)
{
    v = _mm_min_epu8(v, _mm_srli_si128(v, 8));
    v = _mm_min_epu8(v, _mm_srli_si128(v, 4));
    v = _mm_min_epu8(v, _mm_srli_si128(v, 2));
    v = _mm_min_epu8(v, _mm_srli_si128(v, 1));
    return (uint8_t)_mm_cvtsi128_si32(v);
}

static inline uint16_t vaddlvq_u8(uint8x16_t v)
{
    __m128i s = _mm_sad_epu8(v, _mm_setzero_si128());
    return (uint16_t)(_mm_cvtsi128_si32(s) + _mm_extract_epi16(s, 4));
}

static inline int16_t vaddvq_s16(int16x8_t v)
{
    v = _mm_add_epi16(v, _mm_srli_si128(v, 8));
    v = _mm_add_epi16(v, _mm_srli_si128(v, 4));
    v = _mm_add_epi16(v, _mm_srli_si128(v, 2));
    return (int16_t)_mm_extract_epi16(v, 0);
}

/* Extra intrinsics used by the weighted final kernel (#10). */
static inline int16x8_t  vandq_s16(int16x8_t a, int16x8_t b)  { return _mm_and_si128(a, b); }
static inline int32x4_t  vdupq_n_s32(int32_t x)                { return _mm_set1_epi32(x); }
static inline int32x4_t  vpadalq_s16(int32x4_t acc, int16x8_t v)   /* acc + pairwise sums of v */
{
    return _mm_add_epi32(acc, _mm_madd_epi16(v, _mm_set1_epi16(1)));
}

#endif
#endif
