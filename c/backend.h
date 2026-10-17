// zencodec compute backends (internal).
//
// Every heavy layer of the decoder is a matrix product, so a backend only has to
// provide a GEMM, a GEMV and a way to store its weight matrices. Exactly one backend
// is compiled in, chosen with a define:
//
//   (none)                       ref:        portable C, no dependencies
//   -DZENCODEC_BACKEND_NEON      neon:       hand-written AArch64 NEON kernels
//   -DZENCODEC_BACKEND_ACCELERATE accelerate: Apple's BLAS (link -framework Accelerate)
//
// -DZENCODEC_F16_WEIGHTS keeps the weights in memory as IEEE half floats, which halves
// their size (about 23 MB less). The kernels widen them to f32 as they load them, and
// all arithmetic stays in f32. Supported by ref and neon; accelerate's BLAS takes f32 only.
#ifndef ZENCODEC_BACKEND_H
#define ZENCODEC_BACKEND_H

#include "pool.h"

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#if defined(ZENCODEC_F16_WEIGHTS) && defined(ZENCODEC_BACKEND_ACCELERATE)
#error "ZENCODEC_F16_WEIGHTS is not supported by the accelerate backend (its BLAS is f32 only)"
#endif

// IEEE half to float. With hardware conversion (any AArch64, or x86 built with F16C,
// e.g. -mf16c or -march=native) this is one instruction, and loops over it vectorize.
// Otherwise it is done with integer and float ops, branch-free so it still vectorizes:
// the exponent and mantissa move into float position, and multiplying by 2^112 rebiases
// the exponent (15 -> 127), which also handles subnormals exactly. Inf and NaN, which
// weights don't contain, come out as large finite values there.
#if defined(__FLT16_MAX__) && (defined(__aarch64__) || defined(__F16C__))
#define ZC_HW_F16 1
static inline float zc_h2f(uint16_t h) {
    _Float16 v;
    memcpy(&v, &h, sizeof(v));
    return (float) v;
}
#else
static inline float zc_h2f(uint16_t h) {
    const uint32_t em = (uint32_t) (h & 0x7fff) << 13;
    float f;
    memcpy(&f, &em, sizeof(f));
    f *= 0x1p112f;
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    bits |= (uint32_t) (h & 0x8000) << 16;
    memcpy(&f, &bits, sizeof(f));
    return f;
}
#endif

// float to IEEE half, rounding to nearest even (load time only).
static inline uint16_t zc_f2h(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    const uint16_t sign = (uint16_t) ((x >> 16) & 0x8000);
    const uint32_t ax = x & 0x7fffffff;
    if (ax >= 0x7f800000) return sign | 0x7c00 | (ax > 0x7f800000 ? 0x200 : 0); // inf, nan
    if (ax >= 0x477ff000) return sign | 0x7c00;                                 // overflow
    if (ax < 0x38800000) { // half subnormal or zero: let float addition do the rounding
        float a;
        memcpy(&a, &ax, sizeof(a));
        a += 0.5f; // the half subnormals are then the low mantissa bits of 0.5f + |f|
        uint32_t r;
        memcpy(&r, &a, sizeof(r));
        return sign | (uint16_t) (r - 0x3f000000);
    }
    const uint32_t odd = (ax >> 13) & 1;
    return sign | (uint16_t) ((ax + 0xc8000fff + odd) >> 13); // rebias 127 -> 15 and round
}

// The type weights are stored in, and its conversion to float.
#ifdef ZENCODEC_F16_WEIGHTS
typedef uint16_t zc_weight;
static inline float     zc_w2f(zc_weight w) { return zc_h2f(w); }
static inline zc_weight zc_f2w(float f)     { return zc_f2h(f); }
#define ZC_WEIGHTS_SUFFIX "+f16"
#else
typedef float zc_weight;
static inline float     zc_w2f(zc_weight w) { return w; }
static inline zc_weight zc_f2w(float f)     { return f; }
#define ZC_WEIGHTS_SUFFIX ""
#endif

// A weight matrix made of `taps` stacked row-major [M][K] blocks (taps * M rows in all).
// Built by the decoder in `data`; zc_mat_prepare may repack it into `packed` and free
// `data`.
typedef struct {
    int         M, K, taps;
    zc_weight * data;
    void      * packed;
} zc_mat;

bool zc_mat_prepare(zc_mat * a);
void zc_mat_free(zc_mat * a);

// C[M][N] += sum over tap k of A_k[M][K] * B[:, k : k + N]
// B is [K][ldb] and C is [M][ldc], both row-major. With taps == 1 this is a plain GEMM;
// with taps > 1 it is a stride-1 convolution over a padded input. Runs on `pool`
// (may be NULL).
void zc_gemm(const zc_mat * a, int N, const float * B, int ldb, float * C, int ldc, zc_pool * pool);

// y[i] += A[row0 + i] . x  for i in [0, rows)   (taps == 1, single-threaded)
// row0 and rows must be multiples of ZC_GEMV_ROWS, so callers can split a GEMV across
// threads (the LSTM recurrence does).
#define ZC_GEMV_ROWS 8
void zc_gemv(const zc_mat * a, int row0, int rows, const float * x, float * y);

const char * zc_backend_name(void); // e.g. "neon", or "neon+f16" with f16 weights

// Portable implementations, also used by the other backends as a fallback.
void zc_ref_gemm(const zc_mat * a, int N, const float * B, int ldb, float * C, int ldc, zc_pool * pool);
void zc_ref_gemv(const zc_mat * a, int row0, int rows, const float * x, float * y);

#endif // ZENCODEC_BACKEND_H
