// Portable reference backend: plain C loops the compiler can vectorize.
// Always compiled; the other backends fall back to it for matrices they don't handle.

#include "backend.h"

#include <stdlib.h>

// 8 independent accumulators so the compiler can vectorize without -ffast-math.
static float dot(const float * a, const float * b, int n) {
    float acc[8] = {0};
    int i = 0;
    for (; i + 8 <= n; i += 8)
        for (int j = 0; j < 8; j++) acc[j] += a[i + j] * b[i + j];
    float s = 0.0f;
    for (; i < n; i++) s += a[i] * b[i];
    for (int j = 0; j < 8; j++) s += acc[j];
    return s;
}

// Weights a . activations b, widening f16 weights inside the loop: one pass over the
// weights, for the GEMV, which reads each weight only once.
static float dot_w(const zc_weight * a, const float * b, int n) {
    float acc[8] = {0};
    int i = 0;
    for (; i + 8 <= n; i += 8)
        for (int j = 0; j < 8; j++) acc[j] += zc_w2f(a[i + j]) * b[i + j];
    float s = 0.0f;
    for (; i < n; i++) s += zc_w2f(a[i]) * b[i];
    for (int j = 0; j < 8; j++) s += acc[j];
    return s;
}

// A weight row as f32. With f16 weights it is widened into `buf` (K floats) by a plain
// loop, which vectorizes; the dot products that reuse it then run at f32 speed.
#define MAX_K 1024
static const float * weight_row(const zc_weight * w, int K, float * buf) {
#ifdef ZENCODEC_F16_WEIGHTS
    for (int k = 0; k < K; k++) buf[k] = zc_w2f(w[k]);
    return buf;
#else
    (void) K, (void) buf;
    return w;
#endif
}

// Row-by-row widening needs K <= MAX_K, and the decoder's matrices have K <= 512.
static bool k_fits(const zc_mat * a) { return a->K <= MAX_K; }

typedef struct {
    const zc_mat * a;
    int            N, ldb, ldc;
    const float  * B;  // [K][ldb]
    const float  * Bt; // B transposed, [N + taps - 1][K] (dot form only)
    float        * C;
} ref_ctx;

// axpy form, for long rows: C[m] += w * B[k] for every weight w of row m
static void ref_row_axpy(void * p, int m) {
    const ref_ctx * c = p;
    const int M = c->a->M, K = c->a->K, N = c->N;
    float * dst = c->C + (size_t) m * c->ldc;
    for (int tap = 0; tap < c->a->taps; tap++) {
        const zc_weight * w = c->a->data + ((size_t) tap * M + m) * K;
        for (int k = 0; k < K; k++) {
            const float wv = zc_w2f(w[k]);
            const float * src = c->B + (size_t) k * c->ldb + tap;
            for (int n = 0; n < N; n++) dst[n] += wv * src[n];
        }
    }
}

// dot form, for short rows (e.g. the LSTM's 75 frames): C[m][n] += A[m] . B[:, n]
static void ref_row_dot(void * p, int m) {
    const ref_ctx * c = p;
    const int M = c->a->M, K = c->a->K, N = c->N;
    float * dst = c->C + (size_t) m * c->ldc;
    float buf[MAX_K];
    for (int tap = 0; tap < c->a->taps; tap++) {
        const float * w = weight_row(c->a->data + ((size_t) tap * M + m) * K, K, buf);
        for (int n = 0; n < N; n++) dst[n] += dot(w, c->Bt + (size_t) (n + tap) * K, K);
    }
}

void zc_ref_gemm(const zc_mat * a, int N, const float * B, int ldb, float * C, int ldc, zc_pool * pool) {
    ref_ctx ctx = {a, N, ldb, ldc, B, NULL, C};
    float * bt = NULL;
    if (N < 256 && k_fits(a)) {
        const int cols = N + a->taps - 1;
        bt = malloc((size_t) cols * a->K * sizeof(float));
        if (bt) {
            for (int k = 0; k < a->K; k++)
                for (int n = 0; n < cols; n++) bt[(size_t) n * a->K + k] = B[(size_t) k * ldb + n];
            ctx.Bt = bt;
        }
    }
    zc_pool_run(pool, a->M, bt ? ref_row_dot : ref_row_axpy, &ctx);
    free(bt);
}

void zc_ref_gemv(const zc_mat * a, int row0, int rows, const float * x, float * y) {
    for (int i = 0; i < rows; i++) y[i] += dot_w(a->data + (size_t) (row0 + i) * a->K, x, a->K);
}

void zc_mat_free(zc_mat * a) {
    free(a->data);
    free(a->packed);
    a->data = NULL;
    a->packed = NULL;
}

#if !defined(ZENCODEC_BACKEND_NEON) && !defined(ZENCODEC_BACKEND_ACCELERATE)

bool zc_mat_prepare(zc_mat * a) { (void) a; return true; }

void zc_gemm(const zc_mat * a, int N, const float * B, int ldb, float * C, int ldc, zc_pool * pool) {
    zc_ref_gemm(a, N, B, ldb, C, ldc, pool);
}

void zc_gemv(const zc_mat * a, int row0, int rows, const float * x, float * y) { zc_ref_gemv(a, row0, rows, x, y); }

const char * zc_backend_name(void) { return "ref" ZC_WEIGHTS_SUFFIX; }

#endif
