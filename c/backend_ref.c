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
        const float * w = c->a->data + ((size_t) tap * M + m) * K;
        for (int k = 0; k < K; k++) {
            const float wv = w[k];
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
    for (int tap = 0; tap < c->a->taps; tap++) {
        const float * w = c->a->data + ((size_t) tap * M + m) * K;
        for (int n = 0; n < N; n++) dst[n] += dot(w, c->Bt + (size_t) (n + tap) * K, K);
    }
}

void zc_ref_gemm(const zc_mat * a, int N, const float * B, int ldb, float * C, int ldc, int n_threads) {
    ref_ctx ctx = {a, N, ldb, ldc, B, NULL, C};
    float * bt = NULL;
    if (N < 256) {
        const int cols = N + a->taps - 1;
        bt = malloc((size_t) cols * a->K * sizeof(float));
        if (bt) {
            for (int k = 0; k < a->K; k++)
                for (int n = 0; n < cols; n++) bt[(size_t) n * a->K + k] = B[(size_t) k * ldb + n];
            ctx.Bt = bt;
        }
    }
    zc_parallel_for(a->M, n_threads, bt ? ref_row_dot : ref_row_axpy, &ctx);
    free(bt);
}

void zc_ref_gemv(const zc_mat * a, const float * x, float * y) {
    for (int m = 0; m < a->M; m++) y[m] += dot(a->data + (size_t) m * a->K, x, a->K);
}

void zc_mat_free(zc_mat * a) {
    free(a->data);
    free(a->packed);
    a->data = NULL;
    a->packed = NULL;
}

#if !defined(ZENCODEC_BACKEND_NEON) && !defined(ZENCODEC_BACKEND_ACCELERATE)

bool zc_mat_prepare(zc_mat * a) { (void) a; return true; }

void zc_gemm(const zc_mat * a, int N, const float * B, int ldb, float * C, int ldc, int n_threads) {
    zc_ref_gemm(a, N, B, ldb, C, ldc, n_threads);
}

void zc_gemv(const zc_mat * a, const float * x, float * y) { zc_ref_gemv(a, x, y); }

const char * zc_backend_name(void) { return "ref"; }

#endif
