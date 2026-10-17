// zencodec compute backends (internal).
//
// Every heavy layer of the decoder is a matrix product, so a backend only has to
// provide a GEMM, a GEMV and a way to store its weight matrices. Exactly one backend
// is compiled in, chosen with a define:
//
//   (none)                       ref:        portable C, no dependencies
//   -DZENCODEC_BACKEND_NEON      neon:       hand-written AArch64 NEON kernels
//   -DZENCODEC_BACKEND_ACCELERATE accelerate: Apple's BLAS (link -framework Accelerate)
#ifndef ZENCODEC_BACKEND_H
#define ZENCODEC_BACKEND_H

#include <stdbool.h>

// A weight matrix made of `taps` stacked row-major [M][K] blocks (taps * M rows in all).
// Built by the decoder in `data`; zc_mat_prepare may repack it into `packed` and free
// `data`.
typedef struct {
    int     M, K, taps;
    float * data;
    void  * packed;
} zc_mat;

bool zc_mat_prepare(zc_mat * a);
void zc_mat_free(zc_mat * a);

// C[M][N] += sum over tap k of A_k[M][K] * B[:, k : k + N]
// B is [K][ldb] and C is [M][ldc], both row-major. With taps == 1 this is a plain GEMM;
// with taps > 1 it is a stride-1 convolution over a padded input.
void zc_gemm(const zc_mat * a, int N, const float * B, int ldb, float * C, int ldc, int n_threads);

// y[M] += A[M][K] * x[K]   (taps == 1)
void zc_gemv(const zc_mat * a, const float * x, float * y);

const char * zc_backend_name(void);

// Portable implementations, also used by the other backends as a fallback.
void zc_ref_gemm(const zc_mat * a, int N, const float * B, int ldb, float * C, int ldc, int n_threads);
void zc_ref_gemv(const zc_mat * a, const float * x, float * y);

// Runs fn(ctx, i) for i in [0, n) on up to n_threads threads (defined in zencodec.c).
typedef void (*zc_task_fn)(void * ctx, int i);
void zc_parallel_for(int n, int n_threads, zc_task_fn fn, void * ctx);

#endif // ZENCODEC_BACKEND_H
