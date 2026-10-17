// Accelerate backend (macOS, iOS): Apple's BLAS, which runs large products on the
// CPU's matrix units. Accelerate manages its own threads, so n_threads is unused here.
// Build with -DZENCODEC_BACKEND_ACCELERATE -DACCELERATE_NEW_LAPACK -framework Accelerate.

#if defined(ZENCODEC_BACKEND_ACCELERATE)

#include "backend.h"

#include <Accelerate/Accelerate.h>

bool zc_mat_prepare(zc_mat * a) { (void) a; return true; }

void zc_gemm(const zc_mat * a, int N, const float * B, int ldb, float * C, int ldc, int n_threads) {
    (void) n_threads;
    for (int tap = 0; tap < a->taps; tap++) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, a->M, N, a->K, 1.0f,
                    a->data + (size_t) tap * a->M * a->K, a->K, B + tap, ldb, 1.0f, C, ldc);
    }
}

void zc_gemv(const zc_mat * a, const float * x, float * y) {
    cblas_sgemv(CblasRowMajor, CblasNoTrans, a->M, a->K, 1.0f, a->data, a->K, x, 1, 1.0f, y, 1);
}

const char * zc_backend_name(void) { return "accelerate"; }

#endif
