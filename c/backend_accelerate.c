// Accelerate backend (macOS, iOS): Apple's BLAS, which runs large products on the
// CPU's matrix units. Accelerate manages its own threads, so the pool is unused here.
// Build with -DZENCODEC_BACKEND_ACCELERATE -DACCELERATE_NEW_LAPACK -framework Accelerate.

#if defined(ZENCODEC_BACKEND_ACCELERATE)

#include "backend.h"

#include <Accelerate/Accelerate.h>

bool zc_mat_prepare(zc_mat * a) { (void) a; return true; }

void zc_gemm(const zc_mat * a, int N, const float * B, int ldb, float * C, int ldc, zc_pool * pool) {
    (void) pool;
    for (int tap = 0; tap < a->taps; tap++) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, a->M, N, a->K, 1.0f,
                    a->data + (size_t) tap * a->M * a->K, a->K, B + tap, ldb, 1.0f, C, ldc);
    }
}

#if defined(__aarch64__)
#include <arm_neon.h>

// y[i] += A[row0 + i] . x on this core's NEON units, 4 rows at a time so each load
// of x feeds 4 rows. Row-major, so it reads the same `data` as cblas.
static void gemv_neon(const zc_mat * a, int row0, int rows, const float * x, float * y) {
    const int K = a->K;
    for (int i = 0; i < rows; i += 4) {
        const float * w = a->data + (size_t) (row0 + i) * K;
        float32x4_t acc[4][2];
        for (int r = 0; r < 4; r++) acc[r][0] = acc[r][1] = vdupq_n_f32(0);
        int k = 0;
        for (; k + 8 <= K; k += 8) {
            const float32x4_t x0 = vld1q_f32(x + k), x1 = vld1q_f32(x + k + 4);
            for (int r = 0; r < 4; r++) {
                acc[r][0] = vfmaq_f32(acc[r][0], vld1q_f32(w + (size_t) r * K + k), x0);
                acc[r][1] = vfmaq_f32(acc[r][1], vld1q_f32(w + (size_t) r * K + k + 4), x1);
            }
        }
        for (int r = 0; r < 4; r++) {
            float sum = vaddvq_f32(vaddq_f32(acc[r][0], acc[r][1]));
            for (int kk = k; kk < K; kk++) sum += w[(size_t) r * K + kk] * x[kk];
            y[i + r] += sum;
        }
    }
}
#endif

// The matrix unit is shared by a whole CPU cluster, so a GEMV split across threads
// (each calling with a partial row range) only queues up on it. A full GEMV goes to
// cblas; a partial one, which means other threads are running the rest, runs on
// this core's NEON units instead.
void zc_gemv(const zc_mat * a, int row0, int rows, const float * x, float * y) {
#if defined(__aarch64__)
    if (rows < a->M) {
        gemv_neon(a, row0, rows, x, y);
        return;
    }
#endif
    cblas_sgemv(CblasRowMajor, CblasNoTrans, rows, a->K, 1.0f, a->data + (size_t) row0 * a->K, a->K, x, 1, 1.0f, y, 1);
}

const char * zc_backend_name(void) { return "accelerate"; }

#endif
