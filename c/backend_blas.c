// BLAS backend (Linux and others): any CBLAS, such as OpenBLAS, BLIS or MKL, for the
// matrix products. The pool splits each GEMM into row blocks and every block is one
// single-threaded cblas_sgemm, so the thread count stays the caller's n_threads and
// concurrent decoders don't oversubscribe the CPU. OpenBLAS, and OpenMP builds of any
// BLAS, are set to 1 thread on each thread that calls them; for other BLAS libraries,
// set their thread count to 1 (e.g. BLIS_NUM_THREADS=1).
// Build with -DZENCODEC_BACKEND_BLAS and link the BLAS (e.g. -lopenblas).

#if defined(ZENCODEC_BACKEND_BLAS)

#include "backend.h"

#include <cblas.h>

// Present only when the BLAS is OpenBLAS, or uses OpenMP; a weak reference is NULL
// otherwise. OpenMP's thread count is per calling thread, so it is set on every thread
// that calls the BLAS (this also limits OpenMP on the caller's thread to 1). Apple's
// linker rejects weak references it can't resolve, so there the OpenMP part needs
// -DZENCODEC_BLAS_OMP and -Wl,-U,_omp_set_num_threads.
extern void openblas_set_num_threads(int n) __attribute__((weak));
#if defined(__APPLE__) && !defined(ZENCODEC_BLAS_OMP)
static void (*const omp_set_num_threads)(int) = 0;
#else
extern void omp_set_num_threads(int n) __attribute__((weak));
#endif

static void blas_single_thread(void) {
    static _Thread_local bool done = false;
    if (done) return;
    if (openblas_set_num_threads) openblas_set_num_threads(1);
    if (omp_set_num_threads) omp_set_num_threads(1);
    done = true;
}

bool zc_mat_prepare(zc_mat * a) { (void) a; return true; }

typedef struct {
    const zc_mat * a;
    int            N, ldb, ldc, rows; // rows per task
    const float  * B;
    float        * C;
} blas_ctx;

static void gemm_rows(void * p, int i) {
    const blas_ctx * c = p;
    const int m0 = i * c->rows;
    const int m = c->a->M - m0 < c->rows ? c->a->M - m0 : c->rows;
    blas_single_thread();
    for (int tap = 0; tap < c->a->taps; tap++) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, c->N, c->a->K, 1.0f,
                    c->a->data + ((size_t) tap * c->a->M + m0) * c->a->K, c->a->K, c->B + tap, c->ldb, 1.0f,
                    c->C + (size_t) m0 * c->ldc, c->ldc);
    }
}

void zc_gemm(const zc_mat * a, int N, const float * B, int ldb, float * C, int ldc, zc_pool * pool) {
    // one block per thread, in multiples of 16 rows to keep the BLAS kernels' tiles full
    const int threads = zc_pool_threads(pool);
    int rows = (a->M + threads - 1) / threads;
    rows = (rows + 15) / 16 * 16;
    blas_ctx ctx = {a, N, ldb, ldc, rows, B, C};
    zc_pool_run(pool, (a->M + rows - 1) / rows, gemm_rows, &ctx);
}

void zc_gemv(const zc_mat * a, int row0, int rows, const float * x, float * y) {
    blas_single_thread();
    cblas_sgemv(CblasRowMajor, CblasNoTrans, rows, a->K, 1.0f, a->data + (size_t) row0 * a->K, a->K, x, 1, 1.0f, y, 1);
}

const char * zc_backend_name(void) { return "blas"; }

#endif
