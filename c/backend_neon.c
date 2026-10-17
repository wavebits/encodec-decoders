// NEON backend (AArch64: Apple M-series, phones, Raspberry Pi, ...). No dependencies.
//
// GEMM with a register-blocked 8x12 micro-kernel: 24 accumulator registers hold an
// 8-row x 12-column block of C while K is swept, so each loaded value feeds several
// FMAs instead of one. Weights are packed once at load into 8-row panels ([K][8]);
// the input is packed per task into 12-column panels ([K][12]). Work is split into
// MB x NB tiles of C, one task each.

#if defined(ZENCODEC_BACKEND_NEON)

#include "backend.h"

#include <arm_neon.h>
#include <stdlib.h>
#include <string.h>

#define MR 8  // rows per micro-kernel (A panel height)
#define NR 12 // columns per micro-kernel (B panel width)
#define MB 64 // rows per task, a multiple of MR
#define NB 96 // columns per task, a multiple of NR

// Matrices whose block height is not a multiple of MR (only the 1-channel output
// conv) stay row-major and use the reference code.
bool zc_mat_prepare(zc_mat * a) {
    if (a->M % MR) return true;
    const int panels = a->taps * a->M / MR;
    float * p = malloc((size_t) panels * MR * a->K * sizeof(float));
    if (!p) return false;
    for (int pi = 0; pi < panels; pi++)
        for (int k = 0; k < a->K; k++)
            for (int r = 0; r < MR; r++)
                p[((size_t) pi * a->K + k) * MR + r] = a->data[((size_t) pi * MR + r) * a->K + k];
    free(a->data);
    a->data = NULL;
    a->packed = p;
    return true;
}

// c[8][12] (row stride ldc) += a[K][8]^T * b[K][12]
static void kernel_8x12(int K, const float * restrict a, const float * restrict b, float * restrict c, int ldc) {
    float32x4_t c0[3], c1[3], c2[3], c3[3], c4[3], c5[3], c6[3], c7[3];
    for (int j = 0; j < 3; j++) {
        c0[j] = vld1q_f32(c + 0 * ldc + 4 * j);
        c1[j] = vld1q_f32(c + 1 * ldc + 4 * j);
        c2[j] = vld1q_f32(c + 2 * ldc + 4 * j);
        c3[j] = vld1q_f32(c + 3 * ldc + 4 * j);
        c4[j] = vld1q_f32(c + 4 * ldc + 4 * j);
        c5[j] = vld1q_f32(c + 5 * ldc + 4 * j);
        c6[j] = vld1q_f32(c + 6 * ldc + 4 * j);
        c7[j] = vld1q_f32(c + 7 * ldc + 4 * j);
    }
    for (int k = 0; k < K; k++) {
        const float32x4_t a0 = vld1q_f32(a), a1 = vld1q_f32(a + 4);
        const float32x4_t b0 = vld1q_f32(b), b1 = vld1q_f32(b + 4), b2 = vld1q_f32(b + 8);
#define ROW(cr, av, lane)                              \
        cr[0] = vfmaq_laneq_f32(cr[0], b0, av, lane); \
        cr[1] = vfmaq_laneq_f32(cr[1], b1, av, lane); \
        cr[2] = vfmaq_laneq_f32(cr[2], b2, av, lane);
        ROW(c0, a0, 0) ROW(c1, a0, 1) ROW(c2, a0, 2) ROW(c3, a0, 3)
        ROW(c4, a1, 0) ROW(c5, a1, 1) ROW(c6, a1, 2) ROW(c7, a1, 3)
#undef ROW
        a += MR;
        b += NR;
    }
    for (int j = 0; j < 3; j++) {
        vst1q_f32(c + 0 * ldc + 4 * j, c0[j]);
        vst1q_f32(c + 1 * ldc + 4 * j, c1[j]);
        vst1q_f32(c + 2 * ldc + 4 * j, c2[j]);
        vst1q_f32(c + 3 * ldc + 4 * j, c3[j]);
        vst1q_f32(c + 4 * ldc + 4 * j, c4[j]);
        vst1q_f32(c + 5 * ldc + 4 * j, c5[j]);
        vst1q_f32(c + 6 * ldc + 4 * j, c6[j]);
        vst1q_f32(c + 7 * ldc + 4 * j, c7[j]);
    }
}

typedef struct {
    const zc_mat * a;
    int            N, ldb, ldc, tiles_n;
    const float  * B;
    float        * C;
} neon_ctx;

static void gemm_tile(void * p, int i) {
    const neon_ctx * c = p;
    const int M = c->a->M, K = c->a->K;
    const int m0 = (i / c->tiles_n) * MB, n0 = (i % c->tiles_n) * NB;
    const int mb = M - m0 < MB ? M - m0 : MB;
    const int nb = c->N - n0 < NB ? c->N - n0 : NB;
    const int nq = (nb + NR - 1) / NR;

    float * bp = malloc((size_t) K * nq * NR * sizeof(float));
    if (!bp) abort();
    for (int tap = 0; tap < c->a->taps; tap++) {
        // pack B[:, tap + n0 : tap + n0 + nb] into 12-column panels, zero-padded
        for (int q = 0; q < nq; q++) {
            const int cols = nb - q * NR < NR ? nb - q * NR : NR;
            float * dst = bp + (size_t) q * K * NR;
            for (int k = 0; k < K; k++) {
                const float * src = c->B + (size_t) k * c->ldb + tap + n0 + q * NR;
                memcpy(dst + (size_t) k * NR, src, (size_t) cols * sizeof(float));
                for (int j = cols; j < NR; j++) dst[(size_t) k * NR + j] = 0.0f;
            }
        }
        const float * ap = (const float *) c->a->packed + ((size_t) tap * M + m0) * K;
        for (int q = 0; q < nq; q++) {
            const int cols = nb - q * NR < NR ? nb - q * NR : NR;
            const float * b = bp + (size_t) q * K * NR;
            for (int r = 0; r < mb; r += MR) {
                const float * a = ap + (size_t) r * K;
                float * cc = c->C + (size_t) (m0 + r) * c->ldc + n0 + q * NR;
                if (cols == NR) {
                    kernel_8x12(K, a, b, cc, c->ldc);
                } else { // right edge: go through a full-width scratch tile
                    float t[MR * NR] = {0};
                    for (int rr = 0; rr < MR; rr++) memcpy(t + rr * NR, cc + (size_t) rr * c->ldc, cols * sizeof(float));
                    kernel_8x12(K, a, b, t, NR);
                    for (int rr = 0; rr < MR; rr++) memcpy(cc + (size_t) rr * c->ldc, t + rr * NR, cols * sizeof(float));
                }
            }
        }
    }
    free(bp);
}

void zc_gemm(const zc_mat * a, int N, const float * B, int ldb, float * C, int ldc, int n_threads) {
    if (!a->packed) {
        zc_ref_gemm(a, N, B, ldb, C, ldc, n_threads);
        return;
    }
    const int tiles_m = (a->M + MB - 1) / MB, tiles_n = (N + NB - 1) / NB;
    neon_ctx ctx = {a, N, ldb, ldc, tiles_n, B, C};
    zc_parallel_for(tiles_m * tiles_n, n_threads, gemm_tile, &ctx);
}

// y += A x over the packed panels; 4-way unrolled over K to hide FMA latency.
void zc_gemv(const zc_mat * a, const float * x, float * y) {
    if (!a->packed) {
        zc_ref_gemv(a, x, y);
        return;
    }
    const int K = a->K;
    for (int pi = 0; pi < a->M / MR; pi++) {
        const float * p = (const float *) a->packed + (size_t) pi * K * MR;
        float32x4_t lo[4] = {vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0)};
        float32x4_t hi[4] = {vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0)};
        int k = 0;
        for (; k + 4 <= K; k += 4) {
            const float32x4_t xv = vld1q_f32(x + k);
            lo[0] = vfmaq_laneq_f32(lo[0], vld1q_f32(p + 0),  xv, 0);
            hi[0] = vfmaq_laneq_f32(hi[0], vld1q_f32(p + 4),  xv, 0);
            lo[1] = vfmaq_laneq_f32(lo[1], vld1q_f32(p + 8),  xv, 1);
            hi[1] = vfmaq_laneq_f32(hi[1], vld1q_f32(p + 12), xv, 1);
            lo[2] = vfmaq_laneq_f32(lo[2], vld1q_f32(p + 16), xv, 2);
            hi[2] = vfmaq_laneq_f32(hi[2], vld1q_f32(p + 20), xv, 2);
            lo[3] = vfmaq_laneq_f32(lo[3], vld1q_f32(p + 24), xv, 3);
            hi[3] = vfmaq_laneq_f32(hi[3], vld1q_f32(p + 28), xv, 3);
            p += 4 * MR;
        }
        for (; k < K; k++) {
            const float32x4_t xv = vdupq_n_f32(x[k]);
            lo[0] = vfmaq_f32(lo[0], vld1q_f32(p), xv);
            hi[0] = vfmaq_f32(hi[0], vld1q_f32(p + 4), xv);
            p += MR;
        }
        const float32x4_t l = vaddq_f32(vaddq_f32(lo[0], lo[1]), vaddq_f32(lo[2], lo[3]));
        const float32x4_t h = vaddq_f32(vaddq_f32(hi[0], hi[1]), vaddq_f32(hi[2], hi[3]));
        vst1q_f32(y + pi * MR,     vaddq_f32(vld1q_f32(y + pi * MR), l));
        vst1q_f32(y + pi * MR + 4, vaddq_f32(vld1q_f32(y + pi * MR + 4), h));
    }
}

const char * zc_backend_name(void) { return "neon"; }

#endif
