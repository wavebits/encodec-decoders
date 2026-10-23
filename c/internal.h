// Internals shared by the EnCodec decoder (zencodec.c) and the optional Vocos decoder
// (vocos.c): weight loading, activation buffers, math helpers and profiling.
#ifndef ZENCODEC_INTERNAL_H
#define ZENCODEC_INTERNAL_H

#include "backend.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_TENSORS 512

typedef struct {
    char        name[96];
    int         n_dims;
    int         shape[4]; // PyTorch order (outermost first)
    size_t      numel;
    zc_weight * data; // f32, or f16 with ZENCODEC_F16_WEIGHTS (matrices and codebooks)
    float     * f;    // 0- and 1-D tensors (biases) instead: always f32, they are tiny
} tensor;

// The tensors read from a weights file.
typedef struct {
    tensor t[MAX_TENSORS];
    int    n;
} zc_store;

// Reads tensor records until EOF, keeping those whose name starts with one of the
// NULL-terminated `prefixes`.
bool zc_store_load(zc_store * s, FILE * f, const char * const * prefixes);
void zc_store_free(zc_store * s);
const tensor * zc_find(const zc_store * s, const char * name);
// Like zc_find, but reports a missing tensor, or one without `numel` elements (0: any).
const tensor * zc_get(const zc_store * s, const char * name, size_t numel);
// Takes a tensor's data out of the store; the caller frees it.
zc_weight * zc_take(zc_store * s, const tensor * t);
// Hands a weight matrix to the backend. Takes ownership of `data`.
bool zc_make_mat(zc_mat * mat, zc_weight * data, int taps, int M, int K);

// Activations, channel-major: C rows of T samples.
typedef struct {
    int     C, T;
    float * d;
} act;

static inline act act_new(int C, int T) {
    act a = {C, T, calloc((size_t) C * T, sizeof(float))};
    return a;
}
static inline void    act_free(act * a) { free(a->d); a->d = NULL; }
static inline float * row(const act * a, int c) { return a->d + (size_t) c * a->T; }

// exp(x), within about 2 ulp of expf (Cephes coefficients). Branch-free, so loops
// over it vectorize; libm's expf/expm1f/tanhf calls don't.
static inline float exp_approx(float x) {
    x = fminf(fmaxf(x, -87.0f), 88.0f);
    const float n = floorf(x * 1.44269504f + 0.5f); // round(x / ln 2)
    const float r = x - n * 0.693359375f + n * 2.12194440e-4f; // x - n ln 2, in two parts
    float p = 1.9875691500e-4f;
    p = p * r + 1.3981999507e-3f;
    p = p * r + 8.3334519073e-3f;
    p = p * r + 4.1665795894e-2f;
    p = p * r + 1.6666665459e-1f;
    p = p * r + 5.0000001201e-1f;
    p = p * r * r + r + 1.0f;
    const int32_t bits = ((int32_t) n + 127) << 23; // 2^n
    float scale;
    memcpy(&scale, &bits, sizeof(scale));
    return p * scale;
}

// ---------------------------------------------------------------------------
// optional per-stage timing (-DZENCODEC_PROFILE), printed by zencodec_free

enum {
    P_RVQ, P_CONV, P_LSTM_IN, P_LSTM_REC, P_CONVT, P_RES, P_ELU,              // EnCodec
    P_V_EMBED, P_V_DWCONV, P_V_NORM, P_V_MLP, P_V_GELU, P_V_HEAD, P_V_ISTFT, // Vocos
    P_N
};

#ifdef ZENCODEC_PROFILE
extern double zc_prof_ms[P_N];
double zc_now_ms(void);
#define PROF_BEGIN() const double prof_t0 = zc_now_ms()
#define PROF_END(stage) (zc_prof_ms[stage] += zc_now_ms() - prof_t0)
#else
#define PROF_BEGIN() ((void) 0)
#define PROF_END(stage) ((void) 0)
#endif

// ---------------------------------------------------------------------------
// Vocos (vocos.c, compiled in with -DZENCODEC_VOCOS)

#define VOCOS_MAGIC 0x73636f76 // "vocs" in the file

typedef struct vocos_model  vocos_model;
typedef struct vocos_stream vocos_stream;

vocos_model * vocos_load(FILE * f); // f is positioned after the magic
void          vocos_free(vocos_model * m);
int           vocos_n_codebooks(const vocos_model * m);
bool          vocos_valid_n_q(const vocos_model * m, int n_q);

vocos_stream * vocos_stream_new(const vocos_model * m, int n_q);
void           vocos_stream_free(vocos_stream * s);
// Feeds n_frames of codes ([n_q][n_frames], already range-checked) and writes the
// samples that are final to `audio`; with `final`, also the rest. Returns the number
// of samples written, or -1 if out of memory.
int vocos_stream_push(vocos_stream * s, const int32_t * codes, int n_frames, bool final, float * audio,
                      zc_pool * pool);

#define VOCOS_LOOKAHEAD 29 // frames; see vocos.c

#endif // ZENCODEC_INTERNAL_H
