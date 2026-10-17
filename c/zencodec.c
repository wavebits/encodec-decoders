// zencodec: minimal, dependency-free EnCodec (24 kHz) decoder in C17.
//
// Architecture (SEANet decoder, causal, reflect padding), activations are [C][T]:
//   RVQ decode            sum of n_q codebook embeddings          -> [128][T]
//   conv k7               128 -> 512
//   2-layer LSTM + skip   512
//   4x { ELU, conv-transpose (stride r), residual block }  r = 8, 5, 4, 2
//        residual block: conv1x1(x) + conv1x1(ELU(conv k3(ELU(x))))
//   ELU, conv k7          32 -> 1                                  -> [1][T * 320]

#ifdef ZENCODEC_PROFILE
#define _POSIX_C_SOURCE 200809L // clock_gettime
#endif
#ifdef __APPLE__
#define _DARWIN_C_SOURCE // pthread QoS
#endif

#include "zencodec.h"
#include "backend.h"

#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define GGML_MAGIC  0x67676d6c
#define MAX_TENSORS 512
#define MAX_Q       64

static const int RATIOS[4] = {8, 5, 4, 2};

typedef struct {
    char    name[96];
    int     n_dims;
    int     shape[4]; // PyTorch order (outermost first)
    size_t  numel;
    float * data;
} tensor;

typedef struct {
    int     C, T;
    float * d;
} act;

// w is laid out for zc_gemm: K stacked [C_out][C_in] taps (conv), or one
// [K * C_out][C_in] matrix (conv-transpose)
typedef struct {
    zc_mat         w;
    const tensor * b;
    int            C_in, C_out, K;
} conv;

typedef struct {
    zc_mat         w_ih, w_hh; // [4H][H] each
    const tensor * b_ih, * b_hh;
} lstm_layer;

typedef struct {
    conv upsample; // conv-transpose
    conv conv1;    // k3, C -> C/2
    conv conv2;    // k1, C/2 -> C
    conv shortcut; // k1, C -> C
} decoder_block;

struct zencodec_model {
    int hidden_dim, n_bins, sample_rate;

    tensor tensors[MAX_TENSORS];
    int    n_tensors;

    const tensor * codebooks[MAX_Q]; // [n_bins][hidden_dim] each
    int            n_codebooks;
    conv           init_conv;
    lstm_layer     lstm[2];
    decoder_block  blocks[4];
    conv           final_conv;
};

// ---------------------------------------------------------------------------
// optional per-stage timing (-DZENCODEC_PROFILE), printed by zencodec_free

#ifdef ZENCODEC_PROFILE
enum { P_RVQ, P_CONV, P_LSTM_IN, P_LSTM_REC, P_CONVT, P_RES, P_ELU, P_N };
static const char * const prof_names[P_N] = {
    "rvq decode", "conv1d", "lstm input gemm", "lstm recurrence", "conv-transpose", "resblock add", "elu",
};
static double prof_ms[P_N];
static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}
#define PROF_BEGIN() const double prof_t0 = now_ms()
#define PROF_END(stage) (prof_ms[stage] += now_ms() - prof_t0)
#else
#define PROF_BEGIN() ((void) 0)
#define PROF_END(stage) ((void) 0)
#endif

// ---------------------------------------------------------------------------
// helpers

static float fp16_to_fp32(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else { // subnormal: renormalize
            exp = 127 - 15 + 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            bits = sign | (exp << 23) | ((mant & 0x3ff) << 13);
        }
    } else if (exp == 0x1f) {
        bits = sign | 0x7f800000 | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static act act_new(int C, int T) {
    act a = {C, T, calloc((size_t) C * T, sizeof(float))};
    return a;
}

static void act_free(act * a) { free(a->d); a->d = NULL; }

static float * row(const act * a, int c) { return a->d + (size_t) c * a->T; }

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

static inline float sigmoid(float x) { return 1.0f / (1.0f + exp_approx(-x)); }
static inline float tanh_approx(float x) { return 2.0f * sigmoid(2.0f * x) - 1.0f; }

static void elu_row(void * p, int c) {
    act * x = p;
    float * d = row(x, c);
    for (int i = 0; i < x->T; i++) {
        const float e = exp_approx(fminf(d[i], 0.0f)) - 1.0f;
        d[i] = d[i] > 0.0f ? d[i] : e;
    }
}

static void elu(act * x, int n_threads);

// ---------------------------------------------------------------------------
// parallel_for over [0, n) using pthreads

typedef struct {
    zc_task_fn      fn;
    void          * ctx;
    int             n;
    int             next;
    pthread_mutex_t lock;
} task_pool;

static void * pool_worker(void * arg) {
    task_pool * p = arg;
    for (;;) {
        pthread_mutex_lock(&p->lock);
        const int i = p->next++;
        pthread_mutex_unlock(&p->lock);
        if (i >= p->n) return NULL;
        p->fn(p->ctx, i);
    }
}

void zc_parallel_for(int n, int n_threads, zc_task_fn fn, void * ctx) {
    if (n_threads > n) n_threads = n;
    if (n_threads <= 1) {
        for (int i = 0; i < n; i++) fn(ctx, i);
        return;
    }
    task_pool p = {fn, ctx, n, 0, PTHREAD_MUTEX_INITIALIZER};
    pthread_t threads[64];
    if (n_threads > 64) n_threads = 64;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
#ifdef __APPLE__
    // run workers at the caller's QoS, so they get the same (performance or
    // efficiency) cores as the calling thread
    pthread_attr_set_qos_class_np(&attr, qos_class_self(), 0);
#endif
    int spawned = 0;
    for (int t = 1; t < n_threads; t++) {
        if (pthread_create(&threads[spawned], &attr, pool_worker, &p) == 0) spawned++;
    }
    pthread_attr_destroy(&attr);
    pool_worker(&p);
    for (int t = 0; t < spawned; t++) pthread_join(threads[t], NULL);
    pthread_mutex_destroy(&p.lock);
}

static void elu(act * x, int n_threads) {
    PROF_BEGIN();
    zc_parallel_for(x->C, n_threads, elu_row, x);
    PROF_END(P_ELU);
}

// ---------------------------------------------------------------------------
// layers

// encodec's pad1d in reflect mode, including its workaround for inputs
// shorter than the padding.
static act pad_reflect(const act * x, int left, int right) {
    const int max_pad = left > right ? left : right;
    const int extra   = x->T <= max_pad ? max_pad - x->T + 1 : 0;
    const int L       = x->T + extra; // zero-extended length

    act out = act_new(x->C, x->T + left + right);
    for (int c = 0; c < x->C; c++) {
        const float * src = row(x, c);
        float       * dst = row(&out, c);
        for (int i = 0; i < out.T; i++) {
            int j = i - left;
            if (j < 0)  j = -j;
            if (j >= L) j = 2 * (L - 1) - j;
            dst[i] = j < x->T ? src[j] : 0.0f;
        }
    }
    return out;
}

// Causal stride-1 conv1d.
// `hist` carries the last K - 1 input samples from one chunk to the next. On the
// first chunk (hist->d == NULL) the input is reflect-padded instead, as in encodec.
// Unused (may be NULL) when K == 1.
static act conv1d(const act * x, const conv * cv, act * hist, int n_threads) {
    PROF_BEGIN();
    const int C_in = cv->C_in, K = cv->K;

    act padded = {0};
    if (K > 1) {
        if (!hist->d) {
            padded = pad_reflect(x, K - 1, 0);
            *hist  = act_new(C_in, K - 1);
        } else {
            padded = act_new(C_in, x->T + K - 1);
            for (int c = 0; c < C_in; c++) {
                memcpy(row(&padded, c), row(hist, c), (size_t) (K - 1) * sizeof(float));
                memcpy(row(&padded, c) + K - 1, row(x, c), (size_t) x->T * sizeof(float));
            }
        }
        for (int c = 0; c < C_in; c++)
            memcpy(row(hist, c), row(&padded, c) + x->T, (size_t) (K - 1) * sizeof(float));
    }

    // out = bias, then out += sum over taps k of W_k * in[:, k : k + T]
    act out = act_new(cv->C_out, x->T);
    for (int c = 0; c < cv->C_out; c++)
        for (int t = 0; t < x->T; t++) row(&out, c)[t] = cv->b->data[c];
    const act * in = K > 1 ? &padded : x;
    zc_gemm(&cv->w, x->T, in->d, in->T, out.d, x->T, n_threads);

    act_free(&padded);
    PROF_END(P_CONV);
    return out;
}

typedef struct {
    const act * z;    // [K][C_out][T] tap products
    act       * out;  // [C_out][T * stride]
    act       * tail; // [C_out][K - stride]
    const conv * cv;
    int          stride;
} convtr_ctx;

// overlap-add of one output channel: tap k of input t lands on output t * stride + k
static void convtr_scatter(void * p, int co) {
    const convtr_ctx * c = p;
    const int T = c->z->T, s = c->stride, K = c->cv->K, C_out = c->cv->C_out, L = T * s;
    const int n_tail = K - s;
    float * dst  = row(c->out, co);
    float * tail = row(c->tail, co);
    float next[64] = {0}; // K - stride <= 64
    const float b = c->cv->b->data[co];
    for (int i = 0; i < L; i++) dst[i] = b;
    for (int k = 0; k < K; k++) {
        const float * z = row(c->z, k * C_out + co);
        for (int t = 0; t < T; t++) {
            const int o = t * s + k;
            if (o < L) dst[o] += z[t];
            else       next[o - L] += z[t];
        }
    }
    // add the previous chunk's spill-over, then keep this chunk's
    for (int i = 0; i < n_tail; i++) dst[i] += tail[i];
    memcpy(tail, next, (size_t) n_tail * sizeof(float));
}

// Causal conv-transpose1d.
// Each chunk's last K - stride output samples also depend on the next chunk, so
// they are kept in `tail` ([C_out][K - stride], zeros at the start) and added to
// the next chunk's output; at the end of the stream they are dropped, which is
// encodec's causal trim.
// Computed as one GEMM, z = W * x with W: [K * C_out][C_in], followed by an
// overlap-add of the K taps.
static act conv_transpose1d(const act * x, const conv * cv, act * tail, int stride, int n_threads) {
    PROF_BEGIN();
    act z = act_new(cv->K * cv->C_out, x->T);
    zc_gemm(&cv->w, x->T, x->d, x->T, z.d, x->T, n_threads);

    act out = act_new(cv->C_out, x->T * stride);
    convtr_ctx ctx = {&z, &out, tail, cv, stride};
    zc_parallel_for(cv->C_out, n_threads, convtr_scatter, &ctx);
    act_free(&z);
    PROF_END(P_CONVT);
    return out;
}

// Single LSTM layer, channel-major: x and y are [H][T].
// h and c ([H] each) carry the state across chunks.
static void lstm_forward(const act * x, act * y, const lstm_layer * l, float * h, float * c, int n_threads) {
    const int H = x->C, T = x->T, G = 4 * H; // gates i, f, g, o

    // input projections for all timesteps at once: xg = W_ih x + b_ih + b_hh, [4H][T]
    PROF_BEGIN();
    act xg = act_new(G, T);
    for (int g = 0; g < G; g++) {
        const float b = l->b_ih->data[g] + l->b_hh->data[g];
        for (int t = 0; t < T; t++) row(&xg, g)[t] = b;
    }
    zc_gemm(&l->w_ih, T, x->d, T, xg.d, T, n_threads);
    PROF_END(P_LSTM_IN);

    {
        PROF_BEGIN();
        float * gates = malloc(G * sizeof(float));
        for (int t = 0; t < T; t++) {
            for (int g = 0; g < G; g++) gates[g] = row(&xg, g)[t];
            zc_gemv(&l->w_hh, h, gates);
            for (int i = 0; i < H; i++) {
                const float ig = sigmoid(gates[i]);
                const float fg = sigmoid(gates[H + i]);
                const float gg = tanh_approx(gates[2 * H + i]);
                const float og = sigmoid(gates[3 * H + i]);
                c[i] = fg * c[i] + ig * gg;
                h[i] = og * tanh_approx(c[i]);
            }
            for (int i = 0; i < H; i++) row(y, i)[t] = h[i];
        }
        free(gates);
        PROF_END(P_LSTM_REC);
    }
    act_free(&xg);
}

// ---------------------------------------------------------------------------
// loading

static bool read_i32(FILE * f, int32_t * v) { return fread(v, sizeof(*v), 1, f) == 1; }

static bool starts_with(const char * s, const char * prefix) {
    return strncmp(s, prefix, strlen(prefix)) == 0;
}

static bool load_tensors(zencodec_model * m, FILE * f) {
    for (;;) {
        int32_t n_dims, name_len, ftype;
        if (!read_i32(f, &n_dims)) return true; // clean EOF
        if (!read_i32(f, &name_len) || !read_i32(f, &ftype)) return false;
        if (n_dims < 0 || n_dims > 4 || name_len <= 0 || name_len >= 96 || (ftype != 0 && ftype != 1)) return false;

        tensor t = {.n_dims = n_dims, .numel = 1};
        for (int i = n_dims - 1; i >= 0; i--) { // stored innermost first
            if (!read_i32(f, &t.shape[i]) || t.shape[i] <= 0) return false;
            t.numel *= t.shape[i];
        }
        if (fread(t.name, 1, name_len, f) != (size_t) name_len) return false;
        t.name[name_len] = '\0';

        const bool keep = starts_with(t.name, "decoder.") || starts_with(t.name, "quantizer.");
        const size_t nbytes = t.numel * (ftype ? 2 : 4);
        if (!keep) {
            if (fseek(f, (long) nbytes, SEEK_CUR) != 0) return false;
            continue;
        }
        if (m->n_tensors == MAX_TENSORS) return false;

        t.data = malloc(t.numel * sizeof(float));
        if (!t.data) return false;
        m->tensors[m->n_tensors++] = t; // owned by the model from here on
        if (ftype == 0) {
            if (fread(t.data, 4, t.numel, f) != t.numel) return false;
        } else {
            uint16_t * h = malloc(nbytes);
            const bool ok = h && fread(h, 2, t.numel, f) == t.numel;
            if (ok) for (size_t i = 0; i < t.numel; i++) t.data[i] = fp16_to_fp32(h[i]);
            free(h);
            if (!ok) return false;
        }
    }
}

static const tensor * find(const zencodec_model * m, const char * name) {
    for (int i = 0; i < m->n_tensors; i++) {
        if (strcmp(m->tensors[i].name, name) == 0) return &m->tensors[i];
    }
    return NULL;
}

static const tensor * get(const zencodec_model * m, const char * name, size_t numel) {
    const tensor * t = find(m, name);
    if (!t) {
        fprintf(stderr, "zencodec: missing tensor '%s'\n", name);
        return NULL;
    }
    if (numel && t->numel != numel) {
        fprintf(stderr, "zencodec: tensor '%s' has unexpected size\n", name);
        return NULL;
    }
    return t;
}

// Hands a weight matrix to the backend. Takes ownership of `data`.
static bool make_mat(zc_mat * mat, float * data, int taps, int M, int K) {
    *mat = (zc_mat) {M, K, taps, data, NULL};
    return data && zc_mat_prepare(mat);
}

// Takes a tensor's data out of the model (the matrices built from it replace it).
static float * take(zencodec_model * m, const tensor * t) {
    tensor * mt = &m->tensors[t - m->tensors];
    float * d = mt->data;
    mt->data = NULL;
    return d;
}

// conv1d: weight [C_out][C_in][K] -> K taps of [C_out][C_in]
// conv-transpose: weight [C_in][C_out][K] -> [K * C_out][C_in]
static bool bind_conv(zencodec_model * m, conv * cv, const char * prefix, int C_out, int C_in, int K,
                      bool transpose) {
    char name[128];
    snprintf(name, sizeof(name), "%s.weight", prefix);
    const tensor * w = get(m, name, (size_t) C_out * C_in * K);
    snprintf(name, sizeof(name), "%s.bias", prefix);
    cv->b = get(m, name, (size_t) C_out);
    if (!w || !cv->b || w->n_dims != 3) return false;
    cv->C_in = C_in, cv->C_out = C_out, cv->K = K;

    float * src = take(m, w);
    float * dst = malloc((size_t) C_out * C_in * K * sizeof(float));
    if (dst) {
        for (int co = 0; co < C_out; co++)
            for (int ci = 0; ci < C_in; ci++)
                for (int k = 0; k < K; k++) {
                    const float v = transpose ? src[((size_t) ci * C_out + co) * K + k]
                                              : src[((size_t) co * C_in + ci) * K + k];
                    dst[((size_t) k * C_out + co) * C_in + ci] = v;
                }
    }
    free(src);
    return transpose ? make_mat(&cv->w, dst, 1, K * C_out, C_in) : make_mat(&cv->w, dst, K, C_out, C_in);
}

static bool bind(zencodec_model * m) {
    const int H = m->hidden_dim;
    char name[128];

    for (int i = 0; i < MAX_Q; i++) {
        snprintf(name, sizeof(name), "quantizer.vq.layers.%d._codebook.embed", i);
        const tensor * t = find(m, name);
        if (!t) break;
        if (t->numel != (size_t) m->n_bins * H) return false;
        m->codebooks[m->n_codebooks++] = t;
    }
    if (m->n_codebooks == 0) {
        fprintf(stderr, "zencodec: no quantizer codebooks found\n");
        return false;
    }

    int ch = 512; // n_filters * 2^4
    if (!bind_conv(m, &m->init_conv, "decoder.model.0.conv.conv", ch, H, 7, false)) return false;

    for (int l = 0; l < 2; l++) {
        lstm_layer * ll = &m->lstm[l];
        snprintf(name, sizeof(name), "decoder.model.1.lstm.weight_ih_l%d", l);
        const tensor * w_ih = get(m, name, (size_t) 4 * ch * ch);
        snprintf(name, sizeof(name), "decoder.model.1.lstm.weight_hh_l%d", l);
        const tensor * w_hh = get(m, name, (size_t) 4 * ch * ch);
        snprintf(name, sizeof(name), "decoder.model.1.lstm.bias_ih_l%d", l);
        ll->b_ih = get(m, name, (size_t) 4 * ch);
        snprintf(name, sizeof(name), "decoder.model.1.lstm.bias_hh_l%d", l);
        ll->b_hh = get(m, name, (size_t) 4 * ch);
        if (!w_ih || !w_hh || !ll->b_ih || !ll->b_hh) return false;
        if (!make_mat(&ll->w_ih, take(m, w_ih), 1, 4 * ch, ch)) return false;
        if (!make_mat(&ll->w_hh, take(m, w_hh), 1, 4 * ch, ch)) return false;
    }

    for (int i = 0; i < 4; i++) {
        const int up = 3 * (i + 1), res = up + 1, r = RATIOS[i], out = ch / 2;
        decoder_block * b = &m->blocks[i];
        snprintf(name, sizeof(name), "decoder.model.%d.convtr.convtr", up);
        if (!bind_conv(m, &b->upsample, name, out, ch, 2 * r, true)) return false;
        snprintf(name, sizeof(name), "decoder.model.%d.block.1.conv.conv", res);
        if (!bind_conv(m, &b->conv1, name, out / 2, out, 3, false)) return false;
        snprintf(name, sizeof(name), "decoder.model.%d.block.3.conv.conv", res);
        if (!bind_conv(m, &b->conv2, name, out, out / 2, 1, false)) return false;
        snprintf(name, sizeof(name), "decoder.model.%d.shortcut.conv.conv", res);
        if (!bind_conv(m, &b->shortcut, name, out, out, 1, false)) return false;
        ch = out;
    }
    return bind_conv(m, &m->final_conv, "decoder.model.15.conv.conv", 1, ch, 7, false);
}

// ---------------------------------------------------------------------------
// public API

zencodec_model * zencodec_load(const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "zencodec: cannot open '%s'\n", path);
        return NULL;
    }

    zencodec_model * m = calloc(1, sizeof(*m));
    int32_t magic = 0, hp[9];
    bool ok = m && read_i32(f, &magic) && magic == GGML_MAGIC && fread(hp, sizeof(int32_t), 9, f) == 9;
    if (ok) {
        // in_channels, hidden_dim, n_filters, kernel_size, res_kernel_size, n_bins, bandwidth, sr, ftype
        m->hidden_dim  = hp[1];
        m->n_bins      = hp[5];
        m->sample_rate = hp[7];
        ok = hp[0] == 1 && hp[1] > 0 && hp[2] == 32 && hp[3] == 7 && hp[4] == 3 && hp[5] > 0;
    }
    ok = ok && load_tensors(m, f) && bind(m);
    fclose(f);

    if (!ok) {
        fprintf(stderr, "zencodec: failed to load '%s'\n", path);
        zencodec_free(m);
        return NULL;
    }
    return m;
}

static void free_conv(conv * cv) { zc_mat_free(&cv->w); }

void zencodec_free(zencodec_model * model) {
    if (!model) return;
#ifdef ZENCODEC_PROFILE
    double total = 0;
    for (int i = 0; i < P_N; i++) total += prof_ms[i];
    fprintf(stderr, "zencodec profile (%s backend):\n", zc_backend_name());
    for (int i = 0; i < P_N; i++)
        fprintf(stderr, "  %-16s %9.1f ms  %5.1f%%\n", prof_names[i], prof_ms[i], total > 0 ? 100 * prof_ms[i] / total : 0);
#endif
    free_conv(&model->init_conv);
    free_conv(&model->final_conv);
    for (int l = 0; l < 2; l++) {
        zc_mat_free(&model->lstm[l].w_ih);
        zc_mat_free(&model->lstm[l].w_hh);
    }
    for (int i = 0; i < 4; i++) {
        free_conv(&model->blocks[i].upsample);
        free_conv(&model->blocks[i].conv1);
        free_conv(&model->blocks[i].conv2);
        free_conv(&model->blocks[i].shortcut);
    }
    for (int i = 0; i < model->n_tensors; i++) free(model->tensors[i].data);
    free(model);
}

const char * zencodec_backend(void) { return zc_backend_name(); }

int zencodec_sample_rate(const zencodec_model * model) { return model->sample_rate; }
int zencodec_hop_length (const zencodec_model * model) { (void) model; return RATIOS[0] * RATIOS[1] * RATIOS[2] * RATIOS[3]; }
int zencodec_n_codebooks(const zencodec_model * model) { return model->n_codebooks; }

// ---------------------------------------------------------------------------
// streaming decoder

struct zencodec_stream {
    const zencodec_model * m;
    int n_q, n_threads;

    // state carried from one chunk to the next (see conv1d, conv_transpose1d)
    act     init_hist, block_hist[4], final_hist;
    act     block_tail[4];
    float * lstm_state; // h, c for each of the 2 layers, [4][512]
};

zencodec_stream * zencodec_stream_new(const zencodec_model * m, int n_q, int n_threads) {
    if (n_q < 1 || n_q > m->n_codebooks) return NULL;
    zencodec_stream * s = calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->m = m;
    s->n_q = n_q;
    s->n_threads = n_threads < 1 ? 1 : n_threads;
    s->lstm_state = calloc((size_t) 4 * m->init_conv.C_out, sizeof(float));
    for (int i = 0; i < 4; i++) {
        const conv * up = &m->blocks[i].upsample;
        s->block_tail[i] = act_new(up->C_out, up->K - RATIOS[i]);
    }
    return s;
}

void zencodec_stream_free(zencodec_stream * s) {
    if (!s) return;
    act_free(&s->init_hist);
    act_free(&s->final_hist);
    for (int i = 0; i < 4; i++) {
        act_free(&s->block_hist[i]);
        act_free(&s->block_tail[i]);
    }
    free(s->lstm_state);
    free(s);
}

int zencodec_stream_decode(zencodec_stream * s, const int32_t * codes, int n_frames, float * audio) {
    const zencodec_model * m = s->m;
    const int H = m->hidden_dim, T = n_frames, n_threads = s->n_threads;

    if (T < 1) return 1;
    for (size_t i = 0; i < (size_t) s->n_q * T; i++) {
        if (codes[i] < 0 || codes[i] >= m->n_bins) return 2;
    }

    // residual vector quantizer: sum the selected embeddings
    PROF_BEGIN();
    act x = act_new(H, T);
    for (int q = 0; q < s->n_q; q++) {
        const float * embed = m->codebooks[q]->data;
        for (int t = 0; t < T; t++) {
            const float * e = embed + (size_t) codes[(size_t) q * T + t] * H;
            for (int d = 0; d < H; d++) row(&x, d)[t] += e[d];
        }
    }

    PROF_END(P_RVQ);
    act tmp = conv1d(&x, &m->init_conv, &s->init_hist, n_threads);
    act_free(&x);
    x = tmp;

    // 2-layer LSTM with skip connection
    {
        const int C = x.C;
        act y1 = act_new(C, T), y2 = act_new(C, T);
        float * st = s->lstm_state;
        lstm_forward(&x,  &y1, &m->lstm[0], st,         st + C,     n_threads);
        lstm_forward(&y1, &y2, &m->lstm[1], st + 2 * C, st + 3 * C, n_threads);
        for (size_t j = 0; j < (size_t) C * T; j++) x.d[j] += y2.d[j];
        act_free(&y1);
        act_free(&y2);
    }

    for (int i = 0; i < 4; i++) {
        const decoder_block * b = &m->blocks[i];

        elu(&x, n_threads);
        tmp = conv_transpose1d(&x, &b->upsample, &s->block_tail[i], RATIOS[i], n_threads);
        act_free(&x);
        x = tmp;

        act y = act_new(x.C, x.T);
        memcpy(y.d, x.d, (size_t) x.C * x.T * sizeof(float));
        elu(&y, n_threads);
        tmp = conv1d(&y, &b->conv1, &s->block_hist[i], n_threads);
        act_free(&y);
        elu(&tmp, n_threads);
        y = conv1d(&tmp, &b->conv2, NULL, n_threads);
        act_free(&tmp);

        tmp = conv1d(&x, &b->shortcut, NULL, n_threads);
        act_free(&x);
        x = tmp;
        {
            PROF_BEGIN();
            for (size_t j = 0; j < (size_t) x.C * x.T; j++) x.d[j] += y.d[j];
            PROF_END(P_RES);
        }
        act_free(&y);
    }

    elu(&x, n_threads);
    tmp = conv1d(&x, &m->final_conv, &s->final_hist, n_threads);
    act_free(&x);

    memcpy(audio, tmp.d, (size_t) tmp.T * sizeof(float));
    act_free(&tmp);
    return 0;
}

int zencodec_decode(const zencodec_model * m, const int32_t * codes, int n_q, int n_frames,
                    float * audio, int n_threads) {
    if (n_q < 1 || n_q > m->n_codebooks || n_frames < 1) return 1;
    for (size_t i = 0; i < (size_t) n_q * n_frames; i++) {
        if (codes[i] < 0 || codes[i] >= m->n_bins) return 2;
    }

    zencodec_stream * s = zencodec_stream_new(m, n_q, n_threads);
    int32_t * chunk = malloc((size_t) n_q * ZENCODEC_CHUNK_FRAMES * sizeof(int32_t));
    int ret = s && chunk ? 0 : 3;
    for (int f = 0; f < n_frames && ret == 0; f += ZENCODEC_CHUNK_FRAMES) {
        const int n = n_frames - f < ZENCODEC_CHUNK_FRAMES ? n_frames - f : ZENCODEC_CHUNK_FRAMES;
        for (int q = 0; q < n_q; q++)
            memcpy(chunk + (size_t) q * n, codes + (size_t) q * n_frames + f, (size_t) n * sizeof(int32_t));
        ret = zencodec_stream_decode(s, chunk, n, audio + (size_t) f * zencodec_hop_length(m));
    }
    free(chunk);
    zencodec_stream_free(s);
    return ret;
}
