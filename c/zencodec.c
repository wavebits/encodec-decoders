// zencodec: minimal, dependency-free EnCodec (24 kHz) decoder in C17.
//
// Architecture (SEANet decoder, causal, reflect padding), activations are [C][T]:
//   RVQ decode            sum of n_q codebook embeddings          -> [128][T]
//   conv k7               128 -> 512
//   2-layer LSTM + skip   512
//   4x { ELU, conv-transpose (stride r), residual block }  r = 8, 5, 4, 2
//        residual block: conv1x1(x) + conv1x1(ELU(conv k3(ELU(x))))
//   ELU, conv k7          32 -> 1                                  -> [1][T * 320]

#include "zencodec.h"

#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

typedef struct { const tensor * w, * b; } conv;

typedef struct { const tensor * w_ih, * w_hh, * b_ih, * b_hh; } lstm_layer;

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

static float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

static void elu(act * x) {
    const size_t n = (size_t) x->C * x->T;
    for (size_t i = 0; i < n; i++) x->d[i] = x->d[i] > 0.0f ? x->d[i] : expm1f(x->d[i]);
}

// ---------------------------------------------------------------------------
// parallel_for over [0, n) using pthreads

typedef void (*task_fn)(void * ctx, int i);

typedef struct {
    task_fn         fn;
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

static void parallel_for(int n, int n_threads, task_fn fn, void * ctx) {
    if (n_threads > n) n_threads = n;
    if (n_threads <= 1) {
        for (int i = 0; i < n; i++) fn(ctx, i);
        return;
    }
    task_pool p = {fn, ctx, n, 0, PTHREAD_MUTEX_INITIALIZER};
    pthread_t threads[64];
    if (n_threads > 64) n_threads = 64;
    int spawned = 0;
    for (int t = 1; t < n_threads; t++) {
        if (pthread_create(&threads[spawned], NULL, pool_worker, &p) == 0) spawned++;
    }
    pool_worker(&p);
    for (int t = 0; t < spawned; t++) pthread_join(threads[t], NULL);
    pthread_mutex_destroy(&p.lock);
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

typedef struct {
    const act  * in;  // padded input
    act        * out;
    const conv * cv;
    int          C_in, K;
} conv1d_ctx;

static void conv1d_task(void * p, int co) {
    const conv1d_ctx * c = p;
    const int T = c->out->T;
    float * dst = row(c->out, co);
    const float b = c->cv->b->data[co];
    for (int t = 0; t < T; t++) dst[t] = b;
    for (int ci = 0; ci < c->C_in; ci++) {
        const float * wk  = c->cv->w->data + ((size_t) co * c->C_in + ci) * c->K;
        const float * src = row(c->in, ci);
        for (int k = 0; k < c->K; k++) {
            const float wv = wk[k];
            const float * s = src + k;
            for (int t = 0; t < T; t++) dst[t] += wv * s[t];
        }
    }
}

// Causal stride-1 conv1d. w: [C_out][C_in][K].
// `hist` carries the last K - 1 input samples from one chunk to the next. On the
// first chunk (hist->d == NULL) the input is reflect-padded instead, as in encodec.
// Unused (may be NULL) when K == 1.
static act conv1d(const act * x, const conv * cv, act * hist, int n_threads) {
    const int C_out = cv->w->shape[0], C_in = cv->w->shape[1], K = cv->w->shape[2];

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

    act out = act_new(C_out, x->T);
    conv1d_ctx ctx = {K > 1 ? &padded : x, &out, cv, C_in, K};
    parallel_for(C_out, n_threads, conv1d_task, &ctx);

    act_free(&padded);
    return out;
}

typedef struct {
    const act  * x;
    act        * out;
    act        * tail;
    const conv * cv;
    int          C_in, C_out, K, stride, L;
} convtr_ctx;

static void conv_transpose1d_task(void * p, int co) {
    const convtr_ctx * c = p;
    const int T = c->x->T, s = c->stride, L = c->L, n_tail = c->K - s;
    float * acc = calloc((size_t) s * L, sizeof(float));
    for (int ci = 0; ci < c->C_in; ci++) {
        const float * wk  = c->cv->w->data + ((size_t) ci * c->C_out + co) * c->K;
        const float * src = row(c->x, ci);
        for (int k = 0; k < c->K; k++) {
            const float wv = wk[k];
            float * a = acc + (size_t) (k % s) * L + k / s;
            for (int t = 0; t < T; t++) a[t] += wv * src[t];
        }
    }
    float * dst  = row(c->out, co);
    float * tail = row(c->tail, co);
    const float b = c->cv->b->data[co];
    for (int r = 0; r < s; r++) {
        const float * a = acc + (size_t) r * L;
        for (int t = 0; t < T; t++) dst[t * s + r] = a[t] + b;
    }
    // add the previous chunk's spill-over, then keep this chunk's
    for (int i = 0; i < n_tail; i++) dst[i] += tail[i];
    for (int i = 0; i < n_tail; i++) {
        const int pos = T * s + i;
        tail[i] = acc[(size_t) (pos % s) * L + pos / s];
    }
    free(acc);
}

// Causal conv-transpose1d. w: [C_in][C_out][K].
// Each chunk's last K - stride output samples also depend on the next chunk, so
// they are kept in `tail` ([C_out][K - stride], zeros at the start) and added to
// the next chunk's output; at the end of the stream they are dropped, which is
// encodec's causal trim.
// Computed in polyphase form: output sample t * stride + r only receives taps
// k = r + j * stride from input t - j, so each tap is a contiguous axpy into
// phase r's buffer.
static act conv_transpose1d(const act * x, const conv * cv, act * tail, int stride, int n_threads) {
    const int C_in = cv->w->shape[0], C_out = cv->w->shape[1], K = cv->w->shape[2];

    act out = act_new(C_out, x->T * stride);
    convtr_ctx ctx = {x, &out, tail, cv, C_in, C_out, K, stride, x->T + (K + stride - 1) / stride};
    parallel_for(C_out, n_threads, conv_transpose1d_task, &ctx);
    return out;
}

typedef struct {
    const float      * x;
    float            * xg;
    const lstm_layer * l;
    int                H;
} lstm_ctx;

static void lstm_input_task(void * p, int t) {
    const lstm_ctx * c = p;
    const int H = c->H, G = 4 * H;
    const float * xt = c->x + (size_t) t * H;
    for (int g = 0; g < G; g++) {
        c->xg[(size_t) t * G + g] = c->l->b_ih->data[g] + c->l->b_hh->data[g] +
                                    dot(c->l->w_ih->data + (size_t) g * H, xt, H);
    }
}

// Single LSTM layer over x: [T][H] (time-major), writes y: [T][H].
// h and c ([H] each) carry the state across chunks.
static void lstm_forward(const float * x, float * y, int T, int H, const lstm_layer * l, float * h, float * c,
                         int n_threads) {
    const int G = 4 * H; // gates i, f, g, o

    // input projections for all timesteps at once
    float * xg = malloc((size_t) T * G * sizeof(float));
    lstm_ctx ctx = {x, xg, l, H};
    parallel_for(T, n_threads, lstm_input_task, &ctx);

    float * gates = malloc(G * sizeof(float));
    for (int t = 0; t < T; t++) {
        for (int g = 0; g < G; g++) {
            gates[g] = xg[(size_t) t * G + g] + dot(l->w_hh->data + (size_t) g * H, h, H);
        }
        for (int i = 0; i < H; i++) {
            const float ig = sigmoid(gates[i]);
            const float fg = sigmoid(gates[H + i]);
            const float gg = tanhf(gates[2 * H + i]);
            const float og = sigmoid(gates[3 * H + i]);
            c[i] = fg * c[i] + ig * gg;
            h[i] = og * tanhf(c[i]);
        }
        memcpy(y + (size_t) t * H, h, H * sizeof(float));
    }
    free(xg); free(gates);
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

// weight shape [a][b][k]; bias has n_out entries
static bool bind_conv(const zencodec_model * m, conv * cv, const char * prefix, int a, int b, int k, int n_out) {
    char name[128];
    snprintf(name, sizeof(name), "%s.weight", prefix);
    cv->w = get(m, name, (size_t) a * b * k);
    snprintf(name, sizeof(name), "%s.bias", prefix);
    cv->b = get(m, name, (size_t) n_out);
    return cv->w && cv->b && cv->w->n_dims == 3;
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
    if (!bind_conv(m, &m->init_conv, "decoder.model.0.conv.conv", ch, H, 7, ch)) return false;

    for (int l = 0; l < 2; l++) {
        lstm_layer * ll = &m->lstm[l];
        snprintf(name, sizeof(name), "decoder.model.1.lstm.weight_ih_l%d", l);
        ll->w_ih = get(m, name, (size_t) 4 * ch * ch);
        snprintf(name, sizeof(name), "decoder.model.1.lstm.weight_hh_l%d", l);
        ll->w_hh = get(m, name, (size_t) 4 * ch * ch);
        snprintf(name, sizeof(name), "decoder.model.1.lstm.bias_ih_l%d", l);
        ll->b_ih = get(m, name, (size_t) 4 * ch);
        snprintf(name, sizeof(name), "decoder.model.1.lstm.bias_hh_l%d", l);
        ll->b_hh = get(m, name, (size_t) 4 * ch);
        if (!ll->w_ih || !ll->w_hh || !ll->b_ih || !ll->b_hh) return false;
    }

    for (int i = 0; i < 4; i++) {
        const int up = 3 * (i + 1), res = up + 1, r = RATIOS[i], out = ch / 2;
        decoder_block * b = &m->blocks[i];
        snprintf(name, sizeof(name), "decoder.model.%d.convtr.convtr", up);
        if (!bind_conv(m, &b->upsample, name, ch, out, 2 * r, out)) return false;
        snprintf(name, sizeof(name), "decoder.model.%d.block.1.conv.conv", res);
        if (!bind_conv(m, &b->conv1, name, out / 2, out, 3, out / 2)) return false;
        snprintf(name, sizeof(name), "decoder.model.%d.block.3.conv.conv", res);
        if (!bind_conv(m, &b->conv2, name, out, out / 2, 1, out)) return false;
        snprintf(name, sizeof(name), "decoder.model.%d.shortcut.conv.conv", res);
        if (!bind_conv(m, &b->shortcut, name, out, out, 1, out)) return false;
        ch = out;
    }
    return bind_conv(m, &m->final_conv, "decoder.model.15.conv.conv", 1, ch, 7, 1);
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

void zencodec_free(zencodec_model * model) {
    if (!model) return;
    for (int i = 0; i < model->n_tensors; i++) free(model->tensors[i].data);
    free(model);
}

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
    s->lstm_state = calloc((size_t) 4 * m->init_conv.w->shape[0], sizeof(float));
    for (int i = 0; i < 4; i++) {
        const tensor * w = m->blocks[i].upsample.w; // [C_in][C_out][K]
        s->block_tail[i] = act_new(w->shape[1], w->shape[2] - RATIOS[i]);
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
    act x = act_new(H, T);
    for (int q = 0; q < s->n_q; q++) {
        const float * embed = m->codebooks[q]->data;
        for (int t = 0; t < T; t++) {
            const float * e = embed + (size_t) codes[(size_t) q * T + t] * H;
            for (int d = 0; d < H; d++) row(&x, d)[t] += e[d];
        }
    }

    act tmp = conv1d(&x, &m->init_conv, &s->init_hist, n_threads);
    act_free(&x);
    x = tmp;

    // 2-layer LSTM with skip connection (time-major internally)
    {
        const int C = x.C;
        float * seq = malloc((size_t) T * C * sizeof(float));
        float * y   = malloc((size_t) T * C * sizeof(float));
        for (int c = 0; c < C; c++)
            for (int t = 0; t < T; t++) seq[(size_t) t * C + c] = row(&x, c)[t];

        float * st = s->lstm_state;
        lstm_forward(seq, y, T, C, &m->lstm[0], st, st + C, n_threads);
        lstm_forward(y, seq, T, C, &m->lstm[1], st + 2 * C, st + 3 * C, n_threads);

        for (int c = 0; c < C; c++)
            for (int t = 0; t < T; t++) row(&x, c)[t] += seq[(size_t) t * C + c];
        free(seq);
        free(y);
    }

    for (int i = 0; i < 4; i++) {
        const decoder_block * b = &m->blocks[i];

        elu(&x);
        tmp = conv_transpose1d(&x, &b->upsample, &s->block_tail[i], RATIOS[i], n_threads);
        act_free(&x);
        x = tmp;

        act y = act_new(x.C, x.T);
        memcpy(y.d, x.d, (size_t) x.C * x.T * sizeof(float));
        elu(&y);
        tmp = conv1d(&y, &b->conv1, &s->block_hist[i], n_threads);
        act_free(&y);
        elu(&tmp);
        y = conv1d(&tmp, &b->conv2, NULL, n_threads);
        act_free(&tmp);

        tmp = conv1d(&x, &b->shortcut, NULL, n_threads);
        act_free(&x);
        x = tmp;
        for (size_t j = 0; j < (size_t) x.C * x.T; j++) x.d[j] += y.d[j];
        act_free(&y);
    }

    elu(&x);
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
