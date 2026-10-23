// Vocos decoder for EnCodec codes (https://github.com/gemelo-ai/vocos, checkpoint
// charactr/vocos-encodec-24khz). Compiled in with -DZENCODEC_VOCOS.
//
// Architecture, activations are [C][T] (T frames of 320 samples):
//   features            sum of n_q codebook embeddings                     -> [128][T]
//   embed conv k7       128 -> 384, then AdaLayerNorm
//   8x ConvNeXt block   x + gamma * pw2(GELU(pw1(AdaLayerNorm(dwconv k7(x)))))   384 -> 1152 -> 384
//   LayerNorm, linear   384 -> 1282: 641 log-magnitudes and 641 phases per frame
//   ISTFT               n_fft 1280, hop 320, Hann window, "same" padding     -> [T * 320]
// AdaLayerNorm is a LayerNorm whose scale and shift are picked by the bandwidth
// (n_q = 2, 4, 8, 16).
//
// Streaming: the convolutions are centered (3 frames each side, zeros past the ends of
// the stream). Each of the 9 keeps the last 6 frames of its input and outputs a frame
// once it has the 3 after it, so the backbone trails its input by 27 frames. The ISTFT
// keeps the overlap-add of the frames it has seen; a frame's audio is final 1.5 frames
// later. VOCOS_LOOKAHEAD (internal.h) is 27 + 1.5 rounded up. Nothing is recomputed,
// and the output equals a decode of the whole input at once.

#ifdef ZENCODEC_VOCOS

#include "internal.h"

#define IN_CH    128
#define DIM      384
#define INTER    1152
#define N_LAYERS 8
#define N_BINS   1024
#define MAX_Q    16
#define N_BW     4                       // bandwidths: n_q = 2, 4, 8, 16
#define KW       7                       // conv kernel width
#define PAD      (KW / 2)                // frames on each side
#define CTX      (KW - 1)                // input frames a conv keeps between chunks
#define N_FFT    1280
#define HOP      320
#define HALF     (N_FFT / 2)             // 640 = 5 * 2^7
#define N_FREQ   (HALF + 1)              // 641
#define TRIM     ((N_FFT - HOP) / 2)     // "same" padding: samples cut from the start
// The head's N_FFT + 2 = 1282 rows, padded with zeros to a multiple of 8, which the
// neon backend needs to use its own kernels.
#define HEAD_ROWS 1288

typedef struct {
    float re, im;
} cpx;

typedef struct {
    const float * dw_w;         // depthwise conv, [DIM][KW]
    const float * dw_b;         // [DIM]
    const float * scale, * shift; // AdaLayerNorm, [N_BW][DIM] each
    zc_mat        pw1, pw2;     // [INTER][DIM], [DIM][INTER]
    const float * b1, * b2;
    const float * gamma;        // layer scale, [DIM]
} convnext;

struct vocos_model {
    zc_store          store;
    const zc_weight * codebooks; // [MAX_Q * N_BINS][IN_CH]
    zc_mat            embed;     // KW taps of [DIM][IN_CH]
    const float     * embed_b;
    const float     * scale, * shift; // AdaLayerNorm after embed, [N_BW][DIM]
    convnext          blocks[N_LAYERS];
    const float     * final_w, * final_b; // LayerNorm, [DIM]
    zc_mat            head;      // [HEAD_ROWS][DIM]: N_FFT + 2 rows, then zero rows
    float             head_b[HEAD_ROWS];

    float window[N_FFT];  // periodic Hann
    float synth[N_FFT];   // window / N_FFT: the inverse FFT's scale folded in
    cpx   tw_fft[HALF];   // e^(+2 pi i k / HALF)
    cpx   tw_real[HALF];  // e^(+2 pi i k / N_FFT)
};

// ---------------------------------------------------------------------------
// math

// erf(x), within 1.5e-7 (Abramowitz and Stegun 7.1.26). Branch-free, so it vectorizes.
static inline float erf_approx(float x) {
    const float a = fabsf(x);
    const float t = 1.0f / (1.0f + 0.3275911f * a);
    const float p = ((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t - 0.284496736f) * t
                     + 0.254829592f) * t;
    return copysignf(1.0f - p * exp_approx(-a * a), x);
}

static inline cpx cmul(cpx a, cpx b) { return (cpx) {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re}; }

// out[k] = sum over j of in[j * stride] e^(+2 pi i j k / n), for n = HALF / stride.
// Radix 2 down to n = 5, then a direct 5-point DFT. tw[m] = e^(+2 pi i m / HALF), so
// the twiddles of an n-point transform are every stride-th entry.
static void fft(const cpx * in, cpx * out, int n, int stride, const cpx * tw) {
    if (n == 5) {
        for (int k = 0; k < 5; k++) {
            cpx acc = {0, 0};
            for (int j = 0; j < 5; j++) {
                const cpx v = cmul(in[j * stride], tw[(j * k % 5) * stride]);
                acc.re += v.re;
                acc.im += v.im;
            }
            out[k] = acc;
        }
        return;
    }
    const int h = n / 2;
    fft(in, out, h, 2 * stride, tw);
    fft(in + stride, out + h, h, 2 * stride, tw);
    for (int k = 0; k < h; k++) {
        const cpx e = out[k], o = cmul(out[k + h], tw[k * stride]);
        out[k]     = (cpx) {e.re + o.re, e.im + o.im};
        out[k + h] = (cpx) {e.re - o.re, e.im - o.im};
    }
}

// Inverse real FFT of N_FFT points from bins X[0..HALF], times N_FFT (like an
// unnormalized inverse DFT). The imaginary parts of X[0] and X[HALF] are ignored, as in
// torch.fft.irfft. Uses one HALF-point complex FFT: with A[k] = X[k] + X[k + HALF] and
// B[k] = (X[k] - X[k + HALF]) e^(2 pi i k / N_FFT), where X[k + HALF] = conj(X[HALF - k]),
// the inverse transform of A + iB holds the even samples in its real part and the odd
// ones in its imaginary part.
static void irfft(const vocos_model * m, cpx * X, float * x) {
    cpx Z[HALF], z[HALF];
    X[0].im = X[HALF].im = 0.0f;
    for (int k = 0; k < HALF; k++) {
        const cpx a = X[k], b = {X[HALF - k].re, -X[HALF - k].im};
        const cpx s = {a.re + b.re, a.im + b.im};
        const cpx d = cmul((cpx) {a.re - b.re, a.im - b.im}, m->tw_real[k]);
        Z[k] = (cpx) {s.re - d.im, s.im + d.re};
    }
    fft(Z, z, HALF, 1, m->tw_fft);
    for (int k = 0; k < HALF; k++) {
        x[2 * k]     = z[k].re;
        x[2 * k + 1] = z[k].im;
    }
}

// ---------------------------------------------------------------------------
// layers

// AdaLayerNorm / LayerNorm over the channels of each frame (column) of x, in place:
// x = (x - mean) / sqrt(var + 1e-6) * w + b. Tasks take 16 columns each, so the loops
// over a row vectorize.
#define NORM_COLS 16
typedef struct {
    act         * x;
    const float * w, * b;
} norm_ctx;

static void norm_task(void * p, int i) {
    const norm_ctx * c = p;
    const int t0 = i * NORM_COLS, n = c->x->T - t0 < NORM_COLS ? c->x->T - t0 : NORM_COLS;
    float mean[NORM_COLS] = {0}, var[NORM_COLS] = {0};
    for (int ch = 0; ch < c->x->C; ch++) {
        const float * r = row(c->x, ch) + t0;
        for (int t = 0; t < n; t++) mean[t] += r[t];
    }
    for (int t = 0; t < n; t++) mean[t] /= (float) c->x->C;
    for (int ch = 0; ch < c->x->C; ch++) {
        const float * r = row(c->x, ch) + t0;
        for (int t = 0; t < n; t++) var[t] += (r[t] - mean[t]) * (r[t] - mean[t]);
    }
    for (int t = 0; t < n; t++) var[t] = 1.0f / sqrtf(var[t] / (float) c->x->C + 1e-6f);
    for (int ch = 0; ch < c->x->C; ch++) {
        float * r = row(c->x, ch) + t0;
        for (int t = 0; t < n; t++) r[t] = (r[t] - mean[t]) * var[t] * c->w[ch] + c->b[ch];
    }
}

static void layer_norm(act * x, const float * w, const float * b, zc_pool * pool) {
    PROF_BEGIN();
    norm_ctx ctx = {x, w, b};
    zc_pool_run(pool, (x->T + NORM_COLS - 1) / NORM_COLS, norm_task, &ctx);
    PROF_END(P_V_NORM);
}

typedef struct {
    const act      * in;  // [DIM][T + CTX]
    act            * out; // [DIM][T]
    const convnext * l;
} dw_ctx;

static void dwconv_row(void * p, int c) {
    const dw_ctx * d = p;
    const float * x = row(d->in, c), * w = d->l->dw_w + c * KW;
    float * y = row(d->out, c);
    for (int t = 0; t < d->out->T; t++) y[t] = d->l->dw_b[c];
    for (int k = 0; k < KW; k++)
        for (int t = 0; t < d->out->T; t++) y[t] += w[k] * x[t + k];
}

static void gelu_row(void * p, int c) {
    act * x = p;
    float * d = row(x, c);
    for (int t = 0; t < x->T; t++) d[t] = 0.5f * d[t] * (1.0f + erf_approx(d[t] * 0.70710678f));
}

// out = bias, then out += W * in (taps == 1) or the KW-tap conv of in (taps == KW)
static act linear(const zc_mat * w, const float * bias, const act * in, int T, zc_pool * pool) {
    act out = act_new(w->M, T);
    for (int c = 0; c < w->M; c++)
        for (int t = 0; t < T; t++) row(&out, c)[t] = bias[c];
    if (T > 0) zc_gemm(w, T, in->d, in->T, out.d, T, pool);
    return out;
}

// The ISTFT of one frame: log-magnitudes and phases -> windowed N_FFT samples.
typedef struct {
    const vocos_model * m;
    const act         * spec;   // [HEAD_ROWS][T]: log-magnitudes, phases, padding
    float             * frames; // [T][N_FFT]
} istft_ctx;

static void istft_frame(void * p, int t) {
    const istft_ctx * c = p;
    cpx X[N_FREQ];
    for (int k = 0; k < N_FREQ; k++) {
        const float mag = fminf(exp_approx(row(c->spec, k)[t]), 100.0f);
        const float ph  = row(c->spec, N_FREQ + k)[t];
        X[k] = (cpx) {mag * cosf(ph), mag * sinf(ph)};
    }
    float * y = c->frames + (size_t) t * N_FFT;
    irfft(c->m, X, y);
    for (int n = 0; n < N_FFT; n++) y[n] *= c->m->synth[n];
}

// ---------------------------------------------------------------------------
// loading

static const float * get_f(const zc_store * s, const char * name, size_t numel) {
    const tensor * t = zc_get(s, name, numel);
    return t ? t->f : NULL;
}

static bool load_block(vocos_model * m, convnext * b, int i) {
    char name[96];
#define NAME(suffix) (snprintf(name, sizeof(name), "backbone.convnext.%d.%s", i, suffix), name)
    b->dw_w  = get_f(&m->store, NAME("dwconv.weight"), (size_t) DIM * KW);
    b->dw_b  = get_f(&m->store, NAME("dwconv.bias"), DIM);
    b->scale = get_f(&m->store, NAME("norm.scale.weight"), (size_t) N_BW * DIM);
    b->shift = get_f(&m->store, NAME("norm.shift.weight"), (size_t) N_BW * DIM);
    b->b1    = get_f(&m->store, NAME("pwconv1.bias"), INTER);
    b->b2    = get_f(&m->store, NAME("pwconv2.bias"), DIM);
    b->gamma = get_f(&m->store, NAME("gamma"), DIM);
    const tensor * w1 = zc_get(&m->store, NAME("pwconv1.weight"), (size_t) INTER * DIM);
    const tensor * w2 = zc_get(&m->store, NAME("pwconv2.weight"), (size_t) DIM * INTER);
#undef NAME
    if (!b->dw_w || !b->dw_b || !b->scale || !b->shift || !b->b1 || !b->b2 || !b->gamma || !w1 || !w2 ||
        !w1->data || !w2->data)
        return false;
    return zc_make_mat(&b->pw1, zc_take(&m->store, w1), 1, INTER, DIM) &&
           zc_make_mat(&b->pw2, zc_take(&m->store, w2), 1, DIM, INTER);
}

static bool bind(vocos_model * m) {
    const tensor * cb = zc_get(&m->store, "feature_extractor.codebook_weights", (size_t) MAX_Q * N_BINS * IN_CH);
    const tensor * ew = zc_get(&m->store, "backbone.embed.weight", (size_t) DIM * IN_CH * KW);
    const tensor * hw = zc_get(&m->store, "head.out.weight", (size_t) (N_FFT + 2) * DIM);
    m->embed_b = get_f(&m->store, "backbone.embed.bias", DIM);
    m->scale   = get_f(&m->store, "backbone.norm.scale.weight", (size_t) N_BW * DIM);
    m->shift   = get_f(&m->store, "backbone.norm.shift.weight", (size_t) N_BW * DIM);
    m->final_w = get_f(&m->store, "backbone.final_layer_norm.weight", DIM);
    m->final_b = get_f(&m->store, "backbone.final_layer_norm.bias", DIM);
    const float * hb = get_f(&m->store, "head.out.bias", N_FFT + 2);
    if (!cb || !cb->data || !ew || !ew->data || !hw || !hw->data || !m->embed_b || !m->scale || !m->shift ||
        !m->final_w || !m->final_b || !hb)
        return false;
    m->codebooks = cb->data;

    // embed conv: [DIM][IN_CH][KW] -> KW taps of [DIM][IN_CH]
    zc_weight * src = zc_take(&m->store, ew);
    zc_weight * dst = malloc((size_t) DIM * IN_CH * KW * sizeof(zc_weight));
    if (dst) {
        for (int co = 0; co < DIM; co++)
            for (int ci = 0; ci < IN_CH; ci++)
                for (int k = 0; k < KW; k++)
                    dst[((size_t) k * DIM + co) * IN_CH + ci] = src[((size_t) co * IN_CH + ci) * KW + k];
    }
    free(src);
    if (!zc_make_mat(&m->embed, dst, KW, DIM, IN_CH)) return false;
    memcpy(m->head_b, hb, (N_FFT + 2) * sizeof(float));
    zc_weight * head = calloc((size_t) HEAD_ROWS * DIM, sizeof(zc_weight)); // zero is 0x0000 in f16 too
    src = zc_take(&m->store, hw);
    if (head) memcpy(head, src, (size_t) (N_FFT + 2) * DIM * sizeof(zc_weight));
    free(src);
    if (!zc_make_mat(&m->head, head, 1, HEAD_ROWS, DIM)) return false;
    for (int i = 0; i < N_LAYERS; i++) {
        if (!load_block(m, &m->blocks[i], i)) return false;
    }

    const double pi = 3.14159265358979323846;
    for (int n = 0; n < N_FFT; n++) {
        m->window[n] = (float) (0.5 - 0.5 * cos(2 * pi * n / N_FFT));
        m->synth[n]  = m->window[n] / N_FFT;
    }
    for (int k = 0; k < HALF; k++) {
        m->tw_fft[k]  = (cpx) {(float) cos(2 * pi * k / HALF), (float) sin(2 * pi * k / HALF)};
        m->tw_real[k] = (cpx) {(float) cos(2 * pi * k / N_FFT), (float) sin(2 * pi * k / N_FFT)};
    }
    return true;
}

vocos_model * vocos_load(FILE * f) {
    // input_channels, dim, intermediate_dim, num_layers, n_fft, hop_length, n_bins, n_q, sample_rate, ftype
    int32_t hp[10];
    static const int32_t expect[9] = {IN_CH, DIM, INTER, N_LAYERS, N_FFT, HOP, N_BINS, MAX_Q, 24000};
    if (fread(hp, sizeof(int32_t), 10, f) != 10 || memcmp(hp, expect, sizeof(expect)) != 0) {
        fprintf(stderr, "zencodec: unsupported Vocos model (only vocos-encodec-24khz)\n");
        return NULL;
    }
    vocos_model * m = calloc(1, sizeof(*m));
    static const char * const prefixes[] = {"backbone.", "head.out.", "feature_extractor.codebook_weights", NULL};
    if (m && (!zc_store_load(&m->store, f, prefixes) || !bind(m))) {
        vocos_free(m);
        return NULL;
    }
    return m;
}

void vocos_free(vocos_model * m) {
    if (!m) return;
    zc_mat_free(&m->embed);
    zc_mat_free(&m->head);
    for (int i = 0; i < N_LAYERS; i++) {
        zc_mat_free(&m->blocks[i].pw1);
        zc_mat_free(&m->blocks[i].pw2);
    }
    zc_store_free(&m->store);
    free(m);
}

int vocos_n_codebooks(const vocos_model * m) { (void) m; return MAX_Q; }

// bandwidth id for AdaLayerNorm: n_q 2, 4, 8, 16 -> 0, 1, 2, 3; -1 otherwise
static int bandwidth_id(int n_q) {
    for (int i = 0; i < N_BW; i++) {
        if (n_q == 2 << i) return i;
    }
    return -1;
}

bool vocos_valid_n_q(const vocos_model * m, int n_q) { (void) m; return bandwidth_id(n_q) >= 0; }

// ---------------------------------------------------------------------------
// streaming

// A centered conv's input frames not yet fully used: at most CTX of them, [C][CTX].
typedef struct {
    act buf;
    int n;
} conv_hist;

struct vocos_stream {
    const vocos_model * m;
    int       n_q, bw;
    conv_hist hist[1 + N_LAYERS]; // embed conv, then each block's dwconv
    long long n_frames;           // frames pushed so far
    // ISTFT overlap-add of raw (untrimmed) samples [pos, pos + N_FFT), and the sum of
    // the squared windows over the same samples
    long long pos;
    float     ola[N_FFT], env[N_FFT];
};

vocos_stream * vocos_stream_new(const vocos_model * m, int n_q) {
    const int bw = bandwidth_id(n_q);
    if (bw < 0) return NULL;
    vocos_stream * s = calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->m = m, s->n_q = n_q, s->bw = bw;
    bool ok = true;
    for (int i = 0; i <= N_LAYERS; i++) {
        s->hist[i].buf = act_new(i == 0 ? IN_CH : DIM, CTX);
        s->hist[i].n   = PAD; // the zero padding before the first frame
        ok = ok && s->hist[i].buf.d;
    }
    if (!ok) {
        vocos_stream_free(s);
        return NULL;
    }
    return s;
}

void vocos_stream_free(vocos_stream * s) {
    if (!s) return;
    for (int i = 0; i <= N_LAYERS; i++) act_free(&s->hist[i].buf);
    free(s);
}

// The input of a centered conv for this chunk: the kept frames, x, and (at the end of the
// stream) PAD zeros. Output frame t of the conv is centered on frame t + PAD of it.
// Keeps the last CTX frames for the next chunk.
static act with_context(conv_hist * h, const act * x, bool final) {
    act in = act_new(h->buf.C, h->n + x->T + (final ? PAD : 0));
    if (!in.d) return in;
    for (int c = 0; c < in.C; c++) {
        memcpy(row(&in, c), row(&h->buf, c), (size_t) h->n * sizeof(float));
        if (x->T) memcpy(row(&in, c) + h->n, row(x, c), (size_t) x->T * sizeof(float));
    }
    const int keep = in.T < CTX ? in.T : CTX;
    for (int c = 0; c < in.C; c++)
        memcpy(row(&h->buf, c), row(&in, c) + in.T - keep, (size_t) keep * sizeof(float));
    h->n = keep;
    return in;
}

static int n_out(const act * in) { return in->T > CTX ? in->T - CTX : 0; }

// One ConvNeXt block on `in` (block input with context); returns its n_out(in) frames.
static act convnext_forward(const vocos_stream * s, const convnext * l, const act * in, zc_pool * pool) {
    const int T = n_out(in);
    act y = act_new(DIM, T), h, z;
    {
        PROF_BEGIN();
        dw_ctx dw = {in, &y, l};
        zc_pool_run(pool, DIM, dwconv_row, &dw);
        PROF_END(P_V_DWCONV);
    }
    layer_norm(&y, l->scale + s->bw * DIM, l->shift + s->bw * DIM, pool);
    {
        PROF_BEGIN();
        h = linear(&l->pw1, l->b1, &y, T, pool);
        PROF_END(P_V_MLP);
    }
    {
        PROF_BEGIN();
        zc_pool_run(pool, INTER, gelu_row, &h);
        PROF_END(P_V_GELU);
    }
    {
        PROF_BEGIN();
        z = linear(&l->pw2, l->b2, &h, T, pool);
        for (int c = 0; c < DIM; c++) { // residual: the input frame each output is centered on
            const float * r = row(in, c) + PAD;
            float * o = row(&z, c);
            for (int t = 0; t < T; t++) o[t] = r[t] + l->gamma[c] * o[t];
        }
        PROF_END(P_V_MLP);
    }
    act_free(&y);
    act_free(&h);
    return z;
}

// Writes the output samples that the overlap-add has completed. Output sample i is raw
// sample i + TRIM, divided by the window envelope there, as in vocos' ISTFT.
static int istft(vocos_stream * s, const act * spec, bool final, float * audio, zc_pool * pool) {
    const vocos_model * m = s->m;
    const int n_spec = spec->T;
    float * frames = malloc((size_t) (n_spec ? n_spec : 1) * N_FFT * sizeof(float));
    if (!frames) return -1;
    istft_ctx ctx = {m, spec, frames};
    zc_pool_run(pool, n_spec, istft_frame, &ctx);

    // after adding frame t, raw samples [pos, pos + HOP) are complete
    int written = 0;
    for (int t = 0; t < n_spec; t++) {
        const float * y = frames + (size_t) t * N_FFT;
        for (int n = 0; n < N_FFT; n++) {
            s->ola[n] += y[n];
            s->env[n] += m->window[n] * m->window[n];
        }
        for (int n = 0; n < HOP; n++) {
            if (s->pos + n >= TRIM) audio[written++] = s->ola[n] / s->env[n];
        }
        memmove(s->ola, s->ola + HOP, (N_FFT - HOP) * sizeof(float));
        memmove(s->env, s->env + HOP, (N_FFT - HOP) * sizeof(float));
        memset(s->ola + N_FFT - HOP, 0, HOP * sizeof(float));
        memset(s->env + N_FFT - HOP, 0, HOP * sizeof(float));
        s->pos += HOP;
    }
    if (final) { // the rest, up to n_frames * HOP output samples
        for (long long r = s->pos; r < s->n_frames * HOP + TRIM; r++) {
            if (r >= TRIM) audio[written++] = s->ola[r - s->pos] / s->env[r - s->pos];
        }
    }
    free(frames);
    return written;
}

int vocos_stream_push(vocos_stream * s, const int32_t * codes, int n_frames, bool final, float * audio,
                      zc_pool * pool) {
    const vocos_model * m = s->m;
    const int T = n_frames;
    s->n_frames += T;
    act x, in;

    {
        PROF_BEGIN();
        // features: sum of the codebook embeddings
        x = act_new(IN_CH, T);
        for (int q = 0; q < s->n_q && T; q++) {
            const zc_weight * cb = m->codebooks + (size_t) q * N_BINS * IN_CH;
            for (int t = 0; t < T; t++) {
                const zc_weight * e = cb + (size_t) codes[(size_t) q * T + t] * IN_CH;
                for (int d = 0; d < IN_CH; d++) row(&x, d)[t] += zc_w2f(e[d]);
            }
        }
        in = with_context(&s->hist[0], &x, final);
        act_free(&x);
        if (!in.d) return -1;
        x = linear(&m->embed, m->embed_b, &in, n_out(&in), pool);
        act_free(&in);
        PROF_END(P_V_EMBED);
    }
    layer_norm(&x, m->scale + s->bw * DIM, m->shift + s->bw * DIM, pool);

    for (int i = 0; i < N_LAYERS; i++) {
        in = with_context(&s->hist[i + 1], &x, final);
        act_free(&x);
        if (!in.d) return -1;
        x = convnext_forward(s, &m->blocks[i], &in, pool);
        act_free(&in);
    }
    layer_norm(&x, m->final_w, m->final_b, pool);

    act spec;
    {
        PROF_BEGIN();
        spec = linear(&m->head, m->head_b, &x, x.T, pool); // one spectrum per frame
        act_free(&x);
        PROF_END(P_V_HEAD);
    }
    PROF_BEGIN();
    const int written = istft(s, &spec, final, audio, pool);
    act_free(&spec);
    PROF_END(P_V_ISTFT);
    return written;
}

#endif // ZENCODEC_VOCOS
