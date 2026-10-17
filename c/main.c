// zencodec CLI (C17).
//
//   wb_decode-c decode <weights.bin> <codes.bin|in.ecdc> <out.wav> [-t n_threads]
//   wb_decode-c play   <weights.bin> <codes.bin|in.ecdc>           [-t n_threads]
//
// Codes file (little-endian): int32 n_q, int32 n_frames, int32 codes[n_q][n_frames]
// (see scripts/encode.py), or an .ecdc file from `encodec` compressed without --lm.

#if !defined(__APPLE__)
#define _DEFAULT_SOURCE // clock_gettime, sysconf(_SC_NPROCESSORS_ONLN)
#endif

#include "play.h"
#include "zencodec.h"

#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#ifdef __APPLE__
#include <pthread/qos.h>
#include <sys/sysctl.h>
#endif

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static uint8_t * read_file(const char * path, size_t * size) {
    FILE * f = fopen(path, "rb");
    if (!f) return NULL;
    uint8_t * buf = NULL;
    long len = fseek(f, 0, SEEK_END) == 0 ? ftell(f) : -1;
    if (len >= 0 && fseek(f, 0, SEEK_SET) == 0 && (buf = malloc(len > 0 ? (size_t) len : 1))) {
        if (fread(buf, 1, (size_t) len, f) != (size_t) len) {
            free(buf);
            buf = NULL;
        }
        *size = (size_t) len;
    }
    fclose(f);
    return buf;
}

// Finds `"key":` in a flat JSON object and returns a pointer to its value, or NULL.
static const char * json_value(const char * json, const char * key) {
    char pat[16];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char * p = strstr(json, pat);
    if (!p) return NULL;
    p += strlen(pat);
    while (*p == ' ') p++;
    if (*p++ != ':') return NULL;
    while (*p == ' ') p++;
    return p;
}

// .ecdc file from `encodec` (Python): "ECDC", u8 version 0, u32 big-endian JSON length,
// JSON metadata {"m", "al", "nc", "lm"}, then codes bit-packed LSB-first in frame-major
// order. Only encodec_24khz without the language model (lm = false) is supported.
static int32_t * parse_ecdc(const uint8_t * buf, size_t size, int32_t * n_q, int32_t * n_frames,
                            size_t * n_samples, int hop_length) {
    enum { bits_per_code = 10 }; // log2(1024 bins)
    if (size < 9 || buf[4] != 0) return NULL;
    const size_t json_len = (size_t) buf[5] << 24 | (size_t) buf[6] << 16 | (size_t) buf[7] << 8 | buf[8];
    if (json_len > size - 9 || json_len > 4096) return NULL;
    char json[4097];
    memcpy(json, buf + 9, json_len);
    json[json_len] = '\0';

    const char * m = json_value(json, "m");
    const char * al = json_value(json, "al");
    const char * nc = json_value(json, "nc");
    const char * lm = json_value(json, "lm");
    if (!m || strncmp(m, "\"encodec_24khz\"", 15) != 0 || !al || !nc) {
        fprintf(stderr, "error: only encodec_24khz .ecdc files are supported\n");
        return NULL;
    }
    if (!lm || strncmp(lm, "false", 5) != 0) {
        fprintf(stderr, "error: .ecdc files compressed with the language model (lm) are not supported\n");
        return NULL;
    }
    const long long n_al = atoll(al), n_nc = atoll(nc);
    if (n_al <= 0 || n_nc <= 0 || n_nc > 1024 || (n_al + hop_length - 1) / hop_length > (1 << 24)) return NULL;
    *n_q = (int32_t) n_nc;
    *n_frames = (int32_t) ((n_al + hop_length - 1) / hop_length);
    *n_samples = (size_t) n_al;

    const uint8_t * body = buf + 9 + json_len;
    const size_t body_len = size - 9 - json_len, n = (size_t) *n_q * *n_frames;
    if (body_len < (n * bits_per_code + 7) / 8) return NULL;
    int32_t * codes = malloc(n * sizeof(int32_t));
    if (!codes) return NULL;
    uint32_t acc = 0;
    int bits = 0;
    size_t i = 0;
    for (size_t b = 0; i < n; b++) {
        acc |= (uint32_t) body[b] << bits;
        bits += 8;
        for (; bits >= bits_per_code && i < n; i++, bits -= bits_per_code, acc >>= bits_per_code) {
            const size_t t = i / *n_q, k = i % *n_q; // frame-major -> [n_q][n_frames]
            codes[k * *n_frames + t] = (int32_t) (acc & ((1u << bits_per_code) - 1));
        }
    }
    return codes;
}

// Reads a codes file or an .ecdc file. n_samples is the audio length to keep.
static int32_t * read_codes(const char * path, int32_t * n_q, int32_t * n_frames, size_t * n_samples,
                            int hop_length) {
    size_t size;
    uint8_t * buf = read_file(path, &size);
    if (!buf) return NULL;
    int32_t * codes = NULL;
    if (size >= 4 && !memcmp(buf, "ECDC", 4)) {
        codes = parse_ecdc(buf, size, n_q, n_frames, n_samples, hop_length);
    } else if (size >= 8) {
        memcpy(n_q, buf, 4);
        memcpy(n_frames, buf + 4, 4);
        if (*n_q > 0 && *n_q <= 1024 && *n_frames > 0 && *n_frames <= (1 << 24)) {
            const size_t n = (size_t) *n_q * *n_frames;
            if (size - 8 >= n * 4 && (codes = malloc(n * sizeof(int32_t)))) memcpy(codes, buf + 8, n * 4);
            *n_samples = (size_t) *n_frames * hop_length;
        }
    }
    free(buf);
    return codes;
}

static void put_u32(FILE * f, uint32_t v) { fwrite(&v, 4, 1, f); }
static void put_u16(FILE * f, uint16_t v) { fwrite(&v, 2, 1, f); }

static bool write_wav(const char * path, const float * audio, size_t n, int sample_rate) {
    FILE * f = fopen(path, "wb");
    if (!f) return false;
    const uint32_t data_size = (uint32_t) n * 2;
    fwrite("RIFF", 1, 4, f); put_u32(f, 36 + data_size);
    fwrite("WAVEfmt ", 1, 8, f); put_u32(f, 16);
    put_u16(f, 1); put_u16(f, 1);                   // PCM, mono
    put_u32(f, sample_rate); put_u32(f, sample_rate * 2);
    put_u16(f, 2); put_u16(f, 16);                  // block align, bits per sample
    fwrite("data", 1, 4, f); put_u32(f, data_size);
    for (size_t i = 0; i < n; i++) {
        const float s = audio[i] < -1.0f ? -1.0f : audio[i] > 1.0f ? 1.0f : audio[i];
        const int16_t v = (int16_t) lrintf(s * 32767.0f);
        fwrite(&v, 2, 1, f);
    }
    return fclose(f) == 0;
}

// Decodes chunk by chunk on its own thread so playback can start after the first chunk.
typedef struct {
    const zencodec_model * model;
    const int32_t        * codes; // [n_q][n_frames]
    int                    n_q, n_frames, n_threads;
    float                * audio;
    size_t                 n;     // samples to keep
    atomic_size_t          ready; // samples decoded so far
    atomic_bool            done;
    int                    err;
    double                 t_done;
} decode_job;

static void * decode_job_run(void * arg) {
    decode_job * j = arg;
    const int hop = zencodec_hop_length(j->model), max_chunk = ZENCODEC_CHUNK_FRAMES;
    zencodec_stream * s = zencodec_stream_new(j->model, j->n_q, j->n_threads);
    int32_t * chunk = malloc((size_t) j->n_q * max_chunk * sizeof(int32_t));
    j->err = s && chunk ? 0 : 3;
    for (int f = 0; f < j->n_frames && j->err == 0; f += max_chunk) {
        const int n = j->n_frames - f < max_chunk ? j->n_frames - f : max_chunk;
        for (int q = 0; q < j->n_q; q++)
            memcpy(chunk + (size_t) q * n, j->codes + (size_t) q * j->n_frames + f, (size_t) n * sizeof(int32_t));
        j->err = zencodec_stream_decode(s, chunk, n, j->audio + (size_t) f * hop);
        const size_t done = (size_t) (f + n) * hop;
        atomic_store_explicit(&j->ready, done < j->n ? done : j->n, memory_order_release);
    }
    free(chunk);
    zencodec_stream_free(s);
    j->t_done = now_ms();
    atomic_store_explicit(&j->done, true, memory_order_release);
    return NULL;
}

static int usage(const char * argv0) {
    fprintf(stderr,
            "usage: %s decode <weights.bin> <codes.bin|in.ecdc> <out.wav> [-t n_threads]\n"
            "       %s play   <weights.bin> <codes.bin|in.ecdc>           [-t n_threads]\n",
            argv0, argv0);
    return 1;
}

// Default thread count: the performance cores on Apple silicon (efficiency cores
// only slow the other threads down), otherwise all online CPUs.
static int default_threads(void) {
#ifdef __APPLE__
    int n = 0;
    size_t len = sizeof(n);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &n, &len, NULL, 0) == 0 && n > 0) return n;
#endif
    const long n_cpu = sysconf(_SC_NPROCESSORS_ONLN);
    return n_cpu > 0 ? (int) n_cpu : 1;
}

int main(int argc, char ** argv) {
#ifdef __APPLE__
    // ask for performance cores; worker threads inherit this class
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
#endif
    int n_threads = default_threads();
    const char * pos[4];
    int n_pos = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-t") && i + 1 < argc) n_threads = atoi(argv[++i]);
        else if (n_pos < 4) pos[n_pos++] = argv[i];
        else return usage(argv[0]);
    }

    const bool play = n_pos == 3 && !strcmp(pos[0], "play");
    if (!play && !(n_pos == 4 && !strcmp(pos[0], "decode"))) return usage(argv[0]);

    const double t0 = now_ms();
    zencodec_model * model = zencodec_load(pos[1]);
    if (!model) return 1;
    const double t1 = now_ms();

    int32_t n_q, n_frames;
    size_t n;
    int32_t * codes = read_codes(pos[2], &n_q, &n_frames, &n, zencodec_hop_length(model));
    if (!codes) {
        fprintf(stderr, "error: cannot read codes from '%s'\n", pos[2]);
        return 1;
    }

    const int sr = zencodec_sample_rate(model);
    float * audio = calloc((size_t) n_frames * zencodec_hop_length(model), sizeof(float));
    if (!audio) return 1;
    int ret = 0;

    if (play) {
        decode_job job = {model, codes, n_q, n_frames, n_threads, audio, n, 0, false, 0, 0.0};
        pthread_t thread;
        if (pthread_create(&thread, NULL, decode_job_run, &job) != 0) return 1;
        const struct timespec tick = {0, 1000 * 1000};
        while (atomic_load_explicit(&job.ready, memory_order_acquire) == 0 &&
               !atomic_load_explicit(&job.done, memory_order_acquire))
            nanosleep(&tick, NULL);
        fprintf(stderr, "playing %.2f s of audio (%d codebooks, %d frames)\n", (double) n / sr, n_q, n_frames);
        fprintf(stderr, "load %.1f ms, first audio after %.1f ms\n", t1 - t0, now_ms() - t1);
        ret = play_audio_stream(audio, &job.ready, &job.done, sr);
        pthread_join(thread, NULL);
        if (job.err) {
            fprintf(stderr, "error: decoding failed (n_q = %d, max %d)\n", n_q, zencodec_n_codebooks(model));
            ret = 1;
        } else {
            fprintf(stderr, "decode %.1f ms\n", job.t_done - t1);
        }
    } else {
        if (zencodec_decode(model, codes, n_q, n_frames, audio, n_threads) != 0) {
            fprintf(stderr, "error: decoding failed (n_q = %d, max %d)\n", n_q, zencodec_n_codebooks(model));
            return 1;
        }
        const double t2 = now_ms();
        fprintf(stderr, "decoded %.2f s of audio (%d codebooks, %d frames)\n", (double) n / sr, n_q, n_frames);
        fprintf(stderr, "load %.1f ms, decode %.1f ms (%s backend)\n", t1 - t0, t2 - t1, zencodec_backend());
        if (!write_wav(pos[3], audio, n, sr)) {
            fprintf(stderr, "error: cannot write '%s'\n", pos[3]);
            ret = 1;
        }
    }

    free(audio);
    free(codes);
    zencodec_free(model);
    return ret;
}
