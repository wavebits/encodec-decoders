// zencodec: minimal, dependency-free EnCodec (24 kHz) decoder in C17.
// Based on encodec.cpp (https://github.com/PABannier/encodec.cpp).
//
// Two decoders turn the same EnCodec codes into audio, chosen by the weights file:
//   encodec  Meta's EnCodec decoder (scripts/convert.py). Causal.
//   vocos    Vocos (https://github.com/gemelo-ai/vocos), scripts/convert_vocos.py. Only
//            compiled in with -DZENCODEC_VOCOS (zig build -Dvocos=true). Not causal: it
//            looks 29 frames ahead.
#ifndef ZENCODEC_H
#define ZENCODEC_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct zencodec_model zencodec_model;

// Load a weights file produced by scripts/convert.py (or a full encodec.cpp ggml-model.bin),
// or by scripts/convert_vocos.py. Returns NULL on failure.
zencodec_model * zencodec_load(const char * path);
void             zencodec_free(zencodec_model * model);

int zencodec_sample_rate(const zencodec_model * model); // 24000
int zencodec_hop_length (const zencodec_model * model); // 320 samples per frame
int zencodec_n_codebooks(const zencodec_model * model); // max n_q: 32 (encodec), 16 (vocos)

// "encodec" or "vocos"
const char * zencodec_decoder(const zencodec_model * model);

// How many frames past a frame the decoder must see before that frame's audio is
// final: 0 for encodec, 29 for vocos. A stream's output trails its input by about this
// much (see zencodec_stream_push).
int zencodec_lookahead(const zencodec_model * model);

// Name of the compute backend compiled in: "ref", "neon" or "accelerate".
const char * zencodec_backend(void);

// Decode `codes` laid out as [n_q][n_frames] (codebook-major) into
// n_frames * hop_length mono float samples written to `audio`.
// n_q may be any value in [1, n_codebooks]: 2 = 1.5 kbps, 4 = 3 kbps, 8 = 6 kbps,
// 16 = 12 kbps, 32 = 24 kbps. Vocos takes 2, 4, 8 or 16 only. Returns 0 on success.
// Decodes in chunks of ZENCODEC_CHUNK_FRAMES with a zencodec_stream, so memory use
// does not grow with the input length.
int zencodec_decode(const zencodec_model * model, const int32_t * codes, int n_q, int n_frames,
                    float * audio, int n_threads);

// Streaming decoder: feed the codes chunk by chunk and get the audio as soon as it is
// final. The concatenated output is the same as zencodec_decode on the concatenated
// codes (for encodec, provided the first chunk has at least 7 frames, or is the whole
// input).
typedef struct zencodec_stream zencodec_stream;

#define ZENCODEC_CHUNK_FRAMES 75 // 1 s; a good chunk size for zencodec_stream_decode

// A stream with n_threads > 1 starts n_threads - 1 worker threads, which it keeps until
// zencodec_stream_free. Between calls they spin for about 100 us, then sleep. Use a
// stream from one thread at a time.
zencodec_stream * zencodec_stream_new (const zencodec_model * model, int n_q, int n_threads); // NULL on failure
void              zencodec_stream_free(zencodec_stream * stream);

// `codes` is this chunk's [n_q][n_frames]. Writes the audio that is now final to
// `audio`, at most n_frames * hop_length samples, and returns how many samples it wrote,
// or a negative value on error. With encodec that is always n_frames * hop_length;
// with vocos, audio comes out zencodec_lookahead frames later, so the first calls
// write less (nothing for the first 28 frames).
int zencodec_stream_push(zencodec_stream * stream, const int32_t * codes, int n_frames, float * audio);

// Ends the stream: writes the audio still held back, at most
// zencodec_lookahead * hop_length samples, and returns how many samples it wrote (0
// with encodec), or a negative value on error. The stream takes no codes after this.
int zencodec_stream_flush(zencodec_stream * stream, float * audio);

// For encodec only (returns 4 with vocos): zencodec_stream_push, but returns 0 on
// success, having written n_frames * hop_length samples.
int zencodec_stream_decode(zencodec_stream * stream, const int32_t * codes, int n_frames, float * audio);

#ifdef __cplusplus
}
#endif

#endif // ZENCODEC_H
