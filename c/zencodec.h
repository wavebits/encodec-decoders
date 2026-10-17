// zencodec: minimal, dependency-free EnCodec (24 kHz) decoder in C17.
// Based on encodec.cpp (https://github.com/PABannier/encodec.cpp).
#ifndef ZENCODEC_H
#define ZENCODEC_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct zencodec_model zencodec_model;

// Load a weights file produced by scripts/convert.py (or a full encodec.cpp ggml-model.bin).
// Returns NULL on failure.
zencodec_model * zencodec_load(const char * path);
void             zencodec_free(zencodec_model * model);

int zencodec_sample_rate(const zencodec_model * model); // 24000
int zencodec_hop_length (const zencodec_model * model); // 320 samples per frame
int zencodec_n_codebooks(const zencodec_model * model); // 32 (max)

// Decode `codes` laid out as [n_q][n_frames] (codebook-major) into
// n_frames * hop_length mono float samples written to `audio`.
// n_q may be any value in [1, n_codebooks]: 2 = 1.5 kbps, 4 = 3 kbps, 8 = 6 kbps,
// 16 = 12 kbps, 32 = 24 kbps. Returns 0 on success.
// Decodes in chunks of ZENCODEC_CHUNK_FRAMES with a zencodec_stream, so memory use
// does not grow with the input length.
int zencodec_decode(const zencodec_model * model, const int32_t * codes, int n_q, int n_frames,
                    float * audio, int n_threads);

// Streaming decoder: feed the codes chunk by chunk and get each chunk's audio right
// away. The concatenated output is the same as zencodec_decode on the concatenated
// codes, provided the first chunk has at least 7 frames (or is the whole input).
typedef struct zencodec_stream zencodec_stream;

#define ZENCODEC_CHUNK_FRAMES 75 // 1 s; a good chunk size for zencodec_stream_decode

zencodec_stream * zencodec_stream_new (const zencodec_model * model, int n_q, int n_threads); // NULL on failure
void              zencodec_stream_free(zencodec_stream * stream);

// `codes` is this chunk's [n_q][n_frames]; writes n_frames * hop_length samples to
// `audio`. Returns 0 on success.
int zencodec_stream_decode(zencodec_stream * stream, const int32_t * codes, int n_frames, float * audio);

#ifdef __cplusplus
}
#endif

#endif // ZENCODEC_H
