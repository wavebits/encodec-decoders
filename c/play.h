// Blocking mono float playback.
// macOS: AudioQueue (AudioToolbox). Linux: pipes raw PCM into `aplay`.
#ifndef ZENCODEC_PLAY_H
#define ZENCODEC_PLAY_H

#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>

// Plays mono samples in [-1, 1] at `sample_rate` while another thread is still writing
// them: only samples[0, *ready) are available so far, and the writer raises `ready` as
// it goes (release order) and sets `done` after its last sample. Playback starts right
// away, plays silence if it catches up with the writer, and returns once `done` is set
// and every ready sample has been played. 0 on success.
int play_audio_stream(const float * samples, const atomic_size_t * ready, const atomic_bool * done, int sample_rate);

#endif // ZENCODEC_PLAY_H
