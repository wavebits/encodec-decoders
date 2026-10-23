#if defined(__linux__)
#define _POSIX_C_SOURCE 200809L // popen
#endif

#include "play.h"

#include <stdio.h>
#include <string.h>
#include <time.h>

#if defined(__APPLE__)

#include <AudioToolbox/AudioToolbox.h>

#define N_BUFFERS      3
#define BUFFER_FRAMES  4096
#define SILENCE_FRAMES 480 // 20 ms, played when the decoder has not caught up yet

typedef struct {
    const float         * samples;
    size_t                pos;
    const atomic_size_t * ready;
    const atomic_bool   * done;
    atomic_int            n_idle; // buffers returned with nothing left to play
} player;

static void fill(void * user, AudioQueueRef q, AudioQueueBufferRef buf) {
    player * p = user;
    const bool done = atomic_load_explicit(p->done, memory_order_acquire); // before reading ready
    size_t n = atomic_load_explicit(p->ready, memory_order_acquire) - p->pos;
    if (n == 0 && done) {
        atomic_fetch_add(&p->n_idle, 1);
        return;
    }
    if (n == 0) { // underrun: keep the buffer cycling with a little silence
        memset(buf->mAudioData, 0, SILENCE_FRAMES * sizeof(float));
        buf->mAudioDataByteSize = SILENCE_FRAMES * sizeof(float);
        AudioQueueEnqueueBuffer(q, buf, 0, NULL);
        return;
    }
    if (n > BUFFER_FRAMES) n = BUFFER_FRAMES;
    memcpy(buf->mAudioData, p->samples + p->pos, n * sizeof(float));
    buf->mAudioDataByteSize = (UInt32) (n * sizeof(float));
    p->pos += n;
    AudioQueueEnqueueBuffer(q, buf, 0, NULL);
}

int play_audio_stream(const float * samples, const atomic_size_t * ready, const atomic_bool * done, int sample_rate) {
    AudioStreamBasicDescription fmt = {
        .mSampleRate       = sample_rate,
        .mFormatID         = kAudioFormatLinearPCM,
        .mFormatFlags      = kLinearPCMFormatFlagIsFloat | kLinearPCMFormatFlagIsPacked,
        .mBytesPerPacket   = sizeof(float),
        .mFramesPerPacket  = 1,
        .mBytesPerFrame    = sizeof(float),
        .mChannelsPerFrame = 1,
        .mBitsPerChannel   = 32,
    };
    player p = {samples, 0, ready, done, 0};
    AudioQueueRef q;
    // NULL run loop: callbacks run on the queue's own thread
    if (AudioQueueNewOutput(&fmt, fill, &p, NULL, NULL, 0, &q) != noErr) {
        fprintf(stderr, "play: cannot open audio output\n");
        return 1;
    }
    for (int i = 0; i < N_BUFFERS; i++) {
        AudioQueueBufferRef buf;
        if (AudioQueueAllocateBuffer(q, BUFFER_FRAMES * sizeof(float), &buf) != noErr) break;
        fill(&p, q, buf);
    }
    if (AudioQueueStart(q, NULL) != noErr) {
        AudioQueueDispose(q, true);
        fprintf(stderr, "play: cannot start audio output\n");
        return 1;
    }

    // every buffer has come back empty -> all audio was handed to the device;
    // then stop without discarding and wait for the queue to wind down
    const struct timespec tick = {0, 20 * 1000 * 1000};
    while (atomic_load(&p.n_idle) < N_BUFFERS) nanosleep(&tick, NULL);
    AudioQueueStop(q, false);
    for (;;) {
        UInt32 running = 0, size = sizeof(running);
        AudioQueueGetProperty(q, kAudioQueueProperty_IsRunning, &running, &size);
        if (!running) break;
        nanosleep(&tick, NULL);
    }
    AudioQueueDispose(q, true);
    return 0;
}

#elif defined(__linux__)

int play_audio_stream(const float * samples, const atomic_size_t * ready, const atomic_bool * done, int sample_rate) {
    char cmd[128];
    snprintf(cmd, sizeof(cmd), "aplay -q -t raw -f FLOAT_LE -c 1 -r %d", sample_rate);
    FILE * pipe = popen(cmd, "w");
    if (!pipe) {
        fprintf(stderr, "play: cannot run aplay\n");
        return 1;
    }
    // aplay inserts the silence itself if we fall behind
    const struct timespec tick = {0, 5 * 1000 * 1000};
    size_t written = 0;
    bool ok = true;
    for (;;) {
        const bool   fin   = atomic_load_explicit(done, memory_order_acquire); // before reading ready
        const size_t avail = atomic_load_explicit(ready, memory_order_acquire);
        if (avail == written) {
            if (fin) break;
            nanosleep(&tick, NULL);
            continue;
        }
        const size_t want = avail - written, w = fwrite(samples + written, sizeof(float), want, pipe);
        written += w;
        if (w < want) {
            ok = false;
            break;
        }
    }
    const int status = pclose(pipe);
    return ok && status == 0 ? 0 : 1;
}

#else

int play_audio_stream(const float * samples, const atomic_size_t * ready, const atomic_bool * done, int sample_rate) {
    (void) samples; (void) ready; (void) done; (void) sample_rate;
    fprintf(stderr, "play: not supported on this platform\n");
    return 1;
}

#endif
