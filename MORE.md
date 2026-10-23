# zencodec — details

The [README](README.md) gives the overview and the main numbers. This file has the rest:
build options, backends, embedding, the Vocos decoder, the API, accuracy, every
benchmark, file formats and the roadmap.

## Status

What works:

- The **EnCodec 24 kHz** model at all bandwidths: 1.5, 3, 6, 12 and 24 kbps.
- **Decoding only** (codes → waveform). Encoding still needs the Python `encodec` package.
- Batch and **streaming** decoding: audio comes out in 1 s chunks, and playback starts
  after about 0.1 s.
- Optionally, the **[Vocos](#vocos-decoder-optional)** decoder for the same codes, at 1.5
  to 12 kbps.
- CPU only, with four backends: portable C, hand-written NEON kernels, Apple's
  Accelerate, or any CBLAS such as OpenBLAS.
- Playback in the CLI on macOS (AudioToolbox) and Linux (`aplay`). The library itself
  does no audio I/O.

What is not done yet: testing on phones. See the [roadmap](#roadmap).

## Quick start

```bash
zig build -Dstrip=true                       # builds zig-out/bin/wb_decode
pip install torch numpy encodec torchaudio   # for the weights and for encoding
python scripts/convert.py --use-f16          # downloads Meta's checkpoint → weights/encodec_24khz_decoder_f16.bin
encodec -b 6 input.wav input.ecdc            # compress a WAV with Meta's reference CLI
zig-out/bin/wb_decode play   weights/encodec_24khz_decoder_f16.bin input.ecdc
zig-out/bin/wb_decode decode weights/encodec_24khz_decoder_f16.bin input.ecdc out.wav -t 4

zig build -Dvocos=true -Dstrip=true         # optional: with the Vocos decoder
python scripts/convert_vocos.py --use-f16    # → weights/vocos_encodec_24khz_f16.bin
zig-out/bin/wb_decode decode weights/vocos_encodec_24khz_f16.bin input.ecdc out.wav -t 4
```

## Build

With [Zig](https://ziglang.org) 0.17 as the build tool and C compiler (the default
optimize mode is `fast`), or with any C17 compiler (see below):

```bash
zig build                                    # native, best backend for the target
zig build -Dstrip=true                       # drop debug info (smallest binaries)
zig build -Dbackend=neon                     # pick the compute backend (see below)
zig build -Dtarget=aarch64-linux-musl        # cross-compile, e.g. for an ARM64 Linux board
zig build -Dprofile=true                     # print time per decoder stage when done
zig build -Df16=true                         # keep weights in f16 in memory: ~23 MB less (see below)
zig build -Dvocos=true                       # include the Vocos decoder (see below)
```

### Compute backends

All the heavy layers are matrix products, which go through one small interface
(`c/backend.h`). One backend is compiled in:

| Backend | Platforms | What it uses | Extra dependency |
| --- | --- | --- | --- |
| `ref` | everywhere | plain C loops the compiler vectorizes | none |
| `neon` | any AArch64: Apple silicon, Android phones, Raspberry Pi | hand-written NEON kernels (8×12 register blocking) | none |
| `accelerate` | macOS, iOS | Apple's BLAS, which uses the CPU's matrix units | Accelerate (part of the OS) |
| `blas` | Linux servers, anywhere with a CBLAS | OpenBLAS (or BLIS, MKL…), one single-threaded call per thread of the pool | `-lopenblas` |

`-Dbackend=auto`, the default, picks `accelerate` on Apple platforms, `neon` on other
AArch64 targets and `ref` elsewhere. `blas` is never picked automatically by
`zig build`; use `-Dbackend=blas` (and `-Dblas-lib=` for a library other than OpenBLAS).
OpenBLAS has tuned AVX2/AVX-512 kernels, while `ref` relies on the compiler, so on x86-64
Linux `blas` is the backend to use. On an M1 (Vocos, 30 s, 1 thread) it took 0.61 s,
against 0.67 s for `neon` and 1.83 s for `ref`, with the same output as `ref`. The
decoder keeps control of the thread count: it sets OpenBLAS, and OpenMP builds of it, to
1 thread on each thread that calls it, and splits the work itself. On Apple, Homebrew's OpenBLAS uses OpenMP, so build with
`-DZENCODEC_BLAS_OMP -Wl,-U,_omp_set_num_threads` or set `OMP_NUM_THREADS=1`. The CLI
prints the backend it was built with.

### f16 weights in memory

By default the weights are converted to f32 when they are loaded, so 23 MB of f16 weights
take 46 MB of memory. `-Df16=true` (or `-DZENCODEC_F16_WEIGHTS`) keeps them in f16
instead. The kernels widen them to f32 as they read them, and all arithmetic stays in
f32. Biases stay in f32, because they are tiny.

| 30 s clip, Apple M1 | Peak memory, 1 thread | Peak memory, 4 threads | Output |
| --- | --- | --- | --- |
| neon | 81 MB | 86 MB | |
| neon, `-Df16=true` | **56 MB** | **62 MB** | bit-identical to neon |
| ref | 80 MB | 81 MB | |
| ref, `-Df16=true` | **56 MB** | **60 MB** | bit-identical to ref |

- **Speed is about the same.** Measured on single-threaded kernels:
  - **neon:** the matrix products are 1–3% slower, because each 8-row block of weights
    is widened once into a stack buffer. The LSTM's bandwidth-bound matrix-vector
    product is 20% faster, because it reads half the bytes.
  - **ref:** matrix products are 5–20% slower; its matrix-vector product is the same.
  - Whole decodes were within the run-to-run noise
    ([bench/results/2026-10-17T1851_5c5dec9-dirty.md](bench/results/2026-10-17T1851_5c5dec9-dirty.md),
    measured on a busy machine).
- **Output is unchanged** with the f16 weights file, since those are the same values.
  With the f32 weights file, the weights are rounded to f16 on load, which gives the
  same result as the f16 file.
- **Backends:** `ref` and `neon` only. BLAS libraries take f32 only, so with
  `-Df16=true` the `auto` backend is `neon` on Apple silicon, and
  `-Dbackend=accelerate -Df16=true` is an error.
- **Half-float conversion:** it uses the CPU's instruction where there is one: any
  AArch64, or x86 built with F16C (`-mf16c`, `-march=native` or
  `-Dcpu=x86_64_v3`). Elsewhere a portable bit-manipulation fallback is used, which is
  slower.

Without Zig, any C17 compiler works. Add the backend's define:

```bash
cc -std=c17 -O3 c/*.c -lm -lpthread -framework AudioToolbox -o wb_decode                      # ref, macOS
cc -std=c17 -O3 -DZENCODEC_BACKEND_NEON c/*.c -lm -lpthread -o wb_decode                       # neon, Linux
cc -std=c17 -O3 -DZENCODEC_BACKEND_ACCELERATE -DACCELERATE_NEW_LAPACK c/*.c -lm -lpthread \
   -framework AudioToolbox -framework Accelerate -o wb_decode                                   # accelerate, macOS
cc -std=c17 -O3 -DZENCODEC_BACKEND_BLAS c/*.c -lopenblas -lm -lpthread -o wb_decode             # blas, Linux
cc -std=c17 -O3 -DZENCODEC_BACKEND_NEON -DZENCODEC_F16_WEIGHTS c/*.c -lm -lpthread -o wb_decode    # neon + f16, Linux
```

Stripped sizes:

| Target | wb_decode |
| --- | --- |
| aarch64-macos (accelerate) | 72 KB |
| aarch64-linux-musl (neon, static, libc included) | 79 KB |
| x86_64-linux-gnu (ref, dynamic) | 35 KB |

These are without Vocos, which adds 17–27 KB.

### Embedding on phones and devices

To embed the decoder, add `c/zencodec.{h,c}`, `c/internal.h`, `c/pool.{h,c}`, `c/backend.h`,
`c/backend_ref.c` (plus `c/vocos.c` and `-DZENCODEC_VOCOS` for Vocos) and the backend
file you want (`backend_neon.c`, `backend_accelerate.c` or `backend_blas.c`) to your
project, and define its `ZENCODEC_BACKEND_*` macro. They need only libc, libm and
pthreads (plus Accelerate or a CBLAS for those backends). Leave out `c/play.c` and
`c/main.c`, which are the CLI.

- **iOS / macOS:** add the files to an Xcode target. They can be called from Swift through
  a bridging header.
- **Android:** build with the NDK's clang (CMake or `ndk-build`) and call through JNI.
  `zig build` cannot target Android, because Zig does not ship Bionic libc.
- **Linux boards:** `zig build -Dtarget=aarch64-linux-musl -Dstrip=true` gives static
  binaries.

These builds compile, but nothing has been run on a phone yet.

For budgeting:

- **Memory:** memory use is set by the weights. About 80 MB at peak, or about 60 MB
  with [f16 weights in memory](#f16-weights-in-memory) (`-Df16=true`), whatever the
  length of the audio.
- **Weights:** 23 MB in f16. Generate them with `scripts/convert.py`, bundle them as an app
  asset, and load them with `zencodec_load`. Check that the license terms below fit your app.
- **CPU:** the number of threads is a parameter; use the number of performance cores
  (2–4 on most phones). A stream keeps its worker threads for its lifetime. Between
  calls they spin for 100 µs, then sleep (tune with `-DZENCODEC_SPIN_US`). On Apple
  platforms, worker threads run at the creating thread's QoS class, so create the
  stream from a thread with `QOS_CLASS_USER_INITIATED` or higher to stay on the
  performance cores. See [docs/threading.md](docs/threading.md).

## Vocos decoder (optional)

[Vocos](https://github.com/gemelo-ai/vocos) is a second decoder for the same EnCodec
codes, trained by its authors to replace EnCodec's own decoder
(`charactr/vocos-encodec-24khz`). It predicts a spectrum per frame and synthesizes the
waveform with an inverse FFT, instead of upsampling step by step. That needs fewer
operations and less memory. The Vocos paper also reports better perceived quality than
EnCodec's decoder at the same bitrates. [docs/vocos.md](docs/vocos.md) describes the model
and the implementation.

```bash
zig build -Dvocos=true                       # or compile c/vocos.c with -DZENCODEC_VOCOS
python scripts/convert_vocos.py --use-f16    # → weights/vocos_encodec_24khz_f16.bin (20 MB)
zig-out/bin/wb_decode decode weights/vocos_encodec_24khz_f16.bin in.ecdc out.wav
```

The weights file selects the decoder, so the same binary and API decode with either one.

- **Bitrates:** 1.5, 3, 6 and 12 kbps (2, 4, 8 or 16 codebooks). 24 kbps files are rejected,
  because the model has no setting for 32 codebooks.
- **Streaming:** Vocos looks 29 frames (0.39 s) ahead, so a stream's output trails its input
  by that much (see [Library API](#library-api)). The output still equals a whole-file
  decode for any chunk sizes.
- **Accuracy:** against PyTorch Vocos (`tests/compare.py --vocos`), 83–94 dB SNR with f32
  weights, which is the limit of 16-bit output, and 42–55 dB with f16 weights.
- **Cost:** 17–27 KB more binary, depending on the target. Works with every backend and
  with `-Df16=true`.

30 s of speech cut from the 10-minute file below (6 kbps), Apple M1, median of 5
interleaved runs, load average 6.6–8 (times are about 1.4× those of a quiet machine). The
SNR of each decoder is against its own PyTorch reference: the two decoders produce
different audio, so they can't be scored against each other. Full table, with music:
[bench/results/2026-10-22T2321_3e283ee-dirty.md](bench/results/2026-10-22T2321_3e283ee-dirty.md).

| Decoder | Wall, 1 thread | Decode, 1 thread | Decode, 4 threads | Peak memory | Size | SNR (worst 1 s) |
| --- | --- | --- | --- | --- | --- | --- |
| wb_decode EnCodec (accelerate) | 0.97 s | 0.91 s | 0.89 s | 77 MB | 73 KB | 60.6 dB (49.3) |
| wb_decode **Vocos** (accelerate) | **0.30 s** | **0.26 s** | 0.34 s | **52 MB** | 90 KB | 56.8 dB (50.6) |
| wb_decode EnCodec (neon) | 3.01 s | 2.93 s | 1.89 s | 82 MB | 89 KB | 60.6 dB (49.2) |
| wb_decode **Vocos** (neon) | 1.00 s | 0.96 s | 0.61 s | 52 MB | 106 KB | 56.8 dB (50.7) |
| wb_decode EnCodec (ref) | 7.58 s | 7.50 s | 4.97 s | 78 MB | 72 KB | 60.6 dB (49.2) |
| wb_decode **Vocos** (ref) | 3.05 s | 3.01 s | 2.85 s | 52 MB | 90 KB | 56.8 dB (50.6) |
| encodec.cpp | 9.35 s | 9.29 s | 9.02 s | 1.09 GB | 1.0 MB | 53.7 dB (51.8) |
| PyTorch `encodec` | 5.00 s | 3.07 s | 2.41 s | 2.17 GB | 518 MB + Python | reference |
| PyTorch `vocos` | 2.29 s | 0.19 s | 0.20 s | 501 MB | 522 MB + Python | reference |

- **Against C EnCodec:** Vocos decodes 2.5–3.5× faster on one thread, with every backend,
  and needs 25 MB less memory. Its weights file is 20 MB instead of 23 MB.
- **Against PyTorch Vocos:** the whole run is 7.6× faster (0.30 s against 2.29 s) and
  uses 10× less memory. Decoding alone is a little slower (0.26 s against 0.19 s): both
  run the matrix products in Apple's BLAS, and PyTorch does the whole clip in one pass.
  PyTorch's time is mostly startup, and its memory grows with the length of the audio.
- **Threads:** on accelerate the matrix products run on the M1's shared matrix unit, so
  more threads don't help. neon gains 1.6× on 4 threads.
- **encodec.cpp** was the slowest decoder in this session. In an earlier, quieter session
  it took 2.97 s on 4 threads (see below); its threads synchronize once per graph node,
  which suffers most from other load.
- **Music:** on the 30 s music clip, Vocos gives 52.5 dB (worst second 42.6 dB), where the
  EnCodec decoder's f16 weights drop one second to 14.8 dB. Times were the same as for
  speech.

The whole 10-minute speech file (`speech-us-gov-0013.ecdc`, 6 kbps): median of 3
interleaved runs, load average 5.8–8.6. Full table:
[bench/results/2026-10-22T2332_3e283ee-dirty.md](bench/results/2026-10-22T2332_3e283ee-dirty.md).

| Decoder | Wall, 1 thread | Wall, 4 threads | Peak memory |
| --- | --- | --- | --- |
| wb_decode EnCodec (accelerate) | 16.7 s (13.6–38.3) | 29.9 s (21.1–33.3) | 137 MB |
| wb_decode **Vocos** (accelerate) | **6.6 s** (5.5–15.0) | 7.5 s (6.0–10.7) | **109 MB** |
| wb_decode EnCodec (neon) | 54.0 s (44.9–106.3) | 44.6 s (33.4–53.7) | 138 MB |
| wb_decode **Vocos** (neon) | 29.4 s (19.9–35.0) | 20.6 s (12.7–20.7) | 109 MB |
| PyTorch `vocos` | | 12.3–15.9 s (decode 6.9–9.0 s) | 2.25–2.53 GB |

- **Speed:** Vocos is 2.5× faster than EnCodec on accelerate (92× real time), and 1.8–2.2×
  on neon.
- **Memory:** the CLI keeps the whole output in memory to write the WAV: 58 MB for
  10 minutes. Without it, the decoders use the same 80 MB (EnCodec) and 52 MB (Vocos) as
  for 30 s. A program that streams the output (`zencodec_stream_push`) doesn't need that
  buffer. PyTorch Vocos needs 2.3–2.5 GB. PyTorch EnCodec and encodec.cpp keep the whole
  file in memory (2.17 GB and 1.09 GB for 30 s), so they were not run on it.
- **Threads under load:** EnCodec on accelerate was slower on 4 threads than on 1. Its
  LSTM synchronizes its threads twice per frame, so when other programs take cores, a
  descheduled thread stalls every step ([docs/threading.md](docs/threading.md)). On a busy
  machine or a container with a CPU quota, use as many threads as there are free cores.
  Vocos has no recurrence and is barely affected.

## Usage


```bash
wb_decode decode <weights.bin> <codes.bin|in.ecdc> <out.wav> [-t n_threads]
wb_decode play   <weights.bin> <codes.bin|in.ecdc>           [-t n_threads]
```

The input can be:

- An `.ecdc` file from the reference `encodec` CLI, made with the 24 kHz model and
  without `--lm`.
- A raw codes file from `python scripts/encode.py input.wav codes.bin --bandwidth 6`.

`play` decodes in 1 s chunks on a background thread and starts playing after the first
chunk, so playback does not wait for the whole file.

## Library API

```c
zencodec_model * m = zencodec_load("weights/encodec_24khz_decoder_f16.bin");
// codes: int32 [n_q][n_frames], n_q = 2/4/8/16/32 for 1.5/3/6/12/24 kbps
float * audio = malloc(sizeof(float) * n_frames * zencodec_hop_length(m)); // 320 samples/frame
zencodec_decode(m, codes, n_q, n_frames, audio, /*n_threads=*/4);

// or chunk by chunk, e.g. as codes arrive; same output as long as the first chunk has >= 7 frames
zencodec_stream * s = zencodec_stream_new(m, n_q, /*n_threads=*/4);
int n = zencodec_stream_push(s, chunk_codes /* [n_q][chunk_frames] */, chunk_frames, out); // samples ready
// ... more chunks ...
n = zencodec_stream_flush(s, out); // at the end: samples held back (Vocos only)
zencodec_stream_free(s);
zencodec_free(m);
```

`zencodec_stream_push` returns how many samples it wrote. With EnCodec that is always
`chunk_frames × 320`. Vocos looks 29 frames ahead (`zencodec_lookahead`), so its first
pushes write less, and `zencodec_stream_flush` writes the rest. `zencodec_stream_decode`
(returns 0 and always writes a full chunk) is still there for EnCodec.

The EnCodec decoder is causal. Between chunks, the stream carries each layer's state: the
LSTM state, the last `K - 1` inputs of each conv, and the overlap of each transposed
conv. So nothing is recomputed. `decode` uses a stream internally, so memory does not
depend on the length of the input.

## How it works

```
codes [n_q][T]
  → RVQ decode            sum of n_q codebook embeddings          → [128][T]
  → conv k7               128 → 512
  → 2-layer LSTM + skip   512
  → 4× { ELU, transposed conv (stride 8, 5, 4, 2), residual block }
  → ELU, conv k7          32 → 1                                   → [T × 320] samples at 24 kHz
```

There is no graph and no tensor library. Each layer is written as matrix products over
`[C][T]` buffers, which the compiled-in backend runs (see
[compute backends](#compute-backends)). Weights are stored in f16 and converted to
f32 on load, or kept in f16 with `-Df16=true`.

All layers run on one persistent thread pool per stream, whose fork-join costs about a
microsecond. The LSTM recurrence is the only part that must run one frame at a time, so
each frame's matrix-vector product is split across the threads by hidden unit. That
makes it 3.5× faster on 4 threads. [docs/threading.md](docs/threading.md) explains the
pool and the recurrence.
See [docs/architecture.md](docs/architecture.md) for each layer, the streaming state, and
which function implements what.

## Accuracy

```bash
python tests/compare.py zig-out/bin/wb_decode weights/encodec_24khz_decoder_f16.bin
```

Against the PyTorch decoder, it gives an SNR of 55–63 dB with the f16 weights
and about 80–100 dB with the f32 weights.

f16 rounding is not spread evenly. In one 30 s music clip, one second drops to about
14 dB while the rest stay near 60 dB. encodec.cpp dips in the same place, to 35 dB. With
the f32 weights, every second stays at about 80 dB.

Vocos, against PyTorch Vocos:

```bash
python tests/compare.py zig-out/bin/wb_decode weights/vocos_encodec_24khz_f16.bin --vocos   # needs -Dvocos=true
```

- Test cases (0.01 to 5 s, 1.5 to 12 kbps): 83–94 dB with f32 weights, 42–55 dB with f16,
  on every backend.
- The whole 10-minute speech file: 80.3 dB (worst second 66.2 dB) with f32 weights,
  50.2 dB (worst second 30.7 dB) with f16.
- Vocos output can go beyond full scale: 44 samples of that file do. 16-bit output clips
  them, so `bench/bench.py` clips the float reference the same way before comparing.

## Benchmarks

To reproduce:

```bash
python bench/bench.py in.ecdc [more.ecdc ...] -t 1,4 -r 3 [--baseline <git rev>]
```

`bench/bench.py`:

1. Builds `wb_decode` once per backend, including the f16-weights builds (`neon+f16`,
   `ref+f16`). With `--baseline`, it also builds `wb_decode` from an earlier commit, once per backend, for before/after
   comparisons.
2. Clones encodec.cpp at a pinned commit into `bench/work/`, applies
   `bench/encodec_cpp.patch` (see below) and builds it.
3. Uses the PyTorch `encodec` package through `bench/python_decode.py`, if `encodec` and
   `torch` are installed.
4. Runs every decoder on every input and thread count, interleaved, and records:
   - wall time, and the decode time each decoder reports itself
   - CPU time (user and sys)
   - peak memory
   - binary or install size
   - SNR against the PyTorch decoder, overall and for the worst second
5. Prints a Markdown table, and writes it with a description of the machine to
   `bench/results/<date>_<commit>.md`. It also appends one row per measurement to
   `bench/results/history.csv`, so results can be tracked across commits and machines.

Setup for all results below (2026-10-17, Apple M1, 16 GB):

- encodec.cpp was at commit `83deb38` (ggml `c18f9bae`), built in Release mode with the
  CPU backend.
- PyTorch was version 2.13.0 with `encodec` 0.1.1, built with Apple's Accelerate library
  (`BLAS_INFO=accelerate`).
- The input was 6 kbps `.ecdc` clips cut from a music file.
- Times are for the whole process: loading the model, decoding, and writing the WAV.
- The machine had other work running, so times vary between sessions. Compare numbers
  within one table only. Memory and sizes are exact.

### All decoders and backends

Working tree on top of commit `5c5dec9` (the thread pool and parallel LSTM recurrence),
against `5c5dec9` itself. 5 interleaved runs, quiet machine (load average 2.4–3.5). Full
table with 10 s inputs:
[bench/results/2026-10-17T1817_5c5dec9-dirty.md](bench/results/2026-10-17T1817_5c5dec9-dirty.md).

30 s of audio, median of 5 runs. "Decode" is the time each decoder reports for decoding
alone, without process startup or model loading.

| Decoder | Threads | Wall | Decode | × real time | CPU (user) | Peak memory | SNR | Worst 1 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wb_decode (accelerate) | 1 | 0.58 s | 0.54 s | 55.5× | 0.57 s | 77 MB | 27.3 dB | 13.7 dB |
| wb_decode (accelerate) | 4 | **0.45 s** | **0.41 s** | **73.9×** | 0.96 s | 77 MB | 27.3 dB | 13.7 dB |
| wb_decode (accelerate) before the pool | 4 | 0.52 s | 0.48 s | 63.0× | 0.60 s | 77 MB | 27.3 dB | 13.7 dB |
| wb_decode (neon) | 1 | 1.72 s | 1.67 s | 17.9× | 1.70 s | 79 MB | 27.3 dB | 13.7 dB |
| wb_decode (neon) | 4 | 0.68 s | 0.62 s | 48.0× | 2.39 s | 81 MB | 27.3 dB | 13.7 dB |
| wb_decode (neon) before the pool | 4 | 0.87 s | 0.82 s | 36.6× | 1.96 s | 78 MB | 27.3 dB | 13.7 dB |
| wb_decode (ref) | 1 | 4.40 s | 4.35 s | 6.9× | 4.37 s | 77 MB | 27.3 dB | 13.7 dB |
| wb_decode (ref) | 4 | 1.86 s | 1.81 s | 16.6× | 7.12 s | 80 MB | 27.3 dB | 13.7 dB |
| wb_decode (ref) before the pool | 4 | 2.21 s | 2.17 s | 13.8× | 6.43 s | 77 MB | 27.3 dB | 13.7 dB |
| encodec.cpp | 4 | 2.97 s | 2.94 s | 10.2× | 11.51 s | 1.09 GB | 46.2 dB | 34.7 dB |
| PyTorch `encodec` | 1 | 2.36 s | 1.43 s | 21.0× | 2.00 s | 2.17 GB | reference | reference |
| PyTorch `encodec` | 4 | 1.79 s | 0.85 s | 35.2× | 2.21 s | 2.17 GB | reference | reference |

- **accelerate** is the fastest decoder measured. With the thread pool it is now fastest
  on 4 threads. The LSTM recurrence runs on the cores' NEON units, and the matrix unit
  runs the convolutions.
- **The thread pool** made every backend faster on 4 threads: −15% (accelerate), −24%
  (neon) and −17% (ref) wall time. 1-thread times are unchanged. CPU time goes up
  because idle workers spin between jobs. See [docs/threading.md](docs/threading.md).
- **neon** needs no libraries and is about 2.7× faster than `ref`. It is also faster than
  PyTorch's decoding. It is the backend for Android phones and other ARM boards.
- **SNR:** all C backends give the same SNR. ref and neon are bit-identical for any
  thread count; accelerate on several threads differs by a few LSB. The low overall SNR
  on this clip comes from one second that is very sensitive to f16 weight rounding (see
  [Accuracy](#accuracy)); with the f32 weights every backend reaches 81.6 dB. encodec.cpp
  keeps more of its weights in f32, so it scores higher on that second.
- **PyTorch:**
  - Its wall time includes about 1 s of Python and PyTorch startup.
  - Its install is 518 MB of PyTorch, plus a Python runtime.
  - Its memory grows with the length of the audio: 953 MB for 10 s and 2.17 GB for 30 s.

The earlier results, at commit `617db75` under heavier load, are in
[bench/results/2026-10-17T1438_617db75.md](bench/results/2026-10-17T1438_617db75.md).

### Thread scaling and memory growth, against encodec.cpp

Earlier session, at commit `d8a90e1` (before the backends), under heavier load (load
average 4–16).

**Wall time** (lower is better):

| Case | wb_decode | encodec.cpp |
| --- | --- | --- |
| 10 s audio, 1 thread\* | 1.67–2.04 s | 1.94–2.20 s |
| 10 s audio, 4 threads | 2.3–2.9 s | 2.6–3.3 s |
| 30 s audio, 4 threads | 8.1–9.3 s | 13.6–15.4 s |
| 30 s audio, 8 threads | 6.5–6.6 s | 37–175 s |

\* Measured in a separate session at lower load (load average about 2.6), so compare
this row only within itself.

- **1 thread:** wb_decode is about 10–15% faster than encodec.cpp.
- **4 threads:** encodec.cpp uses about twice the CPU time.
- **8 threads:** encodec.cpp slows down sharply. Its graph has one node per LSTM step and
  frame, and its threads synchronize at every node.
- **Metal:** encodec.cpp's Metal backend (`-g 99`) was slower than its CPU backend: 9.5 s
  for 10 s of audio.

**Peak memory** (`/usr/bin/time -l`, peak footprint):

| Audio length | wb_decode | encodec.cpp | PyTorch `encodec` |
| --- | --- | --- | --- |
| 10 s | 71 MB | 377 MB | 953 MB |
| 30 s | 72 MB | 1.09 GB | 2.17 GB |

zencodec decodes in fixed-size chunks, so its memory does not change with the length of
the audio. encodec.cpp and PyTorch process the whole file at once, so their memory grows
with the length of the audio. The PyTorch figures are from the session above.

**Size** (macOS arm64):

| | Binary (stripped) | Weights file |
| --- | --- | --- |
| wb_decode | 70 KB | 23 MB (decoder only, f16) |
| encodec.cpp `decompress`, CPU only | 905 KB | 46.5 MB (encoder + decoder) |
| encodec.cpp `decompress`, default build with Metal | 1.32 MB | same file |
| PyTorch `encodec` | 529 MB `torch` package + 332 KB `encodec` + Python | 93 MB checkpoint |

Upstream encodec.cpp needed four changes to decode these files correctly
(`bench/encodec_cpp.patch`):

- **`convert.py`:** it writes the 1-element bias of `decoder.model.15` as a 0-dim tensor,
  which the loader rejects. Wrapping it in `np.atleast_1d` fixes this.
- **Bandwidth:** `examples/decompress` hardcodes 12 kbps, so it misreads files encoded at
  any other bandwidth. The patch sets the bandwidth from the file's codebook count
  (`n_codebooks * 3 / 4`).
- **Code order:** `examples/decompress` reads `.ecdc` codes in time-major order but
  treats them as codebook-major. As a result, `.ecdc` files from the Python `encodec` CLI
  decode to noise. The patch transposes them.
- **Graph size:** `ENCODEC_MAX_NODES` (80000) runs out after about 15 s of 6 kbps audio.
  The patch makes the limit overridable, and the script sets it to 400000.

## Roadmap

- **Speed:**
  - Done: GEMM-based layers with `neon` and `accelerate` backends.
  - Done: a persistent thread pool, and the LSTM recurrence split across threads
    ([docs/threading.md](docs/threading.md)).
  - Done: a `blas` backend (OpenBLAS) for Linux servers.
  - An `avx2` backend for x86 without a BLAS, by porting the NEON kernels.
  - Faster non-matrix layers in Vocos (depthwise conv, layer norm, GELU, ISTFT).
- **Vocos:** done, as an optional decoder (`-Dvocos=true`).
- **Memory:**
  - Done: optional f16 weights in memory (`-Df16=true`), which cuts the peak by about
    23 MB.
  - int8 weights, which would quarter the weights' resident size.
- **Size:** int8 weights, so the shipped asset is about 12 MB.
- **Devices:** test on Android and iOS hardware, and add sample apps.

Unsure future possibilities. These aren't needed for 24 kHz files encoded without `--lm`,
and the decoder rejects such files today:

- **`--hq` (the 48 kHz model):** stereo 48 kHz audio for music. It isn't causal, since
  it decodes 1 s segments with an overlap, so it can't stream with low latency. It also
  costs more compute and needs its own weights.
- **`--lm` (the entropy coder):** a small Transformer language model plus arithmetic
  coding, which makes files about 25–40% smaller. Decoding has to run the Transformer
  every frame, which works against the speed and memory goals.

## Formats

**Codes file** (little-endian): `int32 n_q`, `int32 n_frames`, then `int32 codes[n_q][n_frames]`.

**`.ecdc` file** (from `encodec`):

- The header is `"ECDC"`, a `uint8` version 0, and a big-endian `uint32` JSON length.
- The JSON metadata has `m` (model), `al` (audio length in samples), `nc` (number of
  codebooks) and `lm`.
- The codes follow, bit-packed at 10 bits each, LSB-first, frame by frame.

Files with `"lm": true` are arithmetic-coded with a language model and are rejected. The
output is trimmed to `al` samples.

**Weights file:**

- The layout is the same as encodec.cpp's `ggml-model.bin`: magic, 9 int32 hparams, then
  the tensors.
- It keeps only the `quantizer.*` and `decoder.*` tensors, with weight-norm already folded
  into the conv weights.
- The loaders skip encoder tensors, so a full encodec.cpp f32 or f16 weights file also
  works.

The Vocos weights file (`scripts/convert_vocos.py`) uses the same tensor records, with
the magic `"vocs"` and its own 10 hparams. It holds the Vocos backbone and head, and the
16 EnCodec codebooks it uses.

The files are not included in this repository. To create them:

```bash
python scripts/convert.py --use-f16      # → weights/encodec_24khz_decoder_f16.bin (23 MB)
python scripts/convert.py                # → weights/encodec_24khz_decoder_f32.bin (46 MB)
python scripts/convert_vocos.py --use-f16  # → weights/vocos_encodec_24khz_f16.bin (20 MB)
```

The script downloads Meta's checkpoint (`encodec_24khz-d7cc33bc.th`) and checks its hash.
To use a checkpoint you already have, pass `--checkpoint path.th`; to choose the output
file, pass `--out`. The f32 file matches PyTorch to within 16-bit WAV rounding.

## Project layout

| Path | Purpose |
| --- | --- |
| `README.md`, `MORE.md` | Overview and main numbers; all the details |
| `c/zencodec.{h,c}` | Decoder library: the API and the EnCodec decoder |
| `c/vocos.c`, `c/internal.h` | Optional Vocos decoder; internals shared by both decoders |
| `c/backend*.{h,c}` | Compute backends: `ref`, `neon`, `accelerate`, `blas` |
| `c/pool.{h,c}` | Persistent fork-join thread pool |
| `c/play.{h,c}`, `c/main.c` | Command-line tool (`wb_decode`) and playback |
| `weights/` | Generated weights files (not in the repository; see `scripts/convert.py`) |
| `scripts/convert.py` | Downloads Meta's checkpoint and builds the weights file |
| `scripts/convert_vocos.py` | Downloads the Vocos checkpoint and builds its weights file |
| `scripts/encode.py` | WAV → raw codes file, using the reference `encodec` package |
| `tests/compare.py` | Checks a CLI's output against the PyTorch decoder (EnCodec, or Vocos with `--vocos`) |
| `bench/` | Benchmark against encodec.cpp and PyTorch, and of Vocos with `--vocos` (scripts and patch) |
| `docs/` | Decoder architecture, Vocos, threading design, and the diagram |
| `build.zig` | Builds `wb_decode`, natively or cross-compiled |
| `.github/workflows/` | CI (every backend, accuracy against PyTorch) and releases |
| `LICENSE` | MIT License for the code |

## Credits & license

The decoder follows the structure of
[encodec.cpp](https://github.com/PABannier/encodec.cpp) by Pierre-Antoine Bannier, and
the model is [EnCodec](https://github.com/facebookresearch/encodec) by Meta AI.

The code is released under the [MIT License](LICENSE).

| Decoder | Weights | License |
| --- | --- | --- |
| EnCodec (default) | Meta's `encodec_24khz` | **CC-BY-NC 4.0**: non-commercial only |
| Vocos (optional) | `charactr/vocos-encodec-24khz` | **MIT**, but it contains EnCodec's CC-BY-NC 4.0 codebooks |

The EnCodec weights are released by Meta under
[CC-BY-NC 4.0](https://github.com/facebookresearch/encodec#license). For that reason this
repository does not include them. `scripts/convert.py` downloads them from Meta. Files
generated from them fall under the same non-commercial terms.

[Vocos](https://github.com/gemelo-ai/vocos) is by Hubert Siuzdak / Charactr (gemelo-ai),
under the MIT License, as is the `charactr/vocos-encodec-24khz` checkpoint. That
checkpoint also contains EnCodec's codebooks, which come from Meta's CC-BY-NC 4.0 weights,
so check both licenses before shipping the Vocos weights file.
