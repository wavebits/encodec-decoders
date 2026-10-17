# zencodec

![EnCodec decoder](docs/encodec_decoder.png)

**zencodec is the smallest, most portable, lowest-memory and fastest decoder for Meta's
[EnCodec](https://github.com/facebookresearch/encodec) audio codec, made to be embedded in
phones and other small devices.** EnCodec compresses speech and music to as little as
1.5 kbps, but turning the codes back into sound normally takes PyTorch or
[encodec.cpp](https://github.com/PABannier/encodec.cpp) with ggml. zencodec does the same
job in two plain C files that need nothing beyond libc and threads, in memory that doesn't grow with the
length of the audio. A second implementation in Zig gives the same output, so the two can
be compared and the faster or smaller approach kept. The diagram above shows the decoder;
[docs/architecture.md](docs/architecture.md) explains it layer by layer.

## Benchmark at a glance

zencodec (C) against encodec.cpp, decoding 30 s of 6 kbps audio on an Apple M1 (median
of 3 runs):

| | zencodec | encodec.cpp | |
| --- | --- | --- | --- |
| 📦 Binary size | **70 KB** | 905 KB | **13× smaller** |
| 🧠 Memory | **72 MB** | 1.09 GB | **15× less** |
| ⚡ Time (4 threads) | **8.3 s** | 15.0 s | **1.8× faster** |
| 🔌 Dependencies | **libc** | ggml + C++ runtime | |

zencodec's memory stays the same for any length of audio; encodec.cpp's grows with it.
See [Benchmarks](#benchmarks) for every measurement, the method, and how to reproduce them.

## Status

What works:

- The **EnCodec 24 kHz** model at all bandwidths: 1.5, 3, 6, 12 and 24 kbps.
- **Decoding only** (codes → waveform). Encoding still needs the Python `encodec` package.
- Batch and **streaming** decoding: audio comes out in 1 s chunks, and playback starts
  after about 0.1 s.
- CPU only. The code is portable, with SIMD from the compiler (C) or `@Vector` (Zig).
- Playback in the CLI on macOS (AudioToolbox) and Linux (`aplay`). The library itself
  does no audio I/O.

What is not done yet: the 48 kHz model, the language-model (`--lm`) entropy coder, and
testing on phones. See the [roadmap](#roadmap).

## Quick start

```bash
zig build -Dstrip=true                       # builds zig-out/bin/zencodec-c and zencodec-zig
pip install torch numpy encodec torchaudio   # for the weights and for encoding
python scripts/convert.py --use-f16          # downloads Meta's checkpoint → weights/encodec_24khz_decoder_f16.bin
encodec -b 6 input.wav input.ecdc            # compress a WAV with Meta's reference CLI
zig-out/bin/zencodec-c play   weights/encodec_24khz_decoder_f16.bin input.ecdc
zig-out/bin/zencodec-c decode weights/encodec_24khz_decoder_f16.bin input.ecdc out.wav -t 4
```

## Build

With [Zig](https://ziglang.org) 0.17, which builds both versions (the default optimize
mode is `fast`):

```bash
zig build                                    # native
zig build -Dstrip=true                       # drop debug info (smallest binaries)
zig build -Dtarget=aarch64-linux-musl        # cross-compile, e.g. for an ARM64 Linux board
```

For the C version on its own, any C17 compiler works:

```bash
cc -std=c17 -O3 c/*.c -lm -lpthread -framework AudioToolbox -o zencodec   # macOS
cc -std=c17 -O3 c/*.c -lm -lpthread -o zencodec                           # Linux
```

Stripped sizes:

| Target | zencodec-c | zencodec-zig |
| --- | --- | --- |
| aarch64-macos | 70 KB | 323 KB |
| aarch64-linux-musl (static, libc included) | 72 KB | 330 KB |
| x86_64-linux-gnu (dynamic) | 28 KB | 366 KB |
| aarch64-linux-android | use the NDK's clang | 333 KB |

### Embedding on phones and devices

To embed the decoder, add the two files `c/zencodec.{h,c}` to your project. They need
only libc, libm and pthreads. Leave out `c/play.c` and `c/main.c`, which are the CLI.

- **iOS / macOS:** add the files to an Xcode target. They can be called from Swift through
  a bridging header.
- **Android:** build with the NDK's clang (CMake or `ndk-build`) and call through JNI.
  Zig cannot cross-compile the C version for Android, because it does not ship Bionic
  libc. The Zig version cross-compiles with `zig build -Dtarget=aarch64-linux-android`.
- **Linux boards:** `zig build -Dtarget=aarch64-linux-musl -Dstrip=true` gives static
  binaries.

These builds compile, but nothing has been run on a phone yet.

For budgeting:

- **Memory:** memory use is set by the weights. About 70 MB at peak, whatever the length
  of the audio.
- **Weights:** 23 MB in f16. Generate them with `scripts/convert.py`, bundle them as an app
  asset, and load them with `zencodec_load`. Check that the license terms below fit your app.
- **CPU:** the number of threads is a parameter; 2–4 suits big cores on a phone.

## Usage

Both CLIs take the same commands:

```bash
zencodec-c decode <weights.bin> <codes.bin|in.ecdc> <out.wav> [-t n_threads]
zencodec-c play   <weights.bin> <codes.bin|in.ecdc>           [-t n_threads]
```

The input can be:

- An `.ecdc` file from the reference `encodec` CLI, made with the 24 kHz model and
  without `--lm`.
- A raw codes file from `python scripts/encode.py input.wav codes.bin --bandwidth 6`.

`play` decodes in 1 s chunks on a background thread and starts playing after the first
chunk, so playback does not wait for the whole file.

## Library API

C:

```c
zencodec_model * m = zencodec_load("weights/encodec_24khz_decoder_f16.bin");
// codes: int32 [n_q][n_frames], n_q = 2/4/8/16/32 for 1.5/3/6/12/24 kbps
float * audio = malloc(sizeof(float) * n_frames * zencodec_hop_length(m)); // 320 samples/frame
zencodec_decode(m, codes, n_q, n_frames, audio, /*n_threads=*/4);

// or chunk by chunk, e.g. as codes arrive; same output as long as the first chunk has >= 7 frames
zencodec_stream * s = zencodec_stream_new(m, n_q, /*n_threads=*/4);
zencodec_stream_decode(s, chunk_codes /* [n_q][chunk_frames] */, chunk_frames, chunk_audio);
zencodec_stream_free(s);
zencodec_free(m);
```

Zig:

```zig
var model = try zencodec.Model.load(gpa, io, "weights/encodec_24khz_decoder_f16.bin");
defer model.deinit();
const audio = try gpa.alloc(f32, n_frames * zencodec.hop_length);
try model.decode(codes, n_q, n_frames, audio, n_threads);

var stream = try zencodec.Stream.init(&model, n_q, n_threads); // chunk by chunk
defer stream.deinit();
try stream.decode(chunk_codes, chunk_frames, chunk_audio);
```

Both decoders are causal. Between chunks, the stream carries each layer's state: the
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

There is no graph and no tensor library: each layer is a plain loop over `[C][T]`
buffers. The convolutions are split across threads by output channel. Weights are stored in f16 and converted on load.
See [docs/architecture.md](docs/architecture.md) for each layer, the streaming state, and
which function implements what.

## Accuracy

```bash
python tests/compare.py zig-out/bin/zencodec-c   weights/encodec_24khz_decoder_f16.bin
python tests/compare.py zig-out/bin/zencodec-zig weights/encodec_24khz_decoder_f16.bin
```

Against the PyTorch decoder, both versions give an SNR of 55–63 dB with the f16 weights
and about 80–100 dB with the f32 weights.

f16 rounding is not spread evenly. In one 30 s music clip, one second drops to about
14 dB while the rest stay near 60 dB. encodec.cpp dips in the same place, to 35 dB. With
the f32 weights, every second stays at about 80 dB.

## Benchmarks

To reproduce:

```bash
bench/compare_encodec_cpp.sh input.ecdc [n_threads] [n_runs]
```

The script clones encodec.cpp at a pinned commit into `bench/work/` and applies
`bench/encodec_cpp.patch` (see below). It then builds both projects and prints wall
time, CPU time, peak memory and binary size.

Results from 2026-10-17 on an M1 (16 GB):

- encodec.cpp was at commit `83deb38` (ggml `c18f9bae`), built in Release mode with the
  CPU backend.
- The input was 6 kbps `.ecdc` clips cut from a music file.
- The machine was under heavy background load (load average 4–16), so treat the times as
  rough. Memory and sizes are exact.

**Wall time** (lower is better):

| Case | zencodec-c | zencodec-zig | encodec.cpp |
| --- | --- | --- | --- |
| 10 s audio, 1 thread\* | 1.67–2.04 s | 1.80–2.29 s | 1.94–2.20 s |
| 10 s audio, 4 threads | 2.3–2.9 s | 3.0–3.2 s | 2.6–3.3 s |
| 30 s audio, 4 threads | 8.1–9.3 s | 8.7–9.7 s | 13.6–15.4 s |
| 30 s audio, 8 threads | 6.5–6.6 s | 9.2–10.1 s | 37–175 s |

\* Measured in a separate session at lower load (load average about 2.6), so compare
this row only within itself.

On a single core, zencodec-c is about 10–15% faster than encodec.cpp, and zencodec-zig
is in between. At 4 threads, encodec.cpp uses about twice the
CPU time, and at 8 threads it slows down sharply. Its graph has one node per LSTM step
and frame, and its threads synchronize at every node. Its Metal backend (`-g 99`) was
slower than its CPU backend: 9.5 s for 10 s of audio.

**Peak memory** (`/usr/bin/time -l`, peak footprint):

| Audio length | zencodec-c | zencodec-zig | encodec.cpp |
| --- | --- | --- | --- |
| 10 s | 71 MB | 92 MB | 377 MB |
| 30 s | 72 MB | 93 MB | 1.09 GB |

zencodec decodes in fixed-size chunks, so its memory does not change with the length of
the audio. encodec.cpp builds one graph for the whole file, so its memory grows with the
length of the audio.

**Size** (macOS arm64):

| | Binary (stripped) | Weights file |
| --- | --- | --- |
| zencodec-c | 70 KB | 23 MB (decoder only, f16) |
| zencodec-zig | 323 KB | same file |
| encodec.cpp `decompress`, CPU only | 905 KB | 46.5 MB (encoder + decoder) |
| encodec.cpp `decompress`, default build with Metal | 1.32 MB | same file |

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
  - Register-blocked, GEMM-style kernels for the conv, transposed-conv and LSTM loops.
    These loops are bound by loads and stores today, at about 2 GMAC/s per core.
  - A persistent thread pool.
- **Memory:** keep the weights in f16 and convert them inside the kernels, or use int8,
  to halve or quarter the resident size.
- **Size:** int8 weights, so the shipped asset is about 12 MB.
- **Devices:** test on Android and iOS hardware, and add sample apps.
- **Models:** the 48 kHz model, and the `--lm` entropy coder.

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

The file is not included in this repository. To create it:

```bash
python scripts/convert.py --use-f16      # → weights/encodec_24khz_decoder_f16.bin (23 MB)
python scripts/convert.py                # → weights/encodec_24khz_decoder_f32.bin (46 MB)
```

The script downloads Meta's checkpoint (`encodec_24khz-d7cc33bc.th`) and checks its hash.
To use a checkpoint you already have, pass `--checkpoint path.th`; to choose the output
file, pass `--out`. The f32 file matches PyTorch to within 16-bit WAV rounding.

## Project layout

| Path | Purpose |
| --- | --- |
| `c/zencodec.{h,c}` | C17 decoder library: the two files to embed |
| `c/play.{h,c}`, `c/main.c` | C command-line tool and playback |
| `zig/src/zencodec.zig` | Zig decoder (`@Vector` SIMD, `std.Thread`) |
| `zig/src/play.zig`, `zig/src/main.zig` | Zig command-line tool and playback |
| `weights/` | Generated weights files (not in the repository; see `scripts/convert.py`) |
| `scripts/convert.py` | Downloads Meta's checkpoint and builds the weights file |
| `scripts/encode.py` | WAV → raw codes file, using the reference `encodec` package |
| `tests/compare.py` | Checks a CLI's output against the PyTorch decoder |
| `bench/` | Benchmark against encodec.cpp (script and patch) |
| `docs/` | Decoder architecture document and diagram |
| `build.zig` | Builds both CLIs, natively or cross-compiled |
| `LICENSE` | MIT License for the code |

## Credits & license

The decoder follows the structure of
[encodec.cpp](https://github.com/PABannier/encodec.cpp) by Pierre-Antoine Bannier, and
the model is [EnCodec](https://github.com/facebookresearch/encodec) by Meta AI.

The zencodec code is released under the [MIT License](LICENSE).

The EnCodec weights are released by Meta under
[CC-BY-NC 4.0](https://github.com/facebookresearch/encodec#license). For that reason this
repository does not include them. `scripts/convert.py` downloads them from Meta. Files
generated from them fall under the same non-commercial terms.
