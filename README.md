# zencodec — a fast, small EnCodec decoder by Wavebits

![EnCodec decoder](docs/encodec_decoder.png)

**Our mission is a decoder for Meta's [EnCodec](https://github.com/facebookresearch/encodec)
codes that is fast, uses little memory and fits in a small binary, so any application can
use it:** phone apps, servers, desktop tools and embedded boards. EnCodec
compresses speech and music to as little as 1.5 kbps, but turning the codes back into
sound normally takes Python with PyTorch. zencodec does it in plain C that needs nothing
beyond libc and threads, in memory that doesn't grow with the length of the audio, and it
can stream. Optionally, it includes [Vocos](#optional-vocos-decoder), a second decoder for
the same codes that is faster still.

## Speed, memory and size

30 s of 6 kbps speech, Apple M1, 1 thread, median of 5 interleaved runs:

| | zencodec | zencodec + Vocos | encodec.cpp | PyTorch `encodec` | PyTorch `vocos` |
| --- | --- | --- | --- | --- | --- |
| ⚡ Time | 0.97 s | **0.30 s** | 9.35 s | 5.00 s | 2.29 s |
| 🧠 Peak memory | 77 MB | **52 MB** | 1.09 GB | 2.17 GB | 501 MB |
| 📦 Size to ship | **73 KB** | 90 KB | 1.0 MB | 518 MB + Python | 522 MB + Python |
| 🎚️ Weights file | 23 MB | **20 MB** | 46.5 MB | 93 MB | 40 MB + EnCodec's 93 MB |
| 🔌 Dependencies | libc (+ Accelerate on Apple, optionally OpenBLAS on Linux) | same | ggml + C++ runtime | Python + PyTorch | Python + PyTorch + torchaudio |
| 📜 Weights license | CC-BY-NC 4.0 (Meta) | MIT (Vocos) ¹ | CC-BY-NC 4.0 | CC-BY-NC 4.0 | MIT ¹ |
| 🎯 Match to PyTorch (SNR) | 60.6 dB | 56.8 dB | 53.7 dB | reference | reference |

¹ The Vocos checkpoint is MIT, but it embeds EnCodec's codebooks. See [License](#license).

- **Fast:** 5× faster than PyTorch and 10× faster than encodec.cpp. With Vocos, 30 s of
  audio takes 0.30 s, which is 100× faster than real time and 7.6× faster than PyTorch
  Vocos.
- **Little memory:** 14× less than encodec.cpp and 28× less than PyTorch. It stays the
  same for any length of audio, while theirs grows.
- **Small:** under 100 KB, 11–14× smaller than encodec.cpp.
- **Portable:** the same C runs everywhere, with an Apple Accelerate backend, hand-written
  NEON kernels for ARM phones and boards, an OpenBLAS backend for Linux servers, or
  portable C. Without Accelerate, the NEON backend takes 3.0 s (EnCodec) or 1.0 s (Vocos)
  here.

Times are for the whole process: loading the model, decoding and writing the WAV. The
machine had other work running (load average 6.6–8), so every time here is about 1.4×
what a quiet M1 gives. The columns are comparable with each other, since the runs were
interleaved. See [MORE.md](MORE.md#benchmarks) for every backend, 4 threads, music,
the method, and how to reproduce the numbers.

## Optional Vocos decoder

[Vocos](https://github.com/gemelo-ai/vocos) (`charactr/vocos-encodec-24khz`) turns the same
EnCodec codes into audio. It predicts a spectrum per frame and synthesizes the waveform
with an inverse FFT, so it needs fewer operations and less memory than EnCodec's decoder.
Its authors also report better perceived quality at the same bitrates.

| 30 s of speech, Apple M1, 1 thread | EnCodec decoder | Vocos decoder |
| --- | --- | --- |
| accelerate | 0.91 s | **0.26 s** (3.5× faster) |
| neon | 2.93 s | **0.96 s** (3.1× faster) |
| portable C | 7.50 s | **3.01 s** (2.5× faster) |
| Peak memory | 77 MB | **52 MB** |
| Binary size | 73 KB | 90 KB (+17 KB) |

These are decode times only, from the same run as the table above.

On 10 minutes of speech (accelerate, 1 thread), Vocos takes 6.6 s and EnCodec 16.7 s.
PyTorch Vocos takes 12–16 s and 2.3–2.5 GB. PyTorch EnCodec and encodec.cpp hold the whole
file in memory (2.17 GB and 1.09 GB for 30 s), so they were not run on it.

- **Build:** `zig build -Dvocos=true`, or compile `c/vocos.c` with `-DZENCODEC_VOCOS`. The
  weights file selects the decoder, so the API doesn't change.
- **Bitrates:** 1.5, 3, 6 and 12 kbps. Not 24 kbps.
- **Streaming:** Vocos looks 0.39 s (29 frames) ahead, so streamed audio comes out that
  much later. The output equals a whole-file decode.
- **Accuracy:** 83–94 dB SNR against PyTorch Vocos with f32 weights, 42–57 dB with f16.

Details: [MORE.md](MORE.md#vocos-decoder-optional) and [docs/vocos.md](docs/vocos.md).

## Quick start

Prebuilt `wb_decode` binaries for Linux and macOS are on the
[releases page](https://github.com/wavebits/encodec-decoders/releases). To build it:

```bash
zig build -Dstrip=true -Dvocos=true          # → zig-out/bin/wb_decode (Zig 0.17)
pip install torch numpy encodec              # for the weights and for encoding
python scripts/convert.py --use-f16          # → weights/encodec_24khz_decoder_f16.bin
python scripts/convert_vocos.py --use-f16    # → weights/vocos_encodec_24khz_f16.bin (optional)
encodec -b 6 input.wav input.ecdc            # compress with Meta's reference CLI
zig-out/bin/wb_decode decode weights/encodec_24khz_decoder_f16.bin input.ecdc out.wav
zig-out/bin/wb_decode decode weights/vocos_encodec_24khz_f16.bin   input.ecdc out.wav
zig-out/bin/wb_decode play   weights/vocos_encodec_24khz_f16.bin   input.ecdc
```

In your own code:

```c
zencodec_model * m = zencodec_load("weights/vocos_encodec_24khz_f16.bin"); // or the EnCodec file
zencodec_decode(m, codes /* int32 [n_q][n_frames] */, n_q, n_frames, audio, /*n_threads=*/4);
zencodec_free(m);
```

To embed it, add the files in `c/` except `main.c` and `play.c` (the CLI). See
[MORE.md](MORE.md#embedding-on-phones-and-devices) for phones, the streaming API, and the
build options. [docs/architecture.md](docs/architecture.md) explains the decoder layer by
layer.

## License

The code is MIT. The decoder you choose decides the license of the weights you ship:

| Decoder | Weights | License |
| --- | --- | --- |
| EnCodec (default) | Meta's `encodec_24khz` | **[CC-BY-NC 4.0](https://github.com/facebookresearch/encodec#license)**: non-commercial only |
| Vocos (optional) | `charactr/vocos-encodec-24khz` | **[MIT](https://huggingface.co/charactr/vocos-encodec-24khz)**, but it contains EnCodec's codebooks, which come from Meta's CC-BY-NC 4.0 weights |

Neither weights file is in this repository: the scripts download them. Check both licenses
before shipping commercially. See [MORE.md](MORE.md#credits--license).
