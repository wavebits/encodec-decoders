# Vocos decoder

[Vocos](https://github.com/gemelo-ai/vocos) (Siuzdak, *Vocos: Closing the gap between
time-domain and Fourier-based neural vocoders for high-quality audio synthesis*, 2023) is a
second decoder for the same EnCodec codes. The `charactr/vocos-encodec-24khz` checkpoint
was trained to turn EnCodec 24 kHz codes into audio. Instead of upsampling step by step
like EnCodec's decoder, it predicts one short-time Fourier spectrum per frame and gets
the waveform with an inverse FFT.

It is optional: `zig build -Dvocos=true`, or compile `c/vocos.c` with `-DZENCODEC_VOCOS`.
`zencodec_load` picks the decoder from the weights file.

## Shape flow

| Stage | Layer | Hyper-parameters | Output (C × L) |
|---|---|---|---|
| 0 | EnCodec codes | `n_q` = 2, 4, 8 or 16 codebooks (1.5–12 kbps), 75 frames/s | `n_q × T` (int) |
| 1 | Features | sum of the `n_q` codebook embeddings (EnCodec's own codebooks) | `128 × T` |
| 2 | Conv1d + AdaLayerNorm | k = 7, 128 → 384 | `384 × T` |
| 3 | ConvNeXt block × 8 | depthwise conv k = 7 → AdaLayerNorm → 384 → 1152 → GELU → 384, × γ, + input | `384 × T` |
| 4 | LayerNorm → Linear | 384 → 1282 = 641 log-magnitudes + 641 phases | `1282 × T` |
| 5 | ISTFT | n_fft 1280, hop 320, Hann window, "same" padding | `1 × 320T` |

- **AdaLayerNorm** is a LayerNorm whose scale and shift come from a 4-entry table indexed
  by the bandwidth (`n_q` 2, 4, 8, 16). That is why only those four `n_q` values work, and
  not 24 kbps (32 codebooks).
- **The head** gives `mag = min(exp(m), 100)` and phase `p`, so `S = mag · (cos p + i sin p)`.
- **The ISTFT** inverse-FFTs each spectrum, multiplies it by the window, overlap-adds the
  frames 320 samples apart, divides by the summed squared window, and drops 480 samples
  at each end.

Vocos needs about 16 MFLOP per frame, against about 40 for EnCodec's decoder, and it has
no recurrence, so every layer is parallel over frames.

## Streaming

Vocos is not causal: its convolutions are centered (3 frames each side, zero-padded at
the ends of the stream), and its ISTFT overlaps 4 frames. A frame's audio depends on the
29 frames after it:

- **Backbone:** each of the 9 convolutions keeps the last 6 frames of its input between
  chunks and outputs a frame once it has the 3 frames after it. Together that is 27
  frames of delay. At the end of the stream, each convolution gets its 3 frames of zero
  padding.
- **ISTFT:** the stream keeps the overlap-add buffer (1280 samples) and the matching
  window envelope. An output sample is final once no later frame overlaps it, which is
  1.5 frames later.

`zencodec_stream_push` returns the samples that are final, which with Vocos is
`(frames pushed − 28.5) × 320` in total. `zencodec_stream_flush` returns the last
`28.5 × 320` samples at the end. Nothing is recomputed, and the output is the same as
decoding the whole input at once, for any chunk sizes. Streaming latency is therefore
about 0.38 s of audio, plus the time to decode a chunk. `zencodec_stream_decode`, which
must return a full chunk of audio per call, only works for EnCodec.

## How zencodec implements it

| Stage | Function (`c/vocos.c`) | Notes |
|---|---|---|
| Load | `vocos_load`, `bind` | `scripts/convert_vocos.py` writes the checkpoint in the same tensor layout as the EnCodec file, with the magic `vocs`. Matrices go through the backend (`zc_mat`). The head is padded from 1282 to 1288 rows so the neon backend can use its own kernels. |
| Features | `vocos_stream_push` | Sums `n_q` rows of the codebook table. |
| Embed conv | `linear` | One 7-tap GEMM over the input with context (`with_context`). |
| ConvNeXt | `convnext_forward` | Depthwise conv (`dwconv_row`, parallel over channels), LayerNorm (`layer_norm`, parallel over 16-frame column blocks), two GEMMs with GELU between (`gelu_row`, using an erf approximation within 1.5e-7). |
| Head | `linear` | One GEMM. |
| ISTFT | `istft`, `istft_frame`, `irfft` | Per frame in parallel: magnitude and phase, then a 1280-point inverse real FFT done as one 640-point complex FFT (radix 2 down to 5-point DFTs). Then a serial overlap-add. |

The shared pieces (weight loading, activation buffers, `exp` approximation, profiling)
are in `c/internal.h`.

## Reference usage (PyTorch)

```python
import torch
from vocos import Vocos

vocos = Vocos.from_pretrained("charactr/vocos-encodec-24khz")
# codes: [n_q, T] (int64), n_q in (2, 4, 8, 16)
features = vocos.codes_to_features(codes)
bandwidth_id = torch.tensor([[2, 4, 8, 16].index(codes.shape[0])])
wav = vocos.decode(features, bandwidth_id=bandwidth_id)   # [1, 320·T] at 24 kHz
```
