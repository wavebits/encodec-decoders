# EnCodec decoder architecture

This document covers EnCodec's own decoder. The optional Vocos decoder for the same
codes is described in [vocos.md](vocos.md).

zencodec runs the decoder of Meta's **EnCodec** neural audio codec (Défossez, Copet,
Synnaeve & Adi, *High Fidelity Neural Audio Compression*, 2022;
[facebookresearch/encodec](https://github.com/facebookresearch/encodec)).

The decoder turns a sequence of discrete RVQ codes back into a waveform. The numbers below
are the defaults of the released 24 kHz model, which is a `SEANetDecoder` with:

- `n_filters=32`, `dimension=128`, `ratios=[8,5,4,2]`
- `lstm=2`, `n_residual_layers=1`, `true_skip=False`
- `causal=True`, `norm="weight_norm"`

![EnCodec decoder](encodec_decoder.png)

The ResBlock shortcut is a 1×1 Conv1d (C → C), not an identity: the released decoder
uses `true_skip=False`, and the checkpoint stores those weights as
`decoder.model.{4,7,10,13}.shortcut`.

## Shape flow

| Stage | Layer | Hyper-parameters | Output (C × L) |
|---|---|---|---|
| 0 | RVQ codes | `n_q` codebooks, 1024 entries each, 75 frames/s | `n_q × T` (int) |
| 1 | RVQ decode | `z = Σᵢ codebookᵢ[cᵢ]` | `128 × T` |
| 2 | Conv1d | k = 7, 128 → 512 | `512 × T` |
| 3 | LSTM × 2 | hidden 512, unidirectional, `y = LSTM(x) + x` | `512 × T` |
| 4 | Block 1: ELU → ConvTranspose1d → ResBlock | k = 16, s = 8, 512 → 256 | `256 × 8T` |
| 5 | Block 2: ELU → ConvTranspose1d → ResBlock | k = 10, s = 5, 256 → 128 | `128 × 40T` |
| 6 | Block 3: ELU → ConvTranspose1d → ResBlock | k = 8, s = 4, 128 → 64 | `64 × 160T` |
| 7 | Block 4: ELU → ConvTranspose1d → ResBlock | k = 4, s = 2, 64 → 32 | `32 × 320T` |
| 8 | ELU → Conv1d | k = 7, 32 → 1 (no output activation) | `1 × 320T` |

`T` is the number of latent frames. The output has 320 · T samples at 24 kHz, because the
hop length is 8 · 5 · 4 · 2 = 320.

## Layer by layer

**RVQ decode.** The quantizer is a residual vector quantizer. Codebook *i* quantises
what codebooks 0…*i*−1 left over, so the latent is rebuilt by *summing* the selected
embeddings.

The number of codebooks `n_q` sets the bitrate. Each code is 10 bits at 75 Hz, which is
0.75 kbps per codebook:

| Codebooks | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| Bitrate | 1.5 kbps | 3 kbps | 6 kbps | 12 kbps | 24 kbps |

The same decoder weights serve every bitrate; fewer codebooks just give a coarser `z`.

**Input Conv1d (k = 7).** Projects the 128-d latent to 512 channels, the decoder's widest
point, and mixes in a little temporal context before the recurrent layers.

**2-layer LSTM with residual skip.** This is the only part with a long (unbounded)
receptive field. The convolutional stack only sees a handful of frames, so the LSTM
carries context across frames instead. The skip connection (`y = LSTM(x) + x`) keeps the
convolutional path usable even when the LSTM contributes little, and stabilises
training.

**Four upsampling blocks.** Each block is `ELU → ConvTranspose1d → ResBlock`:

- ***ELU*** (α = 1) comes before every conv in SEANet (pre-activation layout).
- ***ConvTranspose1d*** upsamples time by the stride and halves the channel count.
  - Its kernel is 2 × stride, so consecutive output windows overlap by exactly half. This
    reduces the checkerboard artefacts of transposed convolutions.
  - The strides `(8, 5, 4, 2)` are the encoder's `(2, 4, 5, 8)` reversed, so the decoder
    mirrors the encoder level by level.
- ***ResBlock*** is a bottleneck residual unit on the upsampled signal:
  `ELU → Conv1d(k=3, C→C/2) → ELU → Conv1d(k=1, C/2→C)`. Its output is added to a 1×1
  Conv1d shortcut of the block input.
  - Dilation is 1, because the released models use a single residual layer per block
    (`dilation_base**j` with `j = 0`).
  - Most of each level's detail synthesis happens here.

**Output ELU + Conv1d (k = 7).** Collapses the 32 channels to a single waveform channel.
There is no `tanh` or other output non-linearity: the model learns the waveform directly
from multi-scale spectrogram and adversarial losses.

## Streaming

- Every convolution is wrapped as `SConv1d` / `SConvTranspose1d`. That means weight-norm,
  **causal (left-only) padding**, and, for the transposed convs, trimming the overlap on
  the right.
  - So the 24 kHz decoder has no look-ahead. It can run frame by frame from a small cached
    state: the LSTM state and a few samples of conv history.
- Channel widths follow `n_filters · 2ⁱ`: 512 at the latent side, 32 at the waveform side.
  The decoder is widest where time is coarsest, which keeps it cheap to run.
- The 48 kHz model has the same topology but differs in four ways:
  - It outputs **2 channels** (stereo).
  - It is **non-causal**.
  - It uses `time_group_norm` instead of weight-norm.
  - It processes 1 s segments with overlap.

## How zencodec implements it

The decoder follows the stages above one to one, as plain loops over `[C][T]` float
buffers. There is no graph and no tensor library. Names in the table are from
`c/zencodec.c`.

| Stage | Function | Notes |
|---|---|---|
| Load | `zencodec_load`, `bind` | Weight-norm is already folded into the weights by `scripts/convert.py`. f16 is converted to f32 on load, unless the build keeps weights in f16 (`-Df16=true`, `ZENCODEC_F16_WEIGHTS`); then the kernels widen them as they read them. The LSTM's gate rows are reordered (`lstm_reorder`) so the recurrence can be split by hidden unit. |
| RVQ decode | `encodec_decode_chunk` | Sums `n_q` codebook rows per frame. |
| Conv1d | `conv1d` | A GEMM per tap, run by the backend. `hist` keeps the last `K − 1` input samples between chunks. The first chunk is reflect-padded, as in EnCodec. |
| LSTM | `lstm_forward`, `lstm_step` | The input projection `W_ih·x` for all frames is one GEMM. The recurrence runs frame by frame, each frame's `W_hh·h` and gates split across threads by hidden unit ([threading.md](threading.md)). `h` and `c` carry over between chunks. |
| ConvTranspose1d | `conv_transpose1d` | One GEMM for all taps, then an overlap-add parallel over output channels. `tail` keeps the `K − stride` overlap samples for the next chunk. |
| ResBlock | `encodec_decode_chunk` | `conv1d` (k = 3, with history) → `conv1d` (k = 1), plus the 1×1 `shortcut` conv. |
| ELU | `elu` | In place. |

Because each layer keeps its own small state, `zencodec_decode` is just a loop over
75-frame (1 s) chunks of `zencodec_stream_push`. Memory therefore does not depend on
the length of the input. The output matches a single full-length decode as long as the
first chunk has at least 7 frames.

All parallel work runs on one persistent thread pool per stream (`c/pool.c`). The
[threading document](threading.md) explains the pool and the parallel recurrence. For
what limits speed now, see the [roadmap](../MORE.md#roadmap).

## Reference usage (PyTorch)

```python
from encodec import EncodecModel

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)            # → n_q = 8 codebooks

# codes: [B, n_q, T] (int64), produced by the encoder / your own model
z   = model.quantizer.decode(codes.transpose(0, 1))   # [B, 128, T]
wav = model.decoder(z)                                # [B, 1, 320·T] at 24 kHz
```
