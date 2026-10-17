"""Convert the EnCodec 24 kHz checkpoint into a decoder-only weights file.

The output uses the same binary layout as encodec.cpp's `ggml-model.bin`
(https://github.com/PABannier/encodec.cpp), so zencodec can also load full
encodec.cpp weight files. Only the quantizer codebooks and the decoder are
written here; encoder tensors are skipped.

Layout (little-endian):
    - magic                      int32  0x67676d6c ("ggml")
    - hparams                    int32[9]
        in_channels, hidden_dim, n_filters, kernel_size, residual_kernel_size,
        n_bins, bandwidth, sample_rate, ftype
    - tensors, each:
        n_dims, name_len, ftype  int32[3]   (ftype: 0 = f32, 1 = f16)
        dims                     int32[n_dims]  (innermost first)
        name                     char[name_len]
        data                     f32 / f16, row-major in PyTorch order

Weight-normalized convolutions are folded into a single `weight` tensor.

The weights are not distributed with zencodec (CC-BY-NC 4.0, see the README); this
script downloads Meta's checkpoint and builds the file locally.

Usage:
    pip install torch numpy
    python scripts/convert.py --use-f16      # -> weights/encodec_24khz_decoder_f16.bin (23 MB)
    python scripts/convert.py                # -> weights/encodec_24khz_decoder_f32.bin (46 MB)
    python scripts/convert.py --checkpoint encodec_24khz-d7cc33bc.th --out my.bin --use-f16
"""
import argparse
import struct
from pathlib import Path

import numpy as np
import torch

CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/encodec/v0/encodec_24khz-d7cc33bc.th"


def load_checkpoint(path):
    if path is None:
        return torch.hub.load_state_dict_from_url(CHECKPOINT_URL, map_location="cpu", check_hash=True)
    return torch.load(path, map_location="cpu")


def write_hparams(out, use_f16):
    # hardcoded: only the 24 kHz model is supported
    in_channels = 1
    hidden_dim = 128
    n_filters = 32
    kernel_size = 7
    residual_kernel_size = 3
    n_bins = 1024
    bandwidth = 24
    sr = 24000
    ftype = int(use_f16)
    out.write(struct.pack("9i", in_channels, hidden_dim, n_filters, kernel_size,
                          residual_kernel_size, n_bins, bandwidth, sr, ftype))


def write_tensors(checkpoint, out, use_f16):
    n_bytes = 0
    for name in checkpoint.keys():
        if not (name.startswith("decoder.") or name.startswith("quantizer.")):
            continue
        if "weight_g" in name:
            continue  # folded together with the matching weight_v
        if "inited" in name or "cluster_size" in name or "embed_avg" in name:
            continue  # training-only quantizer state

        data = checkpoint[name]
        if "weight_v" in name:
            base = name.rsplit(".", 1)[0]
            data = torch._weight_norm(data, checkpoint[base + ".weight_g"], dim=0).numpy()
            name = base + ".weight"
        else:
            data = data.numpy().squeeze()

        is_matrix = "weight" in name or "embed" in name
        ftype = 1 if (use_f16 and is_matrix) else 0
        data = data.astype(np.float16 if ftype else np.float32)

        encoded = name.encode("utf-8")
        out.write(struct.pack("iii", data.ndim, len(encoded), ftype))
        for dim in reversed(data.shape):
            out.write(struct.pack("i", dim))
        out.write(encoded)
        data.tofile(out)

        n_bytes += data.nbytes
        print(f"{name:55s} {str(tuple(data.shape)):20s} {'f16' if ftype else 'f32'}")
    return n_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="path to encodec_24khz-d7cc33bc.th (downloaded if omitted)")
    parser.add_argument("--out", type=str, default=None,
                        help="output path (default: weights/encodec_24khz_decoder_{f16,f32}.bin)")
    parser.add_argument("--use-f16", action="store_true")
    args = parser.parse_args()
    if args.out is None:
        root = Path(__file__).resolve().parent.parent
        args.out = root / "weights" / f"encodec_24khz_decoder_{'f16' if args.use_f16 else 'f32'}.bin"

    checkpoint = load_checkpoint(args.checkpoint)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as out:
        out.write(struct.pack("i", 0x67676D6C))
        write_hparams(out, args.use_f16)
        n_bytes = write_tensors(checkpoint, out, args.use_f16)

    print(f"\nwrote {out_path} ({n_bytes / 1e6:.1f} MB of tensor data)")


if __name__ == "__main__":
    main()
