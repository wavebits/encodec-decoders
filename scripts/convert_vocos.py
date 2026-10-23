"""Convert the Vocos checkpoint for EnCodec codes into a weights file for zencodec.

Vocos (https://github.com/gemelo-ai/vocos) is an alternative decoder for the same
EnCodec 24 kHz codes. Its checkpoint, charactr/vocos-encodec-24khz on Hugging Face,
holds the backbone, the ISTFT head, and the first 16 EnCodec codebooks.

Layout (little-endian), like scripts/convert.py but with its own magic and hparams:
    - magic                      int32  0x73636f76 ("vocs" in the file)
    - hparams                    int32[10]
        input_channels, dim, intermediate_dim, num_layers, n_fft, hop_length,
        n_bins, n_codebooks, sample_rate, ftype
    - tensors, each:
        n_dims, name_len, ftype  int32[3]   (ftype: 0 = f32, 1 = f16)
        dims                     int32[n_dims]  (innermost first)
        name                     char[name_len]
        data                     f32 / f16, row-major in PyTorch order

Matrices (codebooks, embed conv, pointwise layers, head) are f16 with --use-f16.
Per-channel tensors (biases, norms, layer scales, depthwise conv) are written 1-D in f32.
The ISTFT window is not stored: it is a Hann window.

Usage:
    pip install torch numpy
    python scripts/convert_vocos.py --use-f16   # -> weights/vocos_encodec_24khz_f16.bin (20 MB)
    python scripts/convert_vocos.py             # -> weights/vocos_encodec_24khz_f32.bin (40 MB)
    python scripts/convert_vocos.py --checkpoint pytorch_model.bin --out my.bin --use-f16
"""
import argparse
import hashlib
import struct
import urllib.request
from pathlib import Path

import numpy as np
import torch

REPO = "charactr/vocos-encodec-24khz"
REVISION = "4e61d082c08045a4c11e5b148ad93b1d0c591a14"
CHECKPOINT_URL = f"https://huggingface.co/{REPO}/resolve/{REVISION}/pytorch_model.bin"
CHECKPOINT_SHA256 = "7e95bb260b74a1bfc43c52d355831c951acb81c8960e9c62b79bd2b3ab1e3a90"
MAGIC = 0x73636F76
HPARAMS = [128, 384, 1152, 8, 1280, 320, 1024, 16, 24000]


def download(dest):
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {CHECKPOINT_URL}")
        urllib.request.urlretrieve(CHECKPOINT_URL, dest)
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    if digest != CHECKPOINT_SHA256:
        raise SystemExit(f"error: {dest} has sha256 {digest}, expected {CHECKPOINT_SHA256}")
    return dest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help=f"path to {REPO}'s pytorch_model.bin (downloaded if omitted)")
    parser.add_argument("--out", type=Path, default=None,
                        help="output path (default: weights/vocos_encodec_24khz_{f16,f32}.bin)")
    parser.add_argument("--use-f16", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    if args.out is None:
        args.out = root / "weights" / f"vocos_encodec_24khz_{'f16' if args.use_f16 else 'f32'}.bin"
    ckpt = args.checkpoint or download(root / "weights" / "vocos-encodec-24khz.pytorch_model.bin")
    state = torch.load(ckpt, map_location="cpu")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_bytes = 0
    with open(args.out, "wb") as out:
        out.write(struct.pack("i", MAGIC))
        out.write(struct.pack("10i", *HPARAMS, int(args.use_f16)))
        for name, tensor in state.items():
            if name == "head.istft.window" or "encodec" in name:
                continue
            data = tensor.float().numpy()
            if data.ndim == 2 and ("norm.scale" in name or "norm.shift" in name):
                data = data.reshape(-1)  # AdaLayerNorm tables, [4][384]
            elif name.endswith("dwconv.weight"):
                data = data.reshape(-1)  # depthwise conv, [384][1][7]
            ftype = int(args.use_f16 and data.ndim >= 2)
            data = np.ascontiguousarray(data.astype(np.float16 if ftype else np.float32))

            encoded = name.encode("utf-8")
            out.write(struct.pack("iii", data.ndim, len(encoded), ftype))
            for dim in reversed(data.shape):
                out.write(struct.pack("i", dim))
            out.write(encoded)
            data.tofile(out)
            n_bytes += data.nbytes
            print(f"{name:45s} {str(tuple(data.shape)):16s} {'f16' if ftype else 'f32'}")
    print(f"\nwrote {args.out} ({n_bytes / 1e6:.1f} MB of tensor data)")


if __name__ == "__main__":
    main()
