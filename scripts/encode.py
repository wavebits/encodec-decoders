"""Encode a WAV file into a zencodec codes file using the reference `encodec` package.

Codes file (little-endian): int32 n_q, int32 n_frames, int32 codes[n_q][n_frames]

Usage:
    pip install encodec torchaudio
    python scripts/encode.py input.wav codes.bin --bandwidth 6
"""
import argparse
import struct

import numpy as np
import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--bandwidth", type=float, default=6.0, choices=[1.5, 3.0, 6.0, 12.0, 24.0])
    args = parser.parse_args()

    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(args.bandwidth)

    wav, sr = torchaudio.load(args.input)
    wav = convert_audio(wav, sr, model.sample_rate, model.channels).unsqueeze(0)

    with torch.no_grad():
        frames = model.encode(wav)
    codes = torch.cat([c for c, _ in frames], dim=-1)[0].numpy().astype(np.int32)  # [n_q, T]

    with open(args.output, "wb") as f:
        f.write(struct.pack("ii", *codes.shape))
        codes.tofile(f)
    print(f"wrote {args.output}: {codes.shape[0]} codebooks x {codes.shape[1]} frames")


if __name__ == "__main__":
    main()
