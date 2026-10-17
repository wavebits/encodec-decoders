"""Decode an .ecdc file with Meta's reference PyTorch `encodec` package (benchmark baseline).

Usage:
    pip install encodec torch
    python bench/python_decode.py in.ecdc out.wav [-t n_threads]
"""
import argparse
import wave

import numpy as np
import torch
from encodec.compress import decompress


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("-t", "--threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    with open(args.input, "rb") as f:
        data = f.read()
    with torch.no_grad():
        wav, sr = decompress(data)  # loads the model, then decodes

    pcm = (wav[0].clamp(-1, 1).numpy() * 32767).astype(np.int16)
    with wave.open(args.output, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


if __name__ == "__main__":
    main()
