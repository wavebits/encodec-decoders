"""Decode an .ecdc file with Meta's reference PyTorch `encodec` package (benchmark baseline),
or with --vocos, with the reference PyTorch Vocos (charactr/vocos-encodec-24khz).

Prints the model-load and decode times to stderr, like the other decoders.

Usage:
    pip install encodec torch         (and vocos, for --vocos)
    python bench/python_decode.py in.ecdc out.wav [-t n_threads] [--float] [--vocos]
"""
import argparse
import struct
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)  # torch weight_norm deprecation
t_start = time.perf_counter()
import numpy as np  # noqa: E402
import torch  # noqa: E402
from encodec import EncodecModel, binary  # noqa: E402


def write_wav(path, samples, sr, as_float):
    if as_float:  # IEEE float WAV (format 3), used as the accuracy reference
        data = samples.astype("<f4").tobytes()
        fmt = (3, 1, sr, sr * 4, 4, 32)
    else:
        data = (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
        fmt = (1, 1, sr, sr * 2, 2, 16)
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE")
        f.write(b"fmt " + struct.pack("<IHHIIHH", 16, *fmt))
        f.write(b"data" + struct.pack("<I", len(data)) + data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("-t", "--threads", type=int, default=4)
    parser.add_argument("--float", action="store_true", help="write a 32-bit float WAV")
    parser.add_argument("--vocos", action="store_true", help="decode with Vocos")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    t0 = time.perf_counter()
    if args.vocos:
        from vocos import Vocos
        vocos = Vocos.from_pretrained("charactr/vocos-encodec-24khz").eval()
        model = vocos.feature_extractor.encodec  # for its sample rate and bit unpacking
    else:
        model = EncodecModel.encodec_model_24khz()
    t1 = time.perf_counter()

    # same steps as encodec.compress.decompress_from_file, without its model reload
    with open(args.input, "rb") as f:
        meta = binary.read_ecdc_header(f)
        if meta["m"] != "encodec_24khz" or meta["lm"]:
            sys.exit("error: only encodec_24khz files without --lm are supported")
        n_samples, n_q = meta["al"], meta["nc"]
        n_frames = int(np.ceil(n_samples / model.sample_rate * model.frame_rate))
        unpacker = binary.BitUnpacker(model.bits_per_codebook, f)
        codes = [[unpacker.pull() for _ in range(n_q)] for _ in range(n_frames)]
    frame = torch.tensor(codes, dtype=torch.long).T[None]  # [1, n_q, T]
    t2 = time.perf_counter()
    with torch.no_grad():
        if args.vocos:
            if n_q not in (2, 4, 8, 16):
                sys.exit("error: Vocos takes 2, 4, 8 or 16 codebooks (1.5 to 12 kbps)")
            bandwidth_id = torch.tensor([[2, 4, 8, 16].index(n_q)])
            wav = vocos.decode(vocos.codes_to_features(frame[0]), bandwidth_id=bandwidth_id)[0, :n_samples]
        else:
            wav = model.decode([(frame, None)])[0, 0, :n_samples]
    t3 = time.perf_counter()

    write_wav(args.output, wav.numpy(), model.sample_rate, args.float)
    print(f"import {1e3 * (t0 - t_start):.1f} ms, load {1e3 * (t1 - t0):.1f} ms, "
          f"unpack {1e3 * (t2 - t1):.1f} ms, decode {1e3 * (t3 - t2):.1f} ms", file=sys.stderr)


if __name__ == "__main__":
    main()
