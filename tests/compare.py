"""Compare the wb_decode CLI against the reference PyTorch EnCodec decoder, or,
with --vocos, the C Vocos decoder against the reference PyTorch Vocos.

Usage:
    python tests/compare.py zig-out/bin/wb_decode weights/encodec_24khz_decoder_f16.bin
    python tests/compare.py zig-out/bin/wb_decode weights/vocos_encodec_24khz_f16.bin --vocos
        (needs a build with -Dvocos=true, and pip install vocos)
"""
import argparse
import struct
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
import torch
from encodec import EncodecModel


def reference_codes(model, seconds, bandwidth):
    t = torch.arange(int(seconds * model.sample_rate)) / model.sample_rate
    wav = 0.3 * torch.sin(2 * np.pi * (220 + 400 * t) * t) + 0.05 * torch.randn_like(t)
    model.set_target_bandwidth(bandwidth)
    with torch.no_grad():
        frames = model.encode(wav[None, None])
    return torch.cat([c for c, _ in frames], dim=-1)


def reference_decode(model, vocos, codes):
    with torch.no_grad():
        if vocos is None:
            return model.decode([(codes, None)])[0, 0].numpy()
        bandwidth_id = torch.tensor([[2, 4, 8, 16].index(codes.shape[1])])
        return vocos.decode(vocos.codes_to_features(codes[0]), bandwidth_id=bandwidth_id)[0].numpy()


def run_case(binary, weights, model, vocos, codes, tmp):
    codes_path, wav_path = tmp / "codes.bin", tmp / "out.wav"
    arr = codes[0].numpy().astype(np.int32)
    with open(codes_path, "wb") as f:
        f.write(struct.pack("ii", *arr.shape))
        arr.tofile(f)
    subprocess.run([str(binary), "decode", str(weights), str(codes_path), str(wav_path)], check=True,
                   stderr=subprocess.DEVNULL)

    with wave.open(str(wav_path)) as w:
        ours = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32767

    ref = reference_decode(model, vocos, codes)
    ref = np.round(np.clip(ref, -1, 1) * 32767) / 32767

    assert ours.shape == ref.shape, (ours.shape, ref.shape)
    snr = 10 * np.log10(np.sum(ref**2) / max(np.sum((ours - ref) ** 2), 1e-20))
    return snr, np.max(np.abs(ours - ref))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("--min-snr", type=float, default=30.0)
    parser.add_argument("--vocos", action="store_true", help="compare against PyTorch Vocos")
    args = parser.parse_args()

    torch.manual_seed(0)
    model = EncodecModel.encodec_model_24khz().eval()
    vocos = None
    cases = [(2.0, 1.5), (2.0, 6.0), (2.0, 24.0), (0.01, 6.0)]
    if args.vocos:
        from vocos import Vocos
        vocos = Vocos.from_pretrained("charactr/vocos-encodec-24khz").eval()
        # 1.5 to 12 kbps; 0.4 s is shorter than the lookahead, 5 s spans several chunks
        cases = [(2.0, 1.5), (2.0, 3.0), (5.0, 6.0), (2.0, 12.0), (0.4, 6.0), (0.01, 6.0)]
    failed = False
    with tempfile.TemporaryDirectory() as tmp:
        for seconds, bw in cases:
            codes = reference_codes(model, seconds, bw)
            snr, max_err = run_case(args.binary, args.weights, model, vocos, codes, Path(tmp))
            ok = snr >= args.min_snr
            failed |= not ok
            print(f"{'PASS' if ok else 'FAIL'}  {seconds:5.2f}s @ {bw:4.1f} kbps "
                  f"(n_q={codes.shape[1]:2d}, frames={codes.shape[2]:3d})  "
                  f"SNR {snr:6.1f} dB  max|err| {max_err:.5f}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
