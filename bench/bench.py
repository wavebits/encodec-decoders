"""Benchmark the zencodec decoders against encodec.cpp and PyTorch, and record every metric.

For each decoder, input file and thread count it measures:
  - wall time of the whole process, and the decode time the decoder reports itself
  - CPU time (user, sys)
  - peak memory
  - binary size (or install size, for PyTorch)
  - accuracy: SNR against the PyTorch decoder, overall and for the worst second

Runs are interleaved (every decoder once, then again) so background load affects all
of them alike. Results go to:
  - the terminal, as a Markdown table
  - bench/results/<date>_<commit>.md, with the machine description
  - bench/results/history.csv, one row per decoder/input/thread count, appended over time

With --vocos, it also builds wb_decode with the Vocos decoder (-Dvocos=true) for each
backend, runs it with the Vocos weights, and adds PyTorch Vocos, which is the accuracy
reference for those rows.

Usage:
    python bench/bench.py in.ecdc [more.ecdc ...] [-t 1,4] [-r 3]
        [--backends ref,neon,accelerate,neon+f16,ref+f16] [--baseline <git rev>]
        [--vocos] [--no-encodec-cpp] [--no-pytorch] [--no-accuracy] [--note "text"]

Make inputs with `encodec -b 6 input.wav in.ecdc` (24 kHz model, no --lm).
Needs zig. encodec.cpp needs git, cmake and a C++ compiler; PyTorch and the accuracy
check need `pip install encodec torch numpy`.
"""
import argparse
import csv
import datetime
import json
import os
import platform
import re
import shutil
import statistics
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "bench" / "work"
RESULTS = ROOT / "bench" / "results"
WEIGHTS = ROOT / "weights" / "encodec_24khz_decoder_f16.bin"
VOCOS_WEIGHTS = ROOT / "weights" / "vocos_encodec_24khz_f16.bin"
ENCODEC_CPP_COMMIT = "83deb38a757a12a58f993e34a44d32b3fd5d3893"
ENCODEC_MAX_NODES = 400000  # encodec.cpp builds one graph per file; raise its node cap
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/encodec/v0/encodec_24khz-d7cc33bc.th"
SAMPLE_RATE = 24000


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, shell=True).stdout.strip()
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# machine description

def system_info():
    info = {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "commit": sh(f"git -C {ROOT} rev-parse --short HEAD"),
        "dirty": bool(sh(f"git -C {ROOT} status --porcelain --untracked-files=no")),
        "os": f"{platform.system()} {platform.release()}",
        "arch": platform.machine(),
        "cpus": os.cpu_count(),
        "zig": sh("zig version"),
    }
    if platform.system() == "Darwin":
        info["machine"] = sh("sysctl -n hw.model")
        info["cpu"] = sh("sysctl -n machdep.cpu.brand_string")
        info["memory_gb"] = round(int(sh("sysctl -n hw.memsize") or 0) / 2**30)
        info["os"] = f"macOS {sh('sw_vers -productVersion')}"
    else:
        model = re.search(r"model name\s*:\s*(.*)", Path("/proc/cpuinfo").read_text()) \
            if Path("/proc/cpuinfo").exists() else None
        info["cpu"] = model.group(1) if model else platform.processor()
        info["machine"] = sh("cat /sys/firmware/devicetree/base/model 2>/dev/null") or info["cpu"]
        mem = re.search(r"MemTotal:\s*(\d+)", Path("/proc/meminfo").read_text()) \
            if Path("/proc/meminfo").exists() else None
        info["memory_gb"] = round(int(mem.group(1)) / 2**20) if mem else None
    return info


def load_average():
    try:
        return round(os.getloadavg()[0], 2)
    except OSError:
        return None


# ---------------------------------------------------------------------------
# builds

def default_backends():
    if platform.system() == "Darwin":
        if platform.machine() == "arm64":
            return ["ref", "neon", "accelerate", "neon+f16", "ref+f16"]
        return ["ref", "accelerate", "ref+f16"]
    return ["ref", "neon", "neon+f16", "ref+f16"] if platform.machine() in ("aarch64", "arm64") else ["ref", "ref+f16"]


def zig_flags(backend):
    """Build flags for a backend name such as "neon" or "neon+f16" (f16 weights in memory)."""
    name, _, variant = backend.partition("+")
    return [f"-Dbackend={name}"] + (["-Df16=true"] if variant == "f16" else [])


def cli_exe(prefix):
    """The CLI in a build prefix; it was named wb_decode-c, and before that zencodec-c."""
    return next(p for p in (prefix / "bin").iterdir() if p.name in ("wb_decode", "wb_decode-c", "zencodec-c"))


def build_zencodec(backends, vocos):
    """Builds wb_decode once per backend, and with `vocos` once more per backend with the
    Vocos decoder. Returns decoder specs."""
    decoders = []
    for backend in backends:
        prefix = WORK / f"build-{backend}"
        run(["zig", "build", *zig_flags(backend), "-Dstrip=true", "-p", str(prefix)], cwd=ROOT)
        decoders.append(cli_decoder(f"wb_decode ({backend})", cli_exe(prefix), backend))
    for backend in backends if vocos else []:
        prefix = WORK / f"build-{backend}-vocos"
        run(["zig", "build", *zig_flags(backend), "-Dvocos=true", "-Dstrip=true", "-p", str(prefix)], cwd=ROOT)
        decoders.append(cli_decoder(f"wb_decode vocos ({backend})", cli_exe(prefix),
                                    f"{backend}+vocos", VOCOS_WEIGHTS, "vocos"))
    return decoders


def build_baseline(rev, backends):
    """Builds wb_decode from an earlier git revision, once per backend if that revision
    has backends, for before/after comparisons. Returns decoder specs."""
    sha = sh(f"git -C {ROOT} rev-parse --short {rev}")
    if not sha:
        sys.exit(f"error: unknown git revision {rev}")
    tree = WORK / f"rev-{sha}"
    if not tree.exists():
        tree.mkdir(parents=True)
        archive = subprocess.run(["git", "-C", str(ROOT), "archive", sha], check=True, capture_output=True).stdout
        run(["tar", "-x", "-C", str(tree)], input=archive)
    if "backend" not in (tree / "build.zig").read_text():  # before the backends existed
        run(["zig", "build", "-Dstrip=true"], cwd=tree)
        return [cli_decoder(f"wb_decode @{sha}", cli_exe(tree / "zig-out"), f"@{sha}")]
    decoders = []
    has_f16 = '"f16"' in (tree / "build.zig").read_text()
    for backend in backends:
        if "+f16" in backend and not has_f16:
            continue
        prefix = tree / f"build-{backend}"
        run(["zig", "build", *zig_flags(backend), "-Dstrip=true", "-p", str(prefix)], cwd=tree)
        decoders.append(cli_decoder(f"wb_decode ({backend}) @{sha}", cli_exe(prefix), f"{backend}@{sha}"))
    return decoders


def cli_decoder(name, exe, backend, weights=WEIGHTS, ref="encodec"):
    """`ref`: the PyTorch decoder whose output this one is compared with."""
    return {
        "name": name, "backend": backend, "size": exe.stat().st_size, "ref": ref,
        "cmd": lambda inp, out, t, exe=exe: [str(exe), "decode", str(weights), str(inp), str(out), "-t", str(t)],
        "decode_re": r"decode ([\d.]+) ms",
    }


def build_encodec_cpp():
    src = WORK / "encodec.cpp"
    if not src.exists():
        run(["git", "clone", "--recursive", "https://github.com/PABannier/encodec.cpp.git", str(src)])
        run(["git", "-C", str(src), "checkout", "-q", ENCODEC_CPP_COMMIT])
        run(["git", "-C", str(src), "submodule", "update", "-q", "--init", "--recursive"])
        run(["git", "-C", str(src), "apply", str(ROOT / "bench" / "encodec_cpp.patch")])
    weights = WORK / "ggml_weights" / "ggml-model.bin"
    if not weights.exists():
        ckpt_dir = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch")) / "hub" / "checkpoints"
        ckpt = ckpt_dir / "encodec_24khz-d7cc33bc.th"
        if not ckpt.exists():
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            run(["curl", "-L", "-o", str(ckpt), CHECKPOINT_URL])
        run([sys.executable, "convert.py", "--dir-model", str(ckpt_dir), "--out-dir", str(weights.parent),
             "--use-f16"], cwd=src)
    build = src / "build-cpu"
    run(["cmake", "-S", str(src), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release", "-DGGML_METAL=OFF",
         f"-DCMAKE_CXX_FLAGS=-DENCODEC_MAX_NODES={ENCODEC_MAX_NODES}"], capture_output=True)
    run(["cmake", "--build", str(build), "-j", "--target", "decompress"], capture_output=True)
    exe = build / "bin" / "decompress"
    return {
        "name": "encodec.cpp", "backend": "ggml cpu", "size": exe.stat().st_size, "ref": "encodec",
        "cmd": lambda inp, out, t: [str(exe), "-m", str(weights), "-i", str(inp), "-o", str(out), "-t", str(t)],
        "decode_re": r"eval time =\s*([\d.]+) ms",
    }


def pytorch_decoder(vocos=False):
    """PyTorch encodec, or with `vocos` PyTorch Vocos. Its size is that of the installed
    packages it imports (Vocos also needs encodec and torchaudio)."""
    mods = ["torch", "encodec"] + (["torchaudio", "vocos"] if vocos else [])
    probe = f"import os, {', '.join(mods)}; " + "; ".join(f"print(os.path.dirname({m}.__file__))" for m in mods)
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    if out.returncode != 0:
        return None
    size = sum(dir_size(Path(p)) for p in out.stdout.split())
    script = ROOT / "bench" / "python_decode.py"
    flags = ["--float"] + (["--vocos"] if vocos else [])
    return {
        "name": "PyTorch vocos" if vocos else "PyTorch encodec", "backend": "torch",
        "size": size, "ref": "vocos" if vocos else "encodec",
        # float output, like the other decoders, so its SNR isn't limited by 16-bit rounding
        "cmd": lambda inp, out, t: [sys.executable, str(script), str(inp), str(out), "-t", str(t), *flags],
        "decode_re": r"decode ([\d.]+) ms",
    }


def dir_size(path):
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file() and not f.is_symlink())


# ---------------------------------------------------------------------------
# measuring

def timed(cmd):
    """Runs cmd under /usr/bin/time; returns wall, user, sys (s), peak memory (bytes), stderr."""
    env = dict(os.environ, LC_ALL="C")
    darwin = platform.system() == "Darwin"
    tcmd = ["/usr/bin/time", "-l" if darwin else "-v"] + cmd
    p = subprocess.run(tcmd, capture_output=True, text=True, env=env)
    err = p.stdout + p.stderr  # encodec.cpp prints its timings to stdout
    if p.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed:\n{err[-2000:]}")
    if darwin:
        m = re.search(r"([\d.]+) real\s+([\d.]+) user\s+([\d.]+) sys", err)
        wall, user, sys_ = (float(v) for v in m.groups())
        peak = int(re.search(r"(\d+)\s+peak memory footprint", err).group(1))
    else:
        h, mnt, s = re.search(r"Elapsed \(wall clock\) time .*?: (?:(\d+):)?(\d+):([\d.]+)", err).groups()
        wall = int(h or 0) * 3600 + int(mnt) * 60 + float(s)
        user = float(re.search(r"User time \(seconds\): ([\d.]+)", err).group(1))
        sys_ = float(re.search(r"System time \(seconds\): ([\d.]+)", err).group(1))
        peak = int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", err).group(1)) * 1024
    return wall, user, sys_, peak, err


def read_wav(path):
    """Mono 16-bit PCM or 32-bit float WAV -> list of floats (numpy array if available)."""
    import numpy as np
    data = Path(path).read_bytes()
    pos, fmt, samples = 12, None, None
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            tag, ch, _, _, _, bits = struct.unpack("<HHIIHH", body[:16])
            if tag == 0xFFFE:  # WAVE_FORMAT_EXTENSIBLE: real tag in the subformat GUID
                tag = struct.unpack("<H", body[24:26])[0]
            fmt = (tag, ch, bits)
        elif cid == b"data":
            samples = body
        pos += 8 + size + (size & 1)
    tag, ch, bits = fmt
    if tag == 3 and bits == 32:
        x = np.frombuffer(samples, dtype="<f4")
    elif tag == 1 and bits == 16:
        x = np.frombuffer(samples, dtype="<i2") / 32768.0
    else:
        raise ValueError(f"{path}: unsupported WAV format {fmt}")
    return x.reshape(-1, ch)[:, 0].astype(np.float64)


def snr(ref, out):
    """Overall SNR and the worst 1 s SNR, in dB. The reference is clipped to [-1, 1] like
    the 16-bit output of the other decoders, so peaks beyond full scale (Vocos has some)
    don't count as errors."""
    import numpy as np
    n = min(len(ref), len(out))
    r = np.clip(ref[:n], -1.0, 1.0)
    e = r - out[:n]
    def db(a, b):
        return float(10 * np.log10(max(np.sum(a ** 2), 1e-20) / max(np.sum(b ** 2), 1e-20)))
    per_s = [db(r[i:i + SAMPLE_RATE], e[i:i + SAMPLE_RATE]) for i in range(0, n - SAMPLE_RATE + 1, SAMPLE_RATE)]
    return db(r, e), min(per_s) if per_s else db(r, e)


def ecdc_seconds(path):
    head = Path(path).read_bytes()[:4096]
    n = struct.unpack(">I", head[5:9])[0]
    return json.loads(head[9:9 + n])["al"] / SAMPLE_RATE


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("-t", "--threads", default="4", help="comma-separated thread counts (default 4)")
    ap.add_argument("-r", "--runs", type=int, default=3)
    ap.add_argument("--backends", default=None, help="comma-separated C backends (default: all for this machine)")
    ap.add_argument("--baseline", default=None, help="also build wb_decode from this git revision")
    ap.add_argument("--vocos", action="store_true", help="also benchmark the Vocos decoder and PyTorch Vocos")
    ap.add_argument("--no-encodec-cpp", action="store_true")
    ap.add_argument("--no-pytorch", action="store_true")
    ap.add_argument("--no-accuracy", action="store_true",
                    help="skip the PyTorch reference decodes (they need GBs of memory for long inputs)")
    ap.add_argument("--note", default="", help="free text stored with the results")
    args = ap.parse_args()
    threads = [int(t) for t in args.threads.split(",")]
    WORK.mkdir(parents=True, exist_ok=True)

    if not WEIGHTS.exists():
        run([sys.executable, str(ROOT / "scripts" / "convert.py"), "--use-f16", "--out", str(WEIGHTS)])
    if args.vocos and not VOCOS_WEIGHTS.exists():
        run([sys.executable, str(ROOT / "scripts" / "convert_vocos.py"), "--use-f16", "--out", str(VOCOS_WEIGHTS)])

    print("building ...", file=sys.stderr)
    backends = args.backends.split(",") if args.backends else default_backends()
    decoders = build_zencodec(backends, args.vocos)
    if args.baseline:  # each baseline build goes right after the current build of its backend
        for base in build_baseline(args.baseline, backends):
            after = next((i for i, d in enumerate(decoders) if base["backend"].startswith(d["backend"] + "@")),
                         len(decoders) - 1)
            decoders.insert(after + 1, base)
    if not args.no_encodec_cpp:
        decoders.append(build_encodec_cpp())
    kinds = ["encodec"] + (["vocos"] if args.vocos else [])
    for kind in [] if args.no_pytorch else kinds:
        py = pytorch_decoder(kind == "vocos")
        if py:
            decoders.append(py)

    # accuracy references: the PyTorch decoders, in float
    refs = {}  # (kind, input) -> samples
    for kind in [] if args.no_accuracy else kinds:
        if not pytorch_decoder(kind == "vocos"):
            print(f"note: pip install {kind} torch numpy to measure accuracy against it", file=sys.stderr)
            continue
        for inp in args.inputs:
            out = WORK / f"ref_{kind}_{inp.stem}.wav"
            run([sys.executable, str(ROOT / "bench" / "python_decode.py"), str(inp), str(out), "--float"]
                + (["--vocos"] if kind == "vocos" else []), capture_output=True)
            refs[(kind, inp)] = read_wav(out)

    info = system_info()
    info["load_start"] = load_average()
    info["note"] = args.note
    samples = {}  # (decoder, input, threads) -> list of measurements
    for r in range(args.runs):
        for inp in args.inputs:
            for t in threads:
                for d in decoders:
                    out = WORK / "out.wav"
                    wall, user, sys_, peak, err = timed(d["cmd"](inp, out, t))
                    m = re.search(d["decode_re"], err)
                    meas = {"wall": wall, "user": user, "sys": sys_, "peak": peak,
                            "decode": float(m.group(1)) / 1e3 if m else None}
                    if (d["ref"], inp) in refs and d["backend"] != "torch":  # PyTorch is the reference
                        meas["snr"], meas["snr_min"] = snr(refs[(d["ref"], inp)], read_wav(out))
                    samples.setdefault((d["name"], inp, t), []).append(meas)
                    print(f"run {r + 1}/{args.runs}  {d['name']:24s} {inp.name} -t {t}: "
                          f"wall {wall:.2f} s, peak {peak / 1e6:.0f} MB", file=sys.stderr)
    info["load_end"] = load_average()

    # aggregate
    rows = []
    for d in decoders:
        for inp in args.inputs:
            audio_s = ecdc_seconds(inp)
            for t in threads:
                ms = samples[(d["name"], inp, t)]
                def med(k):
                    v = [m[k] for m in ms if m.get(k) is not None]
                    return statistics.median(v) if v else None
                walls = [m["wall"] for m in ms]
                dec = med("decode")
                rows.append({
                    "date": info["date"], "commit": info["commit"], "dirty": info["dirty"],
                    "machine": info["machine"], "os": info["os"], "load_start": info["load_start"],
                    "load_end": info["load_end"], "decoder": d["name"], "backend": d["backend"],
                    "input": inp.name, "audio_s": round(audio_s, 2), "threads": t, "runs": len(ms),
                    "wall_s": round(med("wall"), 3), "wall_min_s": round(min(walls), 3),
                    "wall_max_s": round(max(walls), 3),
                    "decode_s": round(dec, 3) if dec is not None else "",
                    "realtime_x": round(audio_s / dec, 1) if dec else "",
                    "user_s": round(med("user"), 3), "sys_s": round(med("sys"), 3),
                    "peak_mb": round(med("peak") / 1e6, 1), "size_bytes": d["size"],
                    "snr_db": round(med("snr"), 1) if med("snr") is not None else "",
                    "snr_min_1s_db": round(med("snr_min"), 1) if med("snr_min") is not None else "",
                    "note": args.note,
                })

    table = markdown(rows)
    print(table)
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = info["date"][:16].replace(":", "")
    report = RESULTS / f"{stamp}_{info['commit']}{'-dirty' if info['dirty'] else ''}.md"
    report.write_text(
        f"# Benchmark {info['date']}\n\n"
        + "\n".join(f"- **{k}:** {v}" for k, v in info.items() if v not in ("", None)) + "\n"
        + f"- **inputs:** {', '.join(i.name for i in args.inputs)}\n- **runs:** {args.runs}\n\n"
        + table + "\n")
    history = RESULTS / "history.csv"
    new = not history.exists()
    with history.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {report.relative_to(ROOT)} and appended to {history.relative_to(ROOT)}", file=sys.stderr)


def fmt_size(n):
    return f"{n / 1e6:.1f} MB" if n >= 1e6 else f"{n / 1e3:.0f} KB"


def markdown(rows):
    head = ("| Decoder | Input | Threads | Wall (median, min–max) | Decode | × real time | CPU (user) "
            "| Peak memory | Size | SNR | Worst 1 s |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    lines = [head]
    for r in rows:
        dec = f"{r['decode_s']:.2f} s" if r["decode_s"] != "" else "—"
        rt = f"{r['realtime_x']}×" if r["realtime_x"] != "" else "—"
        is_ref = r["backend"] == "torch"
        snr_ = "reference" if is_ref else f"{r['snr_db']} dB" if r["snr_db"] != "" else "—"
        worst = "reference" if is_ref else f"{r['snr_min_1s_db']} dB" if r["snr_min_1s_db"] != "" else "—"
        lines.append(f"| {r['decoder']} | {r['input']} ({r['audio_s']:g} s) | {r['threads']} "
                     f"| {r['wall_s']:.2f} s ({r['wall_min_s']:.2f}–{r['wall_max_s']:.2f}) | {dec} | {rt} "
                     f"| {r['user_s']:.2f} s | {r['peak_mb']:.0f} MB | {fmt_size(r['size_bytes'])} "
                     f"| {snr_} | {worst} |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
