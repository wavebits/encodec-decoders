#!/usr/bin/env bash
# Compare wb_decode (C and Zig) with encodec.cpp and the PyTorch `encodec` package:
# wall time, CPU time, peak memory, binary size.
#
# Usage:
#   bench/compare_encodec_cpp.sh <in.ecdc> [n_threads] [n_runs]
#
# Needs: zig, cmake, a C++ compiler, git, python with torch + numpy (for encodec.cpp's
# weight converter). Make the .ecdc with `encodec -b 6 input.wav in.ecdc` (no --lm).
# encodec.cpp is cloned into bench/work/ at a pinned commit and patched with
# bench/encodec_cpp.patch so it can decode .ecdc files from the Python encodec CLI
# (see the README).
set -euo pipefail

ECDC=${1:?usage: $0 <in.ecdc> [n_threads] [n_runs]}
THREADS=${2:-4}
RUNS=${3:-3}

ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORK=$ROOT/bench/work
ENCODEC_CPP_COMMIT=83deb38a757a12a58f993e34a44d32b3fd5d3893
# encodec.cpp builds one graph for the whole file; raise its node cap so longer inputs fit
ENCODEC_MAX_NODES=${ENCODEC_MAX_NODES:-400000}

mkdir -p "$WORK"

# --- zencodec
(cd "$ROOT" && zig build -Dstrip=true)
ZC=$ROOT/zig-out/bin/wb_decode-c
ZZ=$ROOT/zig-out/bin/wb_decode-zig
ZW=$ROOT/weights/encodec_24khz_decoder_f16.bin
[ -f "$ZW" ] || python "$ROOT/scripts/convert.py" --use-f16 --out "$ZW"

# --- encodec.cpp
E=$WORK/encodec.cpp
if [ ! -d "$E" ]; then
    git clone --recursive https://github.com/PABannier/encodec.cpp.git "$E"
    git -C "$E" checkout -q "$ENCODEC_CPP_COMMIT"
    git -C "$E" submodule update -q --init --recursive
    git -C "$E" apply "$ROOT/bench/encodec_cpp.patch"
fi
EW=$WORK/ggml_weights/ggml-model.bin
if [ ! -f "$EW" ]; then
    CKPT_DIR=${TORCH_HOME:-$HOME/.cache/torch}/hub/checkpoints
    if [ ! -f "$CKPT_DIR/encodec_24khz-d7cc33bc.th" ]; then
        mkdir -p "$CKPT_DIR"
        curl -L -o "$CKPT_DIR/encodec_24khz-d7cc33bc.th" \
            https://dl.fbaipublicfiles.com/encodec/v0/encodec_24khz-d7cc33bc.th
    fi
    (cd "$E" && python convert.py --dir-model "$CKPT_DIR" --out-dir "$WORK/ggml_weights" --use-f16)
fi
cmake -S "$E" -B "$E/build-cpu" -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=OFF \
    -DCMAKE_CXX_FLAGS="-DENCODEC_MAX_NODES=$ENCODEC_MAX_NODES" > /dev/null 2>&1
cmake --build "$E/build-cpu" -j --target decompress > /dev/null
EC=$E/build-cpu/bin/decompress

# --- measure
if [ "$(uname)" = Darwin ]; then TIME=(/usr/bin/time -l); else TIME=(/usr/bin/time -v); fi
LOG=$WORK/time.log

measure() { # name, command...
    local name=$1; shift
    "${TIME[@]}" "$@" > "$LOG" 2>&1 || { echo "$name failed:"; tail -5 "$LOG"; return; }
    if [ "$(uname)" = Darwin ]; then
        awk -v n="$name" '/ real /{w=$1; u=$3} /peak memory footprint/{m=$1}
            END{printf "%-14s wall %7.2f s  user %7.2f s  peak %7.1f MB\n", n, w, u, m/1e6}' "$LOG"
    else
        awk -v n="$name" -F': ' '/Elapsed \(wall/{split($2,a,":"); w=a[1]*60+a[2]} /User time/{u=$2}
            /Maximum resident/{m=$2} END{printf "%-14s wall %7.2f s  user %7.2f s  peak %7.1f MB\n", n, w, u, m/1e3}' "$LOG"
    fi
}

size_of() { wc -c < "$1" | tr -d ' '; }
echo "binary size: wb_decode-c $(size_of "$ZC") B, wb_decode-zig $(size_of "$ZZ") B, encodec.cpp $(size_of "$EC") B"
PY_ENCODEC=$(python -c "import encodec, torch; print(1)" 2>/dev/null || true)
[ -n "$PY_ENCODEC" ] || echo "(pip install encodec torch to include the PyTorch decoder)"
echo "input: $ECDC, $THREADS threads, $RUNS runs (interleaved)"
for _ in $(seq "$RUNS"); do
    measure wb_decode-c  "$ZC" decode "$ZW" "$ECDC" "$WORK/out_zc.wav" -t "$THREADS"
    measure wb_decode-zig "$ZZ" decode "$ZW" "$ECDC" "$WORK/out_zz.wav" -t "$THREADS"
    measure encodec.cpp  "$EC" -m "$EW" -i "$ECDC" -o "$WORK/out_ec.wav" -t "$THREADS"
    if [ -n "$PY_ENCODEC" ]; then
        measure pytorch  python "$ROOT/bench/python_decode.py" "$ECDC" "$WORK/out_py.wav" -t "$THREADS"
    fi
done
