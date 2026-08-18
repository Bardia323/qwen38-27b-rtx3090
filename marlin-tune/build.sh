#!/usr/bin/env bash
# Build the standalone tunable Marlin extension.
#
#   ./build.sh                # full tuning grid (~5 min on 12 cores)
#   ./build.sh --stock-only   # only the 4 stock tiles at stages=4 (fast)
#
# Nothing is installed into the vLLM venv; the .so lands next to this script.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV=${VENV:-~/qwen-serving/venv}
PY="$VENV/bin/python"

# --- CUDA 13 toolchain -------------------------------------------------------
# System nvcc is 12.0, too old for torch 2.13+cu130 extensions.
#
# Create the toolchain once (nothing is installed into the vLLM venv):
#   python3 -m venv ~/cuda13-env
#   ~/cuda13-env/bin/pip install \
#       nvidia-cuda-nvcc==13.0.88     nvidia-cuda-runtime==13.0.96 \
#       nvidia-cuda-cccl==13.0.85     nvidia-cuda-crt==13.0.88 \
#       nvidia-cuda-nvrtc==13.0.88    nvidia-cuda-profiler-api==13.0.85 \
#       nvidia-nvvm==13.0.88
#
# Two gotchas:
#  * The `nvidia-*-cu13` package names are deprecated stubs whose sdists fail
#    to build on purpose.  The unsuffixed names are the CUDA 13 wheels.
#  * Pin 13.0.x.  torch 2.13+cu130 reports torch.version.cuda == 13.0, and the
#    CUDA 13.3 nvcc rejects the 13.0 CCCL headers with
#      "CUDA compiler and CUDA toolkit headers are incompatible".
#    (The torch wheel itself ships a *mismatched* nvidia/cu13 -- nvcc 13.3 on
#     top of 13.0 headers -- so it cannot be used as CUDA_HOME as-is.)
#  * nvidia-nvvm MUST be pinned too.  It is a transitive dep and pip happily
#    resolves it to 13.3, whose cicc emits PTX ISA 9.3 that the 13.0 ptxas
#    rejects: "Unsupported .version 9.3; current version is 9.0".
#
# torch.utils.cpp_extension wants $CUDA_HOME/{bin,include,lib64}; the pip
# layout uses lib/, so build a symlink shim.
CUDA_SRC=${CUDA_SRC:-$HOME/cuda13-env/lib/python3.12/site-packages/nvidia/cu13}
CUDA_HOME="$HERE/cuda13-home"
mkdir -p "$CUDA_HOME"
ln -sfn "$CUDA_SRC/bin" "$CUDA_HOME/bin"
ln -sfn "$CUDA_SRC/include" "$CUDA_HOME/include"
ln -sfn "$CUDA_SRC/lib" "$CUDA_HOME/lib64"
ln -sfn "$CUDA_SRC/lib" "$CUDA_HOME/lib"
ln -sfn "$CUDA_SRC/nvvm" "$CUDA_HOME/nvvm"
[ -d "$CUDA_SRC/cccl" ] && ln -sfn "$CUDA_SRC/cccl" "$CUDA_HOME/cccl"

export CUDA_HOME
# venv/bin so cpp_extension finds the `ninja` binary (parallel build).
export PATH="$CUDA_HOME/bin:$VENV/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.6"
# Be polite: the box also runs a vLLM server.
export MAX_JOBS=${MAX_JOBS:-8}

echo "== nvcc: $(nvcc --version | tail -1)"
echo "== torch cuda: $($PY -c 'import torch;print(torch.version.cuda)')"

cd "$HERE"
"$PY" gen_kernels.py \
    --out-dir csrc/libtorch_stable/quantization/marlin "$@"
"$PY" setup.py build_ext --inplace 2>&1 | tail -40
echo "== built: $(ls -la "$HERE"/marlin_tune_ext*.so)"
