#!/usr/bin/env bash
# End-to-end reproduce: toolchain check -> build -> correctness -> benchmark.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${VENV:-~/qwen-serving/venv}/bin/python
cd "$HERE"

echo "########## build ##########"
./build.sh

echo "########## correctness ##########"
"$PY" test_correctness.py            # tuned-auto must be bit-identical
"$PY" test_correctness.py --use-best # the shipped per-shape table
"$PY" test_integration.py           # table dispatch, prefill, W4A8, clamping
"$PY" test_gemv.py                  # the CUDA-core GEMV (shipped disabled)

echo "########## benchmark: stock vs tuned shortlist ##########"
# Shortlist of tile configs that survived the full --grid sweep.  Fields are
# thread_k,thread_n,stages,smem_mode,blocks_per_sm.
CFG="128,128,4,0,1;128,128,4,1,2;128,128,3,1,2;64,128,4,1,1;64,128,4,1,2"
CFG="$CFG;64,128,4,1,3;64,128,6,1,2;128,64,4,1,2;128,64,3,1,2"
CFG="$CFG;256,64,3,1,1;256,64,4,1,1;64,256,4,1,1;64,256,4,1,2"

# --iters 1: one launch per sample, so a kernel usually lands inside a single
# GPU context-switch slice.  Do not raise this for the ~750us lm_head -- a long
# inner loop straddles context switches and measures whatever else is running.
"$PY" -u bench_marlin.py --grid --m 1,5,16 --iters 1 --rounds 600 \
    --top 20 --only-configs "$CFG"

echo
echo "For the exhaustive sweep instead of the shortlist, drop --only-configs."
echo "Run on an idle GPU when possible: nvidia-smi utilization should be ~0."
