#!/bin/bash
# Qwen3.8-27B on a single RTX 3090 — BATCH / THROUGHPUT mode.
# Measured: 416 tok/s aggregate @ 64 concurrent (256 in / 256 out), 150k context.
#
# Hard-won settings, don't change casually:
#  - --language-model-only skips the vision tower entirely (~2.7 GB saved)
#  - expandable_segments is required: the DeltaNet prefill kernels allocate
#    transient workspace and fragment the allocator, OOMs at util >= 0.978 without it
#  - gpu-memory-utilization 0.972 is the sweet spot on a headless box
#    (X/display holds ~220 MB; 0.98 fails the startup free-memory check)
#  - max-num-batched-tokens 2048 beats 8192 here: bigger chunks inflate the
#    profiled activation peak, which shrinks the KV/state page pool
#  - kv-cache-dtype fp8 roughly doubles the usable context/pool

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$DIR")"
cd "$REPO"

MODEL=${MODEL:-$REPO/models/Qwen3.8-27B-W4A16-AutoRound}
PORT=${PORT:-18020}
MAX_LEN=${MAX_LEN:-150000}
MAX_SEQS=${MAX_SEQS:-64}
GPU_UTIL=${GPU_UTIL:-0.972}
API_SERVERS=${API_SERVERS:-1}

export PATH="$REPO/venv/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# flashinfer's sampling.cu does not build with older system nvcc (12.0);
# the attention kernels JIT fine. Remove this if you have a recent CUDA toolkit.
export VLLM_USE_FLASHINFER_SAMPLER=0

# API key: put it in api_key.txt in the repo root, or export VLLM_API_KEY.
if [ -z "$VLLM_API_KEY" ] && [ -f "$REPO/api_key.txt" ]; then
  export VLLM_API_KEY="$(cat "$REPO/api_key.txt")"
fi

exec venv/bin/vllm serve "$MODEL" \
  --served-model-name qwen3.8-27b \
  --host 0.0.0.0 --port $PORT \
  --gpu-memory-utilization $GPU_UTIL \
  --max-model-len $MAX_LEN \
  --max-num-seqs $MAX_SEQS \
  --api-server-count $API_SERVERS \
  --language-model-only \
  --kv-cache-dtype fp8 \
  --async-scheduling \
  --max-num-batched-tokens 2048 \
  --compilation-config "{\"max_cudagraph_capture_size\":64,\"custom_ops\":[\"+rms_norm\",\"+silu_and_mul\"]}" \
  ${EXTRA_ARGS}
