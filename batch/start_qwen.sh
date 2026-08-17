#!/bin/bash
# Qwen3.8-27B on a single RTX 3090 — BATCH / THROUGHPUT mode.
# Measured: ~1,000 tok/s aggregate @ 64 concurrent (128 in / 512 out), 150k context.
#
# Hard-won settings, don't change casually:
#  - --mamba-ssm-cache-dtype float16: the Gated DeltaNet recurrent state is
#    fp32 by default (Qwen's config says so) and costs ~150 MB per resident
#    request; fp16 halves that AND halves the state traffic per decode step.
#    That is what lets all 64 requests actually run at once (fp32: only 37).
#    Perplexity is unchanged (8.045 vs 8.046 on our en/da/code check).
#  - VLLM_MARLIN_INPUT_DTYPE=int8: int8 tensor-core (W4A8) Marlin path for the
#    MLP GEMMs, weights stay int4. Roughly +35% aggregate on top of the state
#    change for +2.2% perplexity. Needs both marlin patches from patches/.
#    INT8_LAYERS=gate_up is the gentler variant (+0.9% PPL, ~+15%);
#    INT8_ACT= (empty) turns it off entirely (pure W4A16, quality-neutral).
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
# int8 activations: "int8" (default) or empty for W4A16; layers: regex on the
# layer name, "mlp" (default: gate_up_proj + down_proj) or "gate_up"
INT8_ACT=${INT8_ACT-int8}
INT8_LAYERS=${INT8_LAYERS-mlp}

export PATH="$REPO/venv/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# flashinfer's sampling.cu does not build with older system nvcc (12.0);
# the attention kernels JIT fine. Remove this if you have a recent CUDA toolkit.
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_MARLIN_INPUT_DTYPE=$INT8_ACT
export VLLM_MARLIN_INT8_INCLUDE_RE=$INT8_LAYERS

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
  --mamba-ssm-cache-dtype float16 \
  --async-scheduling \
  --max-num-batched-tokens 2048 \
  --compilation-config "{\"max_cudagraph_capture_size\":64,\"custom_ops\":[\"+rms_norm\",\"+silu_and_mul\"]}" \
  --reasoning-parser qwen3 \
  ${EXTRA_ARGS}
