#!/bin/bash
# Qwen3.8-27B on a single RTX 3090 — SINGLE USER / LOW LATENCY mode.
#
# Same base config as batch mode, plus MTP speculative decoding: the checkpoint
# keeps Qwen's multi-token-prediction head, so the model drafts 2 tokens ahead
# and verifies them in one pass. Measured: 63 tok/s single-stream (14.9 ms per
# token) vs 45 tok/s without speculation, at 57% draft acceptance. The price is
# throughput under load, which is why this is a separate mode and not the
# default: from ~8 concurrent users, batch mode wins.
#
# max-num-seqs is 8 here: fewer state slots to reserve, and past a handful of
# concurrent users you should be running batch mode anyway. DRAFT_TOKENS=3
# measured slightly worse on general text (46% acceptance); might still pay
# off on repetitive/code-heavy prompts.

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$DIR")"
cd "$REPO"

MODEL=${MODEL:-$REPO/models/Qwen3.8-27B-W4A16-AutoRound}
PORT=${PORT:-18020}
MAX_LEN=${MAX_LEN:-150000}
MAX_SEQS=${MAX_SEQS:-8}
# 0.90 here, NOT batch mode's 0.972: the DeltaNet workspace in the MTP decode
# path allocates beyond the startup memory profile, and long generations OOM
# the engine at higher settings. With 8 slots the bigger pool buys nothing.
GPU_UTIL=${GPU_UTIL:-0.90}
API_SERVERS=${API_SERVERS:-1}
DRAFT_TOKENS=${DRAFT_TOKENS:-2}

export PATH="$REPO/venv/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_FLASHINFER_SAMPLER=0

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
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$DRAFT_TOKENS}" \
  --compilation-config "{\"max_cudagraph_capture_size\":16,\"custom_ops\":[\"+rms_norm\",\"+silu_and_mul\"]}" \
  ${EXTRA_ARGS}
