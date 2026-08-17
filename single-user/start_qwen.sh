#!/bin/bash
# Qwen3.8-27B on a single RTX 3090 — SINGLE USER / LOW LATENCY mode.
#
# Same base config as batch mode, plus MTP speculative decoding: the checkpoint
# keeps Qwen's multi-token-prediction head, so the model drafts 3 tokens ahead
# and verifies them in one pass. Measured on realistic chat prompts: ~84 tok/s
# single-stream at the model's default sampling, ~91 tok/s greedy (vs 46 tok/s
# without speculation). Three things make 3 drafts pay off:
#  - the MTP module is requantized to int8 (quant_mtp.py): 850 -> 430 MB read
#    per draft
#  - the drafter scores a 40k-token draft head (build_draft_vocab.py + the
#    mtp-draft-vocab patch): 210 MB instead of the 1.3 GB lm_head per draft
#  - draft_sample_method=probabilistic: drafts are sampled from the draft
#    distribution instead of argmax, which lifts acceptance at temperature > 0
#    (+15% tok/s at the default temperature 1.0 / top-p 0.95)
# Speculative decoding is exact: none of this changes what gets sampled.
#
# Why 3 and not 4: k=4 measures ~7% faster at a single stream but the engine
# dies with an illegal memory access as soon as one request finishes while
# another is mid-generation (vLLM 0.27.1 + Qwen3.5 GDN spec decode; club-3090
# reports the same "n=4 eventually dies, n=3 stable" pattern). k=3 survived
# every concurrency soak we threw at it.
#
# max-num-seqs is 8 here: fewer state slots to reserve (each request holds
# k+1 = 4 recurrent-state slots), and past a handful of concurrent users you
# should be running batch mode anyway. Int8 activations are pointless at
# batch size 1 (memory-bound), so this mode stays W4A16.

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$DIR")"
cd "$REPO"

MODEL=${MODEL:-$REPO/models/Qwen3.8-27B-W4A16-AutoRound}
PORT=${PORT:-18020}
MAX_LEN=${MAX_LEN:-150000}
MAX_SEQS=${MAX_SEQS:-8}
# 0.93 here, NOT batch mode's 0.972: the DeltaNet workspace in the MTP decode
# path allocates beyond the startup memory profile (main README, gotcha 4).
GPU_UTIL=${GPU_UTIL:-0.93}
API_SERVERS=${API_SERVERS:-1}
DRAFT_TOKENS=${DRAFT_TOKENS:-3}

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
  --mamba-ssm-cache-dtype float16 \
  --async-scheduling \
  --max-num-batched-tokens 2048 \
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$DRAFT_TOKENS,\"draft_sample_method\":\"probabilistic\"}" \
  --compilation-config "{\"max_cudagraph_capture_size\":32,\"custom_ops\":[\"+rms_norm\",\"+silu_and_mul\"]}" \
  --reasoning-parser qwen3 \
  ${EXTRA_ARGS}
