#!/bin/bash
# Qwen3.8-27B on a single RTX 3090 — SINGLE USER / LOW LATENCY mode.
#
# Same base config as batch mode, plus MTP speculative decoding: the checkpoint
# keeps Qwen's multi-token-prediction head, so the model drafts 3-4 tokens ahead
# and verifies them in one pass. Measured on realistic chat prompts with the
# `-fast` model variant (see "Fast variant" below): ~114 tok/s at the model's
# default sampling, ~124 tok/s greedy (vs 46 tok/s without speculation).
# What makes 4 drafts pay off, in order of importance:
#  - the drafter scores a 40k-token draft head (build_draft_vocab.py) — and the
#    id list matters: a vocabulary counted over the model's OWN outputs covers
#    97.5% of what it generates (96% on code); the earlier web-text list only 92%
#    (83% on code), and every miss is a forced rejection (108 vs 98 tok/s greedy)
#  - the MTP module and lm_head requantized to int4 with GPTQ calibrated on the
#    model's hidden states (drafter/): 850 -> 215 MB per draft, 1.27 -> 0.65 GB
#    lm_head per verify, +0.6% perplexity, acceptance unchanged
#  - patches/spec-decode-attn.patch: split-KV attention for the 5-query verify
#    step (FA2 leaves 58 of 82 SMs idle there); patches/sampler-...: sort-free
#    top-k, multi-block softmax, drafts truncated to the target's top-k/top-p
#  - draft_sample_method=probabilistic: drafts are sampled, not argmax'ed, which
#    lifts acceptance at temperature > 0
# Speculative decoding is exact: none of this changes what gets sampled.
#
# CTX=fast (default here): FlashAttention + bf16 KV, 4 drafts, 64k context.
# CTX=long: fp8 KV via FlashInfer, 150k context, 3 drafts (k=4 crashes on
#   FlashInfer as soon as one request finishes while another is mid-generation,
#   vLLM 0.27.1); the split-KV attention patch is bf16-KV only, so ~90/98 tok/s.
# CTX=huge: KVarN 4/2-bit KV cache (kvarn/), 200k context with MTP.
#
# Fast variant: MODEL defaults to models/Qwen3.8-27B-W4A16-AutoRound-fast when it
# exists (int4-GPTQ lm_head + MTP, own-output draft vocab; drafter/README.md), else
# the base dir (int8 lm_head/MTP: ~108/107 tok/s with the shipped draft vocab).
#
# max-num-seqs is 8 here: fewer state slots to reserve (each request holds
# k+1 recurrent-state slots), and past a handful of concurrent users you
# should be running batch mode anyway. Int8 activations are pointless at
# batch size 1 (memory-bound), so this mode stays W4A16.

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$DIR")"
cd "$REPO"

if [ -z "$MODEL" ] && [ -d "$HOME/models/Qwen3.8-27B-W4A16-AutoRound-fast" ]; then
  MODEL=$HOME/models/Qwen3.8-27B-W4A16-AutoRound-fast
fi
MODEL=${MODEL:-$HOME/models/Qwen3.8-27B-W4A16-AutoRound}
PORT=${PORT:-18020}
# 0.93 here, NOT batch mode's 0.972: the DeltaNet workspace in the MTP decode
# path allocates beyond the startup memory profile (main README, gotcha 4).
GPU_UTIL=${GPU_UTIL:-0.93}
API_SERVERS=${API_SERVERS:-1}
# CTX=long (default): fp8 KV via FlashInfer, 150k context, 3 drafts.
# CTX=fast: bf16 KV via FlashAttention, ~64k context, 4 drafts (~+7%).
# CTX=huge: KVarN 4/2-bit KV cache (kvarn/ in this repo, run kvarn/install.sh
#           once), 200k context with MTP, ~5% slower (see README "262k context").
CTX=${CTX:-long}
# SPEC=mtp (default): Qwen's own MTP head, k drafts chained (the numbers above).
# SPEC=dflash2: the DFlash2 block drafter (incoai/Qwen3.8-27B-DFlash2, requantized
#   to W4A16 by this repo: prepare/fetch_dflash2.py), 7 drafts in ONE non-autoregressive
#   pass + a path selector; runs on vLLM's V2 model runner
#   (patches/dflash2-backport.patch). CTX=fast (bf16, 64k), CTX=long (int8, 128k)
#   or, with kvarn/install.sh, CTX=huge (KVarN 4/2-bit, 240k + prefix caching).
SPEC=${SPEC:-mtp}
# SPEC_ATTN=1: split-KV Triton attention for the multi-query verify step
# (patches/spec-decode-attn.patch); bf16 KV only, so CTX=fast only.
if [ "$CTX" = "fast" ]; then
  MAX_LEN=${MAX_LEN:-65536}
  DRAFT_TOKENS=${DRAFT_TOKENS:-4}
  ATTN_ARGS="--attention-backend FLASH_ATTN --kv-cache-dtype bfloat16"
  export VLLM_SPEC_DECODE_ATTN=${SPEC_ATTN:-1}
elif [ "$CTX" = "huge" ]; then
  MAX_LEN=${MAX_LEN:-200000}
  DRAFT_TOKENS=${DRAFT_TOKENS:-3}
  ATTN_ARGS="--kv-cache-dtype kvarn_k4v2_g128 --block-size 128"
  export KVARN_POOL_MEM_FRAC=${KVARN_POOL_MEM_FRAC:-0.15}
else
  MAX_LEN=${MAX_LEN:-150000}
  DRAFT_TOKENS=${DRAFT_TOKENS:-3}
  ATTN_ARGS="--kv-cache-dtype fp8"
fi

if [ "$SPEC" = "dflash2" ] && [ "$CTX" = "long" ]; then
  # int8 per-token-head KV on the Triton backend: the same 5.2 GiB pool holds 136,429
  # tokens instead of 69,758 (patches/hybrid-sw-block-promote.patch +
  # patches/spec-decode-int8-kv.patch). Costs prefill: 251 s for a 112k document against
  # FLASH_ATTN's ~112 s, so pair it with PREFIX_CACHE=1 for RAG / coding front-ends.
  ATTN_ARGS="--attention-backend TRITON_ATTN --kv-cache-dtype int8_per_token_head"
  export VLLM_SPEC_DECODE_ATTN=${SPEC_ATTN:-1}
elif [ "$SPEC" = "dflash2" ] && [ "$CTX" = "huge" ]; then
  # KVarN 4/2-bit KV on the V2 runner (kvarn/install.sh, kvarn-v2-runner.patch stage).
  # The split-KV verify attention is bf16-KV only; KVarN brings its own dequant path.
  export VLLM_SPEC_DECODE_ATTN=0
elif [ "$SPEC" = "dflash2" ] && [ "$CTX" != "fast" ]; then
  echo "SPEC=dflash2 supports CTX=fast (bf16, 64k), CTX=long (int8, 128k) and CTX=huge (KVarN, 240k; kvarn/install.sh); CTX=$CTX keeps SPEC=mtp" >&2
  SPEC=mtp
fi

if [ "$SPEC" = "dflash2" ]; then
  # Same $HOME-first layout as MODEL above.
  if [ -z "$DRAFT" ]; then
    for d in Qwen3.8-27B-DFlash2-W4A16 Qwen3.8-27B-DFlash2; do
      for base in "$HOME/models" "$REPO/models"; do
        [ -f "$base/$d/model.safetensors" ] && DRAFT="$base/$d" && break 2
      done
    done
  fi
  [ -n "$DRAFT" ] || { echo "SPEC=dflash2 needs the drafter: venv/bin/python prepare/fetch_dflash2.py $HOME/models/Qwen3.8-27B-DFlash2-W4A16" >&2; exit 1; }
  # Draft from the request's own context when it repeats itself
  # (patches/dflash2-lookup-drafting.patch).
  export VLLM_DFLASH2_LOOKUP=${LOOKUP:-1}
  # SPEC=dflash2 runs on the V2 model runner, whose RequestState allocates UVA buffers,
  # and is_uva_available() is just is_pin_memory_available() -- which vLLM turns OFF on
  # WSL2 by default. Without this the engine dies at startup with "UVA is not available".
  # Pinned memory does work here; it is the blanket WSL2 guard, not the hardware. Scoped
  # to this branch so SPEC=mtp keeps vLLM's WSL2 default untouched.
  export VLLM_WSL2_ENABLE_PIN_MEMORY=${VLLM_WSL2_ENABLE_PIN_MEMORY:-1}
  # FlashInfer's top-k JIT does not build against the system nvcc/g++ 11 here (the same
  # reason VLLM_USE_FLASHINFER_SAMPLER=0 is set below); vLLM falls back to torch.topk
  # anyway, so skip the failing build instead of paying for it at every start.
  export VLLM_DFLASH2_TORCH_TOPK=${VLLM_DFLASH2_TORCH_TOPK:-1}
  # DFLASH_TOKENS is the *verify* block; the checkpoint always proposes the 7 it was
  # trained for, and any position past that is filled from the request's own context.
  # DFLASH_TOKENS=15 is "reproduction mode": +50% where the model quotes its context,
  # at 4 request slots instead of 8. The default stays 7.
  DRAFT_TOKENS=${DFLASH_TOKENS:-7}
  SPEC_CFG="{\"method\":\"dflash\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":$DRAFT_TOKENS}"
  # The split-KV verify attention sizes its partial buffers once for the longest query
  # block it will see; a captured CUDA graph holds their addresses, so do not grow them.
  export VLLM_SPEC_DECODE_ATTN_QMAX=${VLLM_SPEC_DECODE_ATTN_QMAX:-$((DRAFT_TOKENS + 1))}
  # The ADAPTIVE verify length corrupts a prefix-cache hit under KVarN. When the block
  # alternates 8<->16 step to step and the request resumed from a cache hit, turn 2 over
  # the same document tracks the source for ~38 characters and then diverges -- turn 1 is
  # correct. Deterministic, at five of six prompt residues tested.
  #
  # It is the length CHANGING, not the lookup content, and not the KVarN kernels. Fresh
  # server per row, three requests each (one to arm the cache, two measured), sha-compared:
  #   baseline, adaptive 8<->16          self-hit 38/793 WRONG   12.71 tok/step
  #   VLLM_DFLASH2_LOOKUP_ADAPTIVE=0     self-hit 794/794 clean  14.71 tok/step
  #   PREFIX_CACHE=0, adaptive on        self-hit 794/794 clean  14.33
  #   KVARN_FUSED_VERIFY=0, adaptive on  self-hit 38/793 WRONG   12.71
  # and the constant-length setting is clean at every residue that broke (16/64/96/124),
  # cold and warm, byte-identical.
  #
  # Note what the earlier controls actually varied: DFLASH_TOKENS=7, LOOKUP=0 and SPEC=mtp
  # all make the block a CONSTANT length, so "clean" there was never evidence about the
  # block being short or the lookup being off. And a wrong draft cannot corrupt greedy
  # output at all -- rejection_sampler.py emits target_argmax whether it accepts or not --
  # so the draft content was never a candidate. The damage is on the target's forward.
  #
  # Pinning the length is not a sacrifice here: 14.71 tok/step is the FASTEST number in the
  # series, above the adaptive path's own 14.29 cold. Root cause still open; this is a
  # correct and fast setting, not a workaround with a cost.
  if [ -z "${LOOKUP_ADAPTIVE:-}" ] && [ "$VLLM_DFLASH2_LOOKUP" = "1" ] && [ "$DRAFT_TOKENS" -gt 7 ] \
     && [ "$CTX" = "huge" ] && [ "${PREFIX_CACHE:-0}" = "1" ]; then
    echo "DFLASH_TOKENS>7 + CTX=huge + PREFIX_CACHE=1: pinning the verify block to" >&2
    echo "$((DRAFT_TOKENS + 1)) tokens (VLLM_DFLASH2_LOOKUP_ADAPTIVE=0). The adaptive length" >&2
    echo "corrupts the second turn over a shared prefix on the KVarN cache; pinned is both" >&2
    echo "correct and faster. LOOKUP_ADAPTIVE=1 asks for the adaptive path anyway." >&2
    export VLLM_DFLASH2_LOOKUP_ADAPTIVE=0
  fi
  if [ "$VLLM_DFLASH2_LOOKUP" = "1" ] && [ "$DRAFT_TOKENS" -gt 7 ]; then
    # Adaptive block length needs the synchronous scheduler; costs under 1% at batch 1.
    ASYNC_SCHED=${ASYNC_SCHED:-0}
  fi
  # The pool is pinned by bytes, not by gpu-memory-utilization: the V2 runner's profiled
  # activation peak swings ~1 GiB between starts. KV_MEM= (empty) falls back to GPU_UTIL.
  if [ "$CTX" = "huge" ]; then
    MAX_SEQS=${MAX_SEQS:-2}
    MAX_LEN=${DFLASH_MAX_LEN:-245760}
    KV_MEM=${KV_MEM-5261334938}
    export VLLM_V2_CUDAGRAPH_MEM_MIB=${VLLM_V2_CUDAGRAPH_MEM_MIB:-1400}
  elif [ "$CTX" = "long" ]; then
    MAX_SEQS=${MAX_SEQS:-4}
    MAX_LEN=${DFLASH_MAX_LEN:-131072}
    KV_MEM=${KV_MEM-5583457484}
    if [ "$DRAFT_TOKENS" -gt 7 ]; then
      export VLLM_V2_CUDAGRAPH_MEM_MIB=${VLLM_V2_CUDAGRAPH_MEM_MIB:-1900}
    else
      export VLLM_V2_CUDAGRAPH_MEM_MIB=${VLLM_V2_CUDAGRAPH_MEM_MIB:-1400}
    fi
  elif [ "$DRAFT_TOKENS" -gt 7 ]; then
    MAX_SEQS=${MAX_SEQS:-4}
    MAX_LEN=${DFLASH_MAX_LEN:-57344}
    KV_MEM=${KV_MEM-5583457484}
    export VLLM_V2_CUDAGRAPH_MEM_MIB=${VLLM_V2_CUDAGRAPH_MEM_MIB:-1900}
  else
    MAX_LEN=${DFLASH_MAX_LEN:-65536}
    KV_MEM=${KV_MEM-5583457484}
    export VLLM_V2_CUDAGRAPH_MEM_MIB=${VLLM_V2_CUDAGRAPH_MEM_MIB:-1400}
  fi
  MAX_SEQS=${MAX_SEQS:-8}
  # The V2 runner captures decode graphs in multiples of k+1 tokens: cover MAX_SEQS requests.
  CG=${CG:-$((MAX_SEQS * (DRAFT_TOKENS + 1)))}
  [ -n "$KV_MEM" ] && EXTRA_ARGS="--kv-cache-memory=$KV_MEM ${EXTRA_ARGS}"
else
  MAX_SEQS=${MAX_SEQS:-8}
  SPEC_CFG="{\"method\":\"mtp\",\"num_speculative_tokens\":$DRAFT_TOKENS,\"draft_sample_method\":\"${DRAFT_SAMPLE:-probabilistic}\"}"
  CG=${CG:-32}
fi

# PREFIX_CACHE=1: reuse the KV of a shared prompt prefix across requests, and resume the
# recurrent (GDN) state from the last cached block boundary instead of re-running the
# prompt. Costs one extra state page per request (~16% of the KV pool). Opt-in, as
# upstream keeps it for hybrid models.
if [ "${PREFIX_CACHE:-0}" = "1" ]; then
  EXTRA_ARGS="--enable-prefix-caching --mamba-cache-mode align ${EXTRA_ARGS}"
  # KVarN runs --block-size 128; a prefix hash unit that is not a multiple of 128
  # corrupts the pool.
  [ "$CTX" = "huge" ] && EXTRA_ARGS="--prefix-match-unit 128 ${EXTRA_ARGS}"
  # CTX=huge captures PIECEWISE: a FULL capture corrupts one prompt length in every 128
  # (R = 117 + k) once a prefix-cache hit fires, for every speculator, mtp included.
  # Treat CUDAGRAPH_MODE=FULL_AND_PIECEWISE as unsafe (bench/bugb_sweep.py).
  [ "$CTX" = "huge" ] &&
    CG_MODE=",\"cudagraph_mode\":\"${CUDAGRAPH_MODE:-PIECEWISE}\""
fi

# --async-scheduling is already the default in 0.27.1; --no-async-scheduling turns it off.
# ASYNC_SCHED=0 (set above for a long DFlash2 verify block) is the only path on which
# vLLM lets the worker choose how many draft tokens to put up for verification.
ASYNC_ARGS=$([ "${ASYNC_SCHED:-1}" = 1 ] && echo --async-scheduling || echo --no-async-scheduling)

# VENV: the repo's own venv sits on the Windows-side 9p mount, where `import vllm` costs
# ~43 s per process -- and the WSL2 branch below runs the worker with spawn, so that is
# paid twice before vLLM logs anything. A byte-identical copy on ext4 imports far faster,
# so prefer it when it exists. Set VENV= explicitly to pin either one; delete the copy and
# this falls back to $REPO/venv on its own.
if [ -z "${VENV:-}" ] && [ -x "$HOME/qwen-venv/bin/vllm" ]; then
  VENV="$HOME/qwen-venv"
fi
VENV=${VENV:-$REPO/venv}
export PATH="$VENV/bin:$PATH"

if ! uname -r | grep -q -i "microsoft"; then
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
else
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
fi

export VLLM_USE_FLASHINFER_SAMPLER=0

# --language-model-only skips the vision tower (~2.7 GB). Qwen3.8-27B ships as a VL
# checkpoint (Qwen3_5ForConditionalGeneration) so this is right for the stock model and
# stays the default. Some third-party quants are exported text-only already
# (Qwen3_5ForCausalLM, e.g. the abliterated builds) and have no tower to skip; set
# LM_ONLY=0 for those.
LM_ONLY_ARG=$([ "${LM_ONLY:-1}" = 1 ] && echo --language-model-only)

if [ -z "$VLLM_API_KEY" ] && [ -f "$REPO/api_key.txt" ]; then
  export VLLM_API_KEY="$(cat "$REPO/api_key.txt")"
fi

# Prefill chunk size (--max-num-batched-tokens below): 2048 measured best on this card.
# Raising it is a large loss, not a win -- a 24k-token cold prompt took 22.7s at 2048 and
# 118.9s at 8192, same config otherwise. MAX_BATCHED is a knob only so that stays
# reproducible without editing this file.
exec "$VENV/bin/vllm" serve "$MODEL" \
  --served-model-name qwen3.8-27b \
  --host 0.0.0.0 --port $PORT \
  --gpu-memory-utilization $GPU_UTIL \
  --max-model-len $MAX_LEN \
  --max-num-seqs $MAX_SEQS \
  --api-server-count $API_SERVERS \
  ${LM_ONLY_ARG} \
  $ATTN_ARGS \
  --mamba-ssm-cache-dtype float16 \
  ${ASYNC_ARGS} \
  --max-num-batched-tokens ${MAX_BATCHED:-2048} \
  --speculative-config "$SPEC_CFG" \
  --compilation-config "{\"max_cudagraph_capture_size\":$CG,\"custom_ops\":[\"+rms_norm\",\"+silu_and_mul\"]${CG_MODE}}" \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --safetensors-load-strategy=prefetch \
  ${EXTRA_ARGS}
