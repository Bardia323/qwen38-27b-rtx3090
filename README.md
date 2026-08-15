# Qwen3.8-27B on one RTX 3090

Serving setup for [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) on a
single 24 GB consumer GPU with vLLM. 150k token context, OpenAI-compatible
API with key auth, and two ready-made configs depending on what you're doing:

| | [batch/](batch/) | [single-user/](single-user/) |
|---|---|---|
| for | API backends, pipelines, many concurrent requests | one or a few people chatting |
| aggregate, 64 concurrent | **417 tok/s** (672 peak) | n/a (8 slots) |
| single-stream (1,024-token generations) | 46 tok/s | **82 tok/s** |
| trick | big batches, no speculation | MTP speculative decoding |

Both share the same install; the mode is just which launch script you run.
The crossover is around 8 concurrent users: below that, speculation wins;
above, plain batching. All numbers measured with `vllm bench serve` on an
RTX 3090 at a 250 W power limit (stock is 350 W, so probably conservative).
Full tables in each mode's README.

Prefill is its own budget, independent of mode and — importantly — of
concurrency: ~1,210 tok/s of prompt processing at 1k inputs, degrading only
to 795 tok/s at 100k (just 16 of 64 layers pay quadratic attention). A 100k
prompt costs ~2 minutes of TTFT, and concurrent prompts queue linearly behind
each other. Full matrix in [batch/README.md](batch/README.md#prefill).

### vs. ninfer-3090

[ninfer-3090](https://github.com/Don-Chad/ninfer-3090) publishes cohort
benchmarks for this exact model on this exact card. Same protocol (C requests,
1,024 output tokens each), end-to-end throughput:

| Cohort | ninfer-3090 (MTP3) | this repo (single-user mode) | |
|---|---|---|---|
| C1 | 70.19 tok/s | 81.85 tok/s | +17% |
| C2 | 89.43 tok/s | 119.38 tok/s | +33% |
| C4 | 97.89 tok/s | 243.62 tok/s | +149% |
| C8 | 161.28 tok/s | 212.75 tok/s | +32% |
| C8, batch mode | — | 289.60 tok/s | +80% |
| C64, batch mode | not supported | 417.5 tok/s | |

Peak VRAM is comparable (23.0 GiB vs their 22.1 GiB at C8). Their engine is
good work — the gap is mostly vLLM's continuous batching plus the extra 2.6 GB
of cache pages this repo's embedding requant frees up.

## Why this isn't just `vllm serve`

Three things in this repo that stock vLLM doesn't give you:

1. **Both embedding matrices requantized.** Qwen3.8-27B has untied embeddings,
   so the public W4A16 quants carry two separate 2.5 GB bf16 matrices (lm_head
   and embed_tokens) that nobody bothered to quantize. `quant_lm_head.py` and
   `quant_embed.py` requantize both to int8 group-128 in place (~0.6%
   round-trip error, no quality regression we could find). That's 2.6 GB of
   VRAM back, and on this architecture spare VRAM converts directly into batch
   size: 48 of the 64 layers are Gated DeltaNet with a fixed ~49 MB state per
   sequence, so concurrency is bounded by the memory pool, not by KV cache.
   Combined effect was +18% aggregate throughput.
2. **A small vLLM patch.** vLLM ships a dequant-on-gather kernel for
   int-quantized embedding tables but the qwen3_5 model code never wires it
   up — neither in the main model nor in the MTP draft module that single-user
   mode depends on. `patches/qwen3_5-embed-quant.patch` fixes both (two lines
   each). Reapply after vLLM upgrades.
3. **Tuned flags that are easy to get wrong**, each documented in the launch
   scripts and the gotchas below.

### What each quantization step buys

Measured cumulatively on the 3090, 256 in / 256 out at 64 concurrent. As
described in point 1 above, spare VRAM converts directly into batch capacity
on this architecture, which is why the weight savings show up as throughput:

| step | what it does | weights in VRAM | cache pool | aggregate |
|---|---|---|---|---|
| W4A16 AutoRound body (as published) | all linears on int4 Marlin kernels | 16.84 GB | 66.7k tokens | 370 tok/s (48 conc, 8k ctx) |
| + fp8 KV cache | halves KV bytes per token | 16.84 GB | 155.2k tokens | 354 tok/s, but 131k+ context becomes possible at all |
| + lm_head int8 (`quant_lm_head.py`) | logits matmul moves to int8 Marlin, 2.5 GB bf16 freed | 15.43 GB | 192.4k tokens | 398 tok/s |
| + embed_tokens int8 (`quant_embed.py` + patch) | embedding gather dequantizes on the fly | 14.26 GB | 200.0k tokens | 417 tok/s |

Quality checked after each step: ~0.6% round-trip error per matrix, no change
we could detect in Danish/English QA, arithmetic, or 20k-token needle
retrieval. Going below int8 on lm_head/embeddings is where we'd start to
worry; int4 there saves another ~1.3 GB if you want to gamble.

## Setup

You need: a 24 GB Ampere or newer NVIDIA card, a recent driver, Python 3.12,
~40 GB disk. Everything below is CPU-safe to run while the GPU does other
things.

```bash
git clone https://github.com/syv-ai/qwen38-27b-rtx3090 ~/qwen-serving
cd ~/qwen-serving

python3 -m venv venv
venv/bin/pip install vllm huggingface_hub hf_transfer ninja

# model, ~19.5 GB
HF_HUB_ENABLE_HF_TRANSFER=1 venv/bin/hf download \
  dbirks/Qwen3.8-27B-W4A16-AutoRound \
  --local-dir models/Qwen3.8-27B-W4A16-AutoRound

# requantize lm_head + embeddings (a minute or two, CPU only)
venv/bin/python quant_lm_head.py models/Qwen3.8-27B-W4A16-AutoRound
venv/bin/python quant_embed.py   models/Qwen3.8-27B-W4A16-AutoRound

# patch vllm so the quantized embedding table is actually used
patch -p1 -d venv/lib/python3.12/site-packages/vllm \
  < patches/qwen3_5-embed-quant.patch

# api key
openssl rand -hex 24 > api_key.txt
```

Then pick a mode and follow its README:

- **[batch/](batch/)** — throughput. `bash batch/start_qwen.sh`
- **[single-user/](single-user/)** — latency. `bash single-user/start_qwen.sh`

First start takes a few minutes (torch.compile, CUDA graph capture, flashinfer
JIT). Test it:

```bash
curl http://localhost:18020/v1/chat/completions \
  -H "Authorization: Bearer $(cat api_key.txt)" \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3.8-27b",
       "messages": [{"role": "user", "content": "hej"}],
       "chat_template_kwargs": {"enable_thinking": false}}'
```

Qwen recommends temperature 0.7 / top_p 0.8 for instruct mode, and 1.0 / 0.95
with thinking enabled (the default).

## Gotchas

Things that each cost us hours, in rough order of pain:

1. **Restart onto a dirty GPU and you silently lose 25%.** vLLM profiles free
   memory once at startup. If the previous process is still releasing VRAM at
   that moment, the cache pool comes out ~40% smaller and stays that way. No
   warning, the server runs fine, throughput is just quietly bad. The systemd
   units in both mode dirs carry an `ExecStartPre` gate that waits for the GPU
   to be actually free.
2. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is not optional.** The
   DeltaNet prefill kernels allocate transient workspace; without it the
   allocator fragments and the engine OOMs at runtime once
   `gpu-memory-utilization` goes past ~0.975.
3. **With MTP enabled, even that isn't enough — single-user mode runs
   `gpu-memory-utilization 0.90`.** The speculative decode path's DeltaNet
   workspace grows beyond what vLLM's startup memory profiling measures, and
   the engine dies mid-request on long generations at 0.95+. It survives short
   benchmarks, which is exactly how it fools you: our 256-token runs passed,
   1,024-token runs crashed reliably. The lower setting costs nothing at 8
   sequence slots.
4. **Bigger prefill chunks make things worse.** `--max-num-batched-tokens
   8192` inflates the profiled activation peak, which shrinks the cache pool,
   which caps concurrency. 2048 wins on this card.
5. **Benchmark twice.** The first run after any restart includes JIT warmup
   and reads 30-50% low. We've seen 216 vs 402 tok/s, same config, back to
   back.
6. **`--language-model-only` drops the vision tower cleanly** (no weights
   loaded). If you don't need images, that's 2.7 GB.
7. **Don't chase the DeltaNet kernels.** `bench/tune_gdn.py` microbenchmarks
   the decode kernel across block/warp configs: it already runs at ~85% of the
   3090's memory bandwidth and every variant lands within 3%. Kept in the repo
   so you don't spend the afternoon we spent.

## Full 256k context?

Doesn't fit, and no serving engine changes that: fp8 KV for 262k tokens is
8.4 GB on its own, plus 14.3 GB of weights, plus state pages. After the
embedding requant the cache pool holds 200k tokens, so the real ceiling is
~195k for a single request; we ship 150k as the default because a max-length
request at the ceiling monopolizes the whole pool while it runs. Raise
`MAX_LEN` toward 195000 if you want it. The remaining gap to 262k is ~2.5 GB
that a 24 GB card simply doesn't have — that's what 32 GB cards are for.

## License

Apache-2.0, same as the model.
