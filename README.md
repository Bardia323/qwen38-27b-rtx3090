# Qwen3.8-27B on one RTX 3090

Serving setup for [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) on a
single 24 GB consumer GPU with vLLM. 150k token context, OpenAI-compatible
API with key auth, and two ready-made configs depending on what you're doing:

| | [batch/](batch/) | [single-user/](single-user/) |
|---|---|---|
| for | API backends, pipelines, many concurrent requests | one or a few people chatting |
| aggregate throughput | **416 tok/s** (672 peak) @ 64 concurrent | ~145 tok/s |
| single-stream speed | ~17 tok/s | **~40 tok/s** (25 ms/token) |
| trick | big batches, no speculation | MTP speculative decoding |

Both share the same install; the mode is just which launch script you run.
For reference, [ninfer-3090](https://github.com/Don-Chad/ninfer-3090)
publishes 161 tok/s at 8 concurrent for this model on the same card. Our 3090
also runs at a 250 W power limit, so these numbers are probably conservative.

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
2. **A two-line vLLM patch.** vLLM ships a dequant-on-gather kernel for
   int-quantized embedding tables but the qwen3_5 model code never wires it
   up. `patches/qwen3_5-embed-quant.patch` fixes that. Reapply after vLLM
   upgrades.
3. **Tuned flags that are easy to get wrong**, each documented in the launch
   scripts and the gotchas below.

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
3. **Bigger prefill chunks make things worse.** `--max-num-batched-tokens
   8192` inflates the profiled activation peak, which shrinks the cache pool,
   which caps concurrency. 2048 wins on this card.
4. **Benchmark twice.** The first run after any restart includes JIT warmup
   and reads 30-50% low. We've seen 216 vs 402 tok/s, same config, back to
   back.
5. **`--language-model-only` drops the vision tower cleanly** (no weights
   loaded). If you don't need images, that's 2.7 GB.
6. **Don't chase the DeltaNet kernels.** `bench/tune_gdn.py` microbenchmarks
   the decode kernel across block/warp configs: it already runs at ~85% of the
   3090's memory bandwidth and every variant lands within 3%. Kept in the repo
   so you don't spend the afternoon we spent.

## Full 256k context?

Doesn't fit, and no serving engine changes that: fp8 KV for 262k tokens is
8.4 GB on its own, plus 14.3 GB of weights, plus state pages. ~150k is the
honest ceiling for this model on 24 GB. If you need the full window, that's
what 32 GB cards are for.

## License

Apache-2.0, same as the model.
