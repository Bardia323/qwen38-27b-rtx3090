# Qwen3.8-27B on one RTX 3090

Serving setup for [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) on a
single 24 GB consumer GPU with vLLM. 150k token context, OpenAI-compatible
API with key auth, and two ready-made configs depending on what you're doing:

| | [batch/](batch/) | [single-user/](single-user/) |
|---|---|---|
| for | API backends, pipelines, many concurrent requests | one or a few people chatting |
| aggregate, 64 concurrent (128 in / 512 out) | **942 tok/s** end-to-end, ~1,094 steady-state decode (1,042 / ~1,222 with all layers int8) | n/a (8 slots) |
| single-stream, realistic prompts | 46 tok/s | **~114 tok/s** at default sampling, **118-124 tok/s** greedy (`CTX=fast`, 64k context; 95 / 100 with `CTX=long`, 150k); **117-127 / 120-134 tok/s** with the DFlash2 block drafter (`SPEC=dflash2`) |
| trick | 16-bit recurrent state + int8 tensor-core GEMMs | MTP speculation with 4 cheap drafts, a draft vocabulary that covers what the model says, calibrated int4 lm_head/drafter, split-KV verify attention; optionally DFlash2 (7 drafts in one pass, int4-requantized, vLLM PR #52816 backported) |

Both modes share one install — the mode is just which launch script you run.
Speculation wins below ~8 concurrent users, plain batching above. Numbers are
`vllm bench serve` on an RTX 3090 at a 250 W power limit.

Prefill is a separate budget from either: ~1,810 tok/s at 1k inputs in batch
mode (~1,210 single-user), ~1,000 tok/s at 100k, so a 100k prompt costs ~100 s
of TTFT ([full matrix](batch/README.md#prefill)). How each number was won:
[docs/optimizations.md](docs/optimizations.md).

## Quick start

Docker (recommended — image build, model download and requantization, then
the server; the API is OpenAI-compatible on port 18020):

```bash
git clone https://github.com/syv-ai/qwen38-27b-rtx3090 && cd qwen38-27b-rtx3090
echo "VLLM_API_KEY=$(openssl rand -hex 24)" > .env
docker compose --profile single up -d      # one or a few users; or --profile batch
```

Or by hand in a venv (same steps: model download, requantization, vLLM
patches, `verify.sh`) — see [Setup](#setup). Then pick a mode:
[batch/](batch/) for throughput, [single-user/](single-user/) for latency.

## Benchmarks

Full tables per mode in [batch/README.md](batch/README.md) and
[single-user/README.md](single-user/README.md); quality in
[docs/quality.md](docs/quality.md). Reproduce any of it with
`bash bench/run_benchmarks.sh batch|single` against your own server.

### vs. ninfer-3090

[ninfer-3090](https://github.com/Don-Chad/ninfer-3090) publishes cohort
benchmarks for this exact model on this exact card. Their protocol is C
requests of random tokens, 1,024 output tokens each. Random-token prompts turn
out to be a bad yardstick for speculative decoding — the model's continuation
of noise is either extremely repetitive (drafts accepted 80%+ of the time,
we've measured 151 tok/s single-stream that way) or full of rare tokens (near
zero acceptance) depending on the sample — so we report both modes on
realistic chat prompts instead (8 mixed English/Danish/code tasks, 1,024-token
answers, model-default sampling):

| Cohort | ninfer-3090 (MTP3, random tokens) | this repo, batch mode | this repo, single-user mode | |
|---|---|---|---|---|
| C1 | 70.19 tok/s | 45.4 tok/s | **111.1 tok/s** | +58% |
| C2 | 89.43 tok/s | 81.8 tok/s | **173.1 tok/s** | +94% |
| C4 | 97.89 tok/s | 153.8 tok/s | **220.9 tok/s** (256.8 with `CTX=long`, k=3) | +126% |
| C8 | 161.28 tok/s | 298.4 tok/s | **309.2 tok/s** (327.9 with `CTX=long`, k=3) | +92% |
| C64, batch mode (128 in / 512 out) | not supported | 942 tok/s | | |

With the DFlash2 drafter (`SPEC=dflash2`, the fastest single-user config for a
handful of users): 117.8 / 172.6 / 233.8 / 257.3 tok/s at C1 / C2 / C4 / C8, and
125.7 / 191.1 / 254.1 / 241.3 greedy.

(single-user numbers: `CTX=fast` + fast variant, e2e output tok/s from
`bench/run_benchmarks.sh single`; four drafts win up to C2, three drafts win
from C4 up.) Peak VRAM is comparable (23.0 GiB vs their 22.1 GiB at C8). Their engine is
good work — the gap is mostly vLLM's continuous batching plus the extra memory
this repo's requantization frees up.

### Quality

The whole stack is quantized, so the honest question is what it costs. Short
version: **IFBench 78.3** prompt-level strict vs 79.5 for the unquantized model
(one point), **perplexity 8.09** on 33k held-out tokens, **GSM8K 96.5%** (200
questions, greedy). Speculative decoding — MTP, DFlash2 and the lookup drafter —
is exact by construction and changes none of it; the int8-activation steps in
batch mode are the only knobs that trade accuracy for speed, and they cost
0.9-3.7% perplexity depending on how far you push them. Per-configuration
tables: [docs/quality.md](docs/quality.md).

### Why this isn't just `vllm serve`

Nine things, one line each — the reasoning, measurements and code pointers are
in [docs/optimizations.md](docs/optimizations.md):

1. **Both embedding matrices requantized** (`quant_lm_head.py`, `quant_embed.py`)
   — the public W4A16 quants leave two 2.5 GB bf16 matrices alone. 2.6 GB back.
2. **A two-line vLLM patch** so the model code actually uses vLLM's quantized
   embedding kernel (`patches/qwen3_5-embed-quant.patch`).
3. **16-bit recurrent state** — the GDN state, not the KV cache, is what bounds
   concurrency here: 37 of 64 requests were running before this.
4. **int8 tensor cores for the batched GEMMs, with a bug fix** — vLLM's W4A8
   Marlin path produces garbage on this checkpoint (negative group scales read
   as unsigned); two patches fix it and make it per-layer selectable.
5. **Cheap speculative drafts, and a draft vocabulary counted over the model's
   own outputs** — 97.5% coverage vs 92% for a web-text list, and every miss is
   a forced rejection. Worth 10% of single-stream throughput on its own.
6. **Two decode-path patches for the verify step** — split-KV attention for
   multi-query decode (FA2 leaves 58 of 82 SMs idle there) and a sort-free
   top-k/top-p sampler.
7. **Tuned flags that are easy to get wrong**, plus vLLM PR #50021 vendored for
   an illegal memory access in the DeltaNet spec-decode kernels.
8. **Speculation that reads the context** — when the model is reproducing
   something from its prompt, draft it from the prompt
   (`patches/dflash2-lookup-drafting.patch`): +29% tokens per step on quoting
   and listing work, 0.075 ms per step, still lossless.
9. **Prefix caching for a hybrid model** — opt-in upstream; `PREFIX_CACHE=1`
   makes a follow-up chat turn on a 24k document cost ~1 s instead of ~23 s, and
   64 requests sharing a system prompt 17 s instead of 222 s.

### What each step buys

Measured cumulatively on the 3090, 64 concurrent, 128 in / 512 out, `vllm bench
serve` random dataset:

| step | what it does | e2e output tok/s | steady-state decode |
|---|---|---|---|
| W4A16 AutoRound body (as published) + fp8 KV | int4 Marlin kernels, 66.7k-token pool | 370 (48 conc, 256/256) | — |
| + lm_head / embed_tokens int8 | 2.6 GB of cache pages back | 516 | ~585 (37 requests resident) |
| + fp16 recurrent state | 64 requests resident, half the state traffic | 707 | ~830 |
| + int8 activations, MLP (default) | int8 tensor cores on 74% of the FLOPs | 942 | ~1,094 |
| + int8 activations, everything (`INT8_LAYERS=.`, needs `GPU_UTIL=0.95`) | | 1,042 | ~1,222 |

And single-stream on realistic prompts (single-user mode, T = model default /
greedy):

| step | tok/s | tokens per step | draft acceptance, position 0 |
|---|---|---|---|
| no speculation | 46 / 46 | 1.0 | — |
| MTP-2 as shipped (bf16 drafter, full head, fp32 state) | 66 / 79 | 2.1 / 2.4 | 65% / 80% |
| MTP-4, int8 drafter, 40k draft head, fp16 state | 78 / 99 | 2.2 / 2.7 | 58% / 70% |
| + probabilistic draft sampling (`CTX=fast`, k=4) | 90 / 98 | 2.6 / 2.7 | 69% / 70% |
| same with 3 drafts on FlashInfer/fp8 KV (`CTX=long`, 150k) | 84 / 89 | 2.5 / 2.4 | 69% / 71% |
| + sampler patch, split-KV verify attention | 93 / 99 | 2.6 / 2.6 | 69% / 70% |
| + draft vocab counted over the model's own outputs | 107 / 109 | 2.9 / 2.9 | 74% / 74% |
| + GPTQ-int4 lm_head (calibrated) | 109 / 112 | 2.8 / 2.8 | 73% / 73% |
| + GPTQ-int4 MTP module (**fast variant, shipped**) | **~114 / 118-124** | 2.8 / 2.9-3.0 | 74% / 77% |
| DFlash2 block drafter instead of MTP (`SPEC=dflash2`, int4-requantized) | **118 / 126** | 3.14 / 3.34 | ~75% / ~78% |
| + drafting from the context (`LOOKUP=1`, on by default) | up to **131** on quoting/listing work | 3.3-4.65 | |

(Steps 4-6 are the same 8-prompt protocol; greedy is deterministic for a
given server and request order but differs between configs and even with
prefix-cache hits, so single runs carry ±3-5% on tokens/step —
`bench/run_benchmarks.sh single` reproduces 113.6 / 118.3 tok/s decode at C1,
the best repeats read 115 / 124.)
Going deeper (k=5) loses again: 106 / 105. k=4 is the knee, but on vLLM
0.27.1's FlashInfer backend (needed for fp8 KV, i.e. for 150k context) four
drafts crash the engine with an illegal memory access as soon as one request
finishes while another is mid-generation — club-3090 reports the same "n=4
eventually dies, n=3 stable" pattern — so `CTX=long` drafts 3 and gives up
~7%; `CTX=fast` (FlashAttention, bf16 KV, ~64k context, the default) keeps k=4
and is also the only backend the split-KV attention patch applies to.

Two things that did *not* help, measured rather than assumed: fine-tuning the
MTP head on the model's own outputs (KL halves, greedy top-1 on response
tokens unchanged; `drafter/README.md`), and retuning Marlin's tile
configuration for M ≤ 16 on sm86 (`marlin-tune/`: 3-7% per GEMM in isolation,
nothing measurable end to end — the remaining gap to peak bandwidth is the
memory system's ramp on 16-92 MB reads, not the kernel).

## Setup

You need: a 24 GB Ampere or newer NVIDIA card, a recent driver, Python 3.12,
~40 GB disk. Everything below is CPU-safe to run while the GPU does other
things. (Or skip the venv and use the container: [docs/docker.md](docs/docker.md).)

```bash
git clone https://github.com/syv-ai/qwen38-27b-rtx3090 ~/qwen-serving
cd ~/qwen-serving

python3 -m venv venv
venv/bin/pip install vllm huggingface_hub hf_transfer ninja

# model, ~19.5 GB
HF_HUB_ENABLE_HF_TRANSFER=1 venv/bin/hf download \
  dbirks/Qwen3.8-27B-W4A16-AutoRound \
  --local-dir models/Qwen3.8-27B-W4A16-AutoRound

# requantize lm_head + embeddings + the MTP draft module (CPU only, a few minutes)
venv/bin/python quant_lm_head.py models/Qwen3.8-27B-W4A16-AutoRound
venv/bin/python quant_embed.py   models/Qwen3.8-27B-W4A16-AutoRound
venv/bin/python quant_mtp.py     models/Qwen3.8-27B-W4A16-AutoRound
# 40k-token draft head for single-user mode (uses the shipped id list)
venv/bin/python build_draft_vocab.py models/Qwen3.8-27B-W4A16-AutoRound --ids draft_vocab_ids.json
# single-user "fast" variant (~1 GB from the Hub, hardlinks the rest): int4-GPTQ
# lm_head + drafter; single-user/start_qwen.sh picks it up automatically
venv/bin/python fetch_fast_variant.py
# optional: the W4A16 DFlash2 block drafter (1.2 GB) for SPEC=dflash2 single-user mode
venv/bin/python fetch_dflash2.py

# patch vllm (all written against 0.27.1; reapply after upgrades)
for p in patches/*.patch; do
  patch -p1 -d venv/lib/python3.12/site-packages/vllm < $p
done
# optional: the KVarN 4/2-bit KV cache for 262k context (docs/long-context.md)
bash kvarn/install.sh

# api key
openssl rand -hex 24 > api_key.txt
```

Then `bash verify.sh --no-server` — it checks the venv and vLLM version, that
every patch in `patches/` is actually applied, and that the model has been
requantized (lm_head, embeddings, MTP module, draft head). Then pick a mode
and follow its README:

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

To check the numbers on your own card: `bash verify.sh` (also probes the live
server and prints which attention backend and KV pool it came up with), then
`bash bench/run_benchmarks.sh batch` or `... single` reproduces the tables
above against the running server (`--prefill` and `--long` add the prefill
matrix and the long-context rows), `bash bench/real_rep.sh <tag> 3 0` repeats
the single-stream row, and `python bench/quality_battery.py <tag>` the
perplexity / GSM8K rows.

## The rest

| | |
|---|---|
| [docs/optimizations.md](docs/optimizations.md) | Every optimization in full: why it was needed, what it measured, which patch implements it. Includes the two speculative-decoding modes (MTP and DFlash2) and the lookup drafter. |
| [docs/gotchas.md](docs/gotchas.md) | 18 things that each cost us hours — read before debugging something that looks like a vLLM bug. |
| [docs/quality.md](docs/quality.md) | IFBench, perplexity and GSM8K per configuration. |
| [docs/docker.md](docs/docker.md) | The container image, and an independent WSL2 reproduction. |
| [docs/long-context.md](docs/long-context.md) | 262k context with the KVarN 4/2-bit KV cache, and what vLLM's own per-token-head KV modes are worth here. |
| [batch/](batch/) · [single-user/](single-user/) | The two serving modes: full benchmark tables, every env knob, systemd units. |
| [drafter/](drafter/) | How the draft vocabulary, the int4 drafters and the DFlash2 requantization were built — including what did not work. |
| [kvarn/](kvarn/) · [marlin-tune/](marlin-tune/) | The KVarN port, and a Marlin tile-tuning experiment that did not pay off. |

## License

Apache-2.0, same as the model.
