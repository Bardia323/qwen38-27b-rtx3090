# Qwen3.8-27B on one RTX 3090

Serving setup for [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) on a
single 24 GB consumer GPU with vLLM. 150k token context, OpenAI-compatible
API with key auth, and two ready-made configs depending on what you're doing:

| | [batch/](batch/) | [single-user/](single-user/) |
|---|---|---|
| for | API backends, pipelines, many concurrent requests | one or a few people chatting |
| aggregate, 64 concurrent (128 in / 512 out) | **876 tok/s** end-to-end, ~1,050 steady-state decode (1,025 / ~1,150 with all layers int8) | n/a (8 slots) |
| single-stream, realistic prompts | 46 tok/s | **~114 tok/s** at default sampling, **118-124 tok/s** greedy (`CTX=fast`, 64k context; 95 / 100 with `CTX=long`, 150k) |
| trick | 16-bit recurrent state + int8 tensor-core GEMMs | MTP speculation with 4 cheap drafts, a draft vocabulary that covers what the model says, calibrated int4 lm_head/drafter, split-KV verify attention |

Both share the same install; the mode is just which launch script you run.
The crossover is around 8 concurrent users: below that, speculation wins;
above, plain batching. All numbers measured with `vllm bench serve` on an
RTX 3090 at a 250 W power limit (stock is 350 W, so probably conservative).
Full tables in each mode's README.

Where these numbers came from, in one line each: fixing a per-request memory
cost that silently capped the batch server at 37 running requests (2.4×), an
int8-activation kernel path that vLLM already ships but that produces garbage
on this checkpoint until a sign bug is worked around (1.4×), making
speculative drafts cheap enough that four of them pay off (1.3×), and — the
single biggest single-user win — a draft vocabulary counted over the model's
own outputs instead of web text (1.1×; see "Single-user: 124 tok/s"). All of it
is in `patches/`, the `quant_*.py` / `build_draft_vocab.py` scripts and
`drafter/`.

Prefill is its own budget, independent of mode and of concurrency: ~1,810
tok/s of prompt processing at 1k inputs in batch mode (int8 tensor cores;
~1,210 tok/s on the W4A16 kernels single-user mode uses), degrading gently
with length because only 16 of 64 layers pay quadratic attention — 1,000 tok/s
at 100k. A 100k prompt costs ~100 s of TTFT (~130 s in single-user mode), and
concurrent prompts queue linearly behind each other. Full matrix in
[batch/README.md](batch/README.md#prefill).

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
| C64, batch mode (128 in / 512 out) | not supported | 876 tok/s | | |

(single-user numbers: `CTX=fast` + fast variant, e2e output tok/s from
`bench/run_benchmarks.sh single`; four drafts win up to C2, three drafts win
from C4 up.) Peak VRAM is comparable (23.0 GiB vs their 22.1 GiB at C8). Their engine is
good work — the gap is mostly vLLM's continuous batching plus the extra memory
this repo's requantization frees up.

## Quality

Does W4A16 + int8 embeddings + fp8 KV + the tricks above cost accuracy? Three
checks against this exact serving stack:

**IFBench** ([AllenAI's](https://github.com/allenai/IFBench) out-of-distribution
instruction-following benchmark, 299 prompts, official eval scripts), thinking
enabled at `reasoning_effort: xhigh` (the model default), model-default
sampling:

| accuracy | prompt-level | instruction-level |
|---|---|---|
| strict, W4A16 stack | **78.3** | 79.9 |
| loose, W4A16 stack | 81.7 | 82.8 |
| strict, batch mode default (int8 MLP activations, fp16 state) | **78.3** | 80.5 |
| loose, batch mode default | 80.3 | 82.3 |

Qwen's [model card](https://huggingface.co/Qwen/Qwen3.8-27B) reports **79.5**
for the unquantized model, so the W4A16 quantization stack costs about one
point on the headline metric (prompt-level strict), and the batch-mode int8
activations cost nothing measurable on it (the two runs trade places within
sampling noise on the sub-metrics).

**Perplexity** on ~33k tokens of held-out text (English Wikipedia, Danish web
text, Python source), and **GSM8K** (200 test questions, greedy, thinking off):

| batch-mode config | PPL en | PPL da | PPL code | PPL all | GSM8K | 64-conc e2e (128/512) |
|---|---|---|---|---|---|---|
| W4A16, fp32 state (as shipped before) | 10.68 | 10.85 | 3.05 | 8.045 | — | 516 tok/s |
| W4A16, fp16 state | 10.68 | 10.85 | 3.05 | 8.044 | 95.5% | 707 tok/s |
| + int8 activations, gate/up projections | 10.74 | 10.93 | 3.10 | 8.12 (+0.9%) | 95.5% | 787 tok/s |
| + int8 activations, whole MLP (**default**) | 10.88 | 11.05 | 3.15 | 8.22 (+2.2%) | 95.0% | 876 tok/s |
| + int8 activations, all linear layers | 10.93 | 11.29 | 3.20 | 8.34 (+3.7%) | — | 1,025 tok/s |

Reading: the 16-bit recurrent state is free; every int8-activation step costs a
little perplexity, mostly on code, and buys throughput. The default takes the
middle row; `INT8_LAYERS=gate_up` and `INT8_ACT=` (off) are one env var away,
and `INT8_LAYERS=.` gives you the last row.

Single-user mode is W4A16 (int8 activations buy nothing at batch size 1) and
speculative decoding is exact by construction, so with the base requantization
its quality is the W4A16 row. The single-user **fast variant** additionally
runs lm_head at int4 (GPTQ, calibrated on the model's own hidden states):

| single-user config | PPL en | PPL da | PPL code | PPL all | GSM8K | C1 tok/s (default / greedy) |
|---|---|---|---|---|---|---|
| base requantization (int8 lm_head), new draft vocab | 10.68 | 10.85 | 3.05 | 8.045 | 95.5% | 107 / 109 |
| int4 lm_head, round-to-nearest (not shipped) | 10.81 | 11.09 | 3.07 | 8.17 (+1.5%) | — | 109 / 112 |
| **fast variant**: int4 lm_head GPTQ + int4 MTP GPTQ | 10.77 | 10.91 | 3.06 | 8.095 (+0.6%) | 96.5% | ~114 / ~124 |

The MTP module's precision never touches output quality (drafts are verified
exactly); it only moves acceptance, and the calibrated int4 keeps it.

## Why this isn't just `vllm serve`

Seven things in this repo that stock vLLM doesn't give you:

1. **Both embedding matrices requantized.** Qwen3.8-27B has untied embeddings,
   so the public W4A16 quants carry two separate 2.5 GB bf16 matrices (lm_head
   and embed_tokens) that nobody bothered to quantize. `quant_lm_head.py` and
   `quant_embed.py` requantize both to int8 group-128 in place (~0.6%
   round-trip error, no quality regression we could find). That's 2.6 GB of
   VRAM back.
2. **A small vLLM patch for those embeddings.** vLLM ships a dequant-on-gather
   kernel for int-quantized embedding tables but the qwen3_5 model code never
   wires it up — neither in the main model nor in the MTP draft module.
   `patches/qwen3_5-embed-quant.patch` fixes both (two lines each).
3. **16-bit recurrent state.** 48 of the 64 layers are Gated DeltaNet with a
   fixed recurrent state per sequence, and Qwen's config asks for it in fp32:
   ~150 MB per request, allocated up front, read and written on every decode
   step. On this architecture that state — not the KV cache — is what bounds
   concurrency: with `--max-num-seqs 64` only 37 requests were ever actually
   running (log line `Running: 37 reqs, Waiting: 27`). `--mamba-ssm-cache-dtype
   float16` halves the footprint and the traffic; all 64 run, and perplexity is
   unchanged to three decimals (fp16 keeps 10 mantissa bits; we did not use
   bf16's 7).
4. **int8 tensor cores for the batched GEMMs, with a bug fix.** At 40-64
   concurrent sequences the decode step is bound by fp16 tensor-core math
   (~63 TFLOPS sustained at 250 W). vLLM already has a W4A8-INT8 Marlin path
   (`VLLM_MARLIN_INPUT_DTYPE=int8`: weights stay int4, activations are
   quantized to int8 per token, the MMA runs on int8 tensor cores at 4× the
   rate) — but on this checkpoint it produced garbage while benchmarking
   beautifully. The kernel reads its int16-requantized group scales as
   *unsigned*, and AutoRound symmetric exports have ~50% negative scales.
   `patches/marlin-int8-negative-scales.patch` folds the sign into the int4
   codes at load time; `patches/marlin-int8-layer-select.patch` lets you pick
   which layers get int8 activations (and keeps it off the int8-weight lm_head,
   which would otherwise refuse to load).
5. **Cheap speculative drafts, and a draft vocabulary that covers what the
   model actually says.** The shipped MTP draft module is bf16 (850 MB) and
   every draft token also runs the full 248k-row lm_head (1.3 GB), so each
   extra draft cost ~3 ms and MTP-3 was already slower than MTP-2.
   `quant_mtp.py` requantizes the draft module (int8; the fast variant uses
   GPTQ int4, `drafter/`), `build_draft_vocab.py` builds a 40k-token draft head
   and `patches/qwen3_5-mtp-draft-vocab.patch` makes the drafter use it. A
   draft now costs ~0.5-1 ms and four of them pay off. The id list matters
   more than anything else in this repo's single-user numbers: a token outside
   the draft vocabulary can never be proposed, so it is a guaranteed rejection
   that also cuts the chain. The list we now ship (`draft_vocab_ids.json`) is
   counted over 5.4M tokens of the model's own outputs and covers 97.5% of what
   it generates (96% on code); the earlier web-text list covered 92% (83% on
   code) and cost 10% of single-stream throughput on its own.
6. **Two decode-path patches for the multi-query verify step.**
   `patches/spec-decode-attn.patch`: FlashAttention-2 only splits the KV
   sequence across thread blocks when a request has one query token; the MTP
   verify step has five, so a 24-head model runs attention on 24 of the 3090's
   82 SMs — 57 µs per layer at 1.5k context, 1.3 ms at 16k. A small Triton
   split-KV kernel replaces it (23 µs / 120 µs). `patches/sampler-small-topk-
   fast-softmax.patch`: vLLM's top-k/top-p masking sorts the whole 248k vocab
   for every row and its softmax runs one thread block per row (140 µs for a
   single 248k-wide row, called several times per step); with top-k ≤ 64 known
   on the host the mask is one `torch.topk`, the softmax is multi-block, and
   drafts are sampled from the same truncated support as the target. Together
   +4% at default sampling.
7. **Tuned flags that are easy to get wrong**, each documented in the launch
   scripts and the gotchas below, plus vLLM PR
   [#50021](https://github.com/vllm-project/vllm/pull/50021) vendored as
   `patches/vllm-pr50021-gdn-spec-bounds.patch` (bounds checks in the DeltaNet
   speculative-decode kernels; we hit the illegal-memory-access it fixes with
   several concurrent MTP requests).

### What each step buys

Measured cumulatively on the 3090, 64 concurrent, 128 in / 512 out, `vllm bench
serve` random dataset:

| step | what it does | e2e output tok/s | steady-state decode |
|---|---|---|---|
| W4A16 AutoRound body (as published) + fp8 KV | int4 Marlin kernels, 66.7k-token pool | 370 (48 conc, 256/256) | — |
| + lm_head / embed_tokens int8 | 2.6 GB of cache pages back | 516 | ~585 (37 requests resident) |
| + fp16 recurrent state | 64 requests resident, half the state traffic | 707 | ~830 |
| + int8 activations, MLP (default) | int8 tensor cores on 74% of the FLOPs | 876 | ~1,050 |
| + int8 activations, everything | | 1,025 | ~1,150 |

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

# requantize lm_head + embeddings + the MTP draft module (CPU only, a few minutes)
venv/bin/python quant_lm_head.py models/Qwen3.8-27B-W4A16-AutoRound
venv/bin/python quant_embed.py   models/Qwen3.8-27B-W4A16-AutoRound
venv/bin/python quant_mtp.py     models/Qwen3.8-27B-W4A16-AutoRound
# 40k-token draft head for single-user mode (uses the shipped id list)
venv/bin/python build_draft_vocab.py models/Qwen3.8-27B-W4A16-AutoRound --ids draft_vocab_ids.json
# single-user "fast" variant (~1 GB from the Hub, hardlinks the rest): int4-GPTQ
# lm_head + drafter; single-user/start_qwen.sh picks it up automatically
venv/bin/python fetch_fast_variant.py

# patch vllm (all written against 0.27.1; reapply after upgrades)
for p in patches/*.patch; do
  patch -p1 -d venv/lib/python3.12/site-packages/vllm < $p
done
# optional: the KVarN 4/2-bit KV cache for 262k context (see below)
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

To check the numbers on your own card: `bash verify.sh` (now also probes the
live server and prints which attention backend and KV pool it came up with),
then `bash bench/run_benchmarks.sh batch` or `... single` reproduces the
tables in this README against the running server (`--prefill` and `--long`
add the prefill matrix and the long-context rows), and
`python bench/quality_battery.py <tag>` the perplexity / GSM8K rows.

### WSL2 notes

An independent WSL2 reproduction at `e81fa39` used kernel
`6.6.87.2-microsoft-standard-WSL2`, Ubuntu 24.04, NVIDIA driver 591.86,
Docker Engine 29.2.0 / Compose 5.0.2, and one RTX 3090 exposed to the
container. All six launch configurations passed authenticated API/chat and
GPU-isolation checks with zero failed benchmark requests. The full failure
signatures and earlier five-profile matrix are in [issue #1](../../issues/1).

| profile | measured cache | representative output throughput |
|---|---:|---:|
| `single-long` | 159,326 tokens, fixed as described below | 95.39 tok/s greedy, C1 |
| `single-fast` | 93,791 tokens | 114.17 tok/s greedy, C1 |
| `single-huge` | 320,000 cold / 327,272 warm | 79.84 / 81.66 tok/s, C1 sampled |
| `batch` | 201,832 tokens | 1,041.99 / 1,038.25 tok/s, C64 |
| `batch`, `KV=int4pth` | 437,414 tokens | 1,043.84 / 1,044.06 tok/s, C64 |
| `batch`, `KV=kvarn` | 334,183 cold / 350,192 warm | 843.72 / 852.42 tok/s, C64 |

Two WSL-specific memory behaviors are worth accounting for:

1. **The ordinary batch default may fail vLLM's startup free-memory gate.**
   On an otherwise clean card, WSL reported 22.75/24.0 GiB free, less than
   the 23.33 GiB requested by `GPU_UTIL=0.972`. Launching with
   `GPU_UTIL=0.93 bash batch/start_qwen.sh` retained a 201,832-token FP8
   pool, preserving the 150k context contract and expected C64 throughput.
   Keep 0.972 as the tuned native-Linux default; 0.93 is a WSL fallback.
2. **Cold and cached starts can profile different activation peaks.** A warm
   start may turn the difference into extra KV pages and leave less transient
   headroom than the cold start. For a deterministic service, compile once
   from a cold cache, record vLLM's conservative
   `Replace gpu_memory_utilization config with --kv-cache-memory=...`
   recommendation, verify that the resulting token pool exceeds
   `MAX_LEN`, and pass that machine/profile-specific byte value through
   `EXTRA_ARGS` on later starts. Stress concurrent prefill or
   `prompt_logprobs` before promoting it. Do not copy a byte value from a
   different card or profile.

The separate Marlin-repack allocator observation from issue #1 is not changed
by this documentation PR; it may belong in vLLM rather than this recipe.

## Gotchas

Things that each cost us hours, in rough order of pain:

1. **A benchmark cannot tell you the output is garbage.** The int8-activation
   path served nonsense for an hour of beautiful throughput numbers before a
   perplexity check caught it. Whatever you change, run
   `bench/quality_battery.py` (perplexity + GSM8K against the live server)
   before you believe a tok/s number.
2. **Restart onto a dirty GPU and you silently lose 25%.** vLLM profiles free
   memory once at startup. If the previous process is still releasing VRAM at
   that moment, the cache pool comes out ~40% smaller and stays that way. No
   warning, the server runs fine, throughput is just quietly bad. The systemd
   units in both mode dirs carry an `ExecStartPre` gate that waits for the GPU
   to be actually free.
3. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is not optional.** The
   DeltaNet prefill kernels allocate transient workspace; without it the
   allocator fragments and the engine OOMs at runtime once
   `gpu-memory-utilization` goes past ~0.975.
4. **With MTP enabled, even that isn't enough — single-user mode runs
   `gpu-memory-utilization 0.93`.** The speculative decode path's DeltaNet
   workspace grows beyond what vLLM's startup memory profiling measures, and
   the engine dies mid-request on long generations at 0.95+. It survives short
   benchmarks, which is exactly how it fools you. We soak-tested 0.93 with a
   100k-token prompt plus 6k-token generations at 4 concurrent.
5. **The torch.compile cache does not know about your env vars.** Switching
   `INT8_LAYERS` between runs replays a compiled graph that expects the other
   layer set and dies with `KeyError: 'input_global_scale'`. Our patch
   registers the selection env vars with vLLM so they become part of the cache
   key; if you invent your own, `VLLM_DISABLE_COMPILE_CACHE=1`.
6. **Random-token benchmarks flatter speculative decoding.** See the ninfer
   section: the same server does 35, 83 or 151 tok/s on `--dataset-name random`
   depending on what the noise turns into. Use real prompts
   (`--dataset-name custom`).
7. **Bigger prefill chunks make things worse.** `--max-num-batched-tokens
   8192` inflates the profiled activation peak, which shrinks the cache pool,
   which caps concurrency. 2048 wins on this card.
8. **Benchmark twice.** The first run after any restart includes JIT warmup
   and reads 30-50% low.
9. **`--language-model-only` drops the vision tower cleanly** (no weights
   loaded). If you don't need images, that's 2.7 GB.
10. **`prompt_logprobs` on long prompts OOMs the engine at 0.972 utilization**
    (a 300-token prompt needs ~300 MB of fp32 logits and there is no headroom).
    Run quality checks at 0.93.
11. **Don't chase the DeltaNet kernels.** `bench/tune_gdn.py` microbenchmarks
    the decode kernel across block/warp configs: it already runs at ~85% of the
    3090's memory bandwidth and every variant lands within 3%. The state dtype
    (point 3 above) is the lever, not the kernel.

12. **The draft vocabulary is the single-user ceiling.** A draft head can only
    propose tokens in its id list; a miss is a certain rejection that also ends
    the chain. Count the list over the model's *own* outputs (`drafter/gen_data.py`,
    then the frequency step in `build_draft_vocab.py`), not over web text —
    92% vs 97.5% coverage was the difference between 98 and 109 tok/s greedy.
    Coverage saturates around 40k rows; the model only ever emits ~54k distinct
    tokens.
13. **FlashAttention-2 does not split KV for multi-query decode.** With k
    speculative tokens the verify step has k+1 queries per request and FA2's
    varlen path then runs one thread block per (request, head): 24 blocks on 82
    SMs, 57 µs per layer at 1.5k context and 1.3 ms at 16k. vLLM's Triton
    unified attention has the same restriction (`max_seqlen_q > 1` → 2-D
    kernel). `patches/spec-decode-attn.patch` (`VLLM_SPEC_DECODE_ATTN=1`, bf16
    KV only) is a 170-line Triton fix.
14. **Greedy is not deterministic across drafter configs.** The target rounds
    differently when it verifies 5 tokens vs 1, so a different drafter changes
    the generated text at near-ties and the 8-prompt acceptance numbers move
    ±3%. Repeat before trusting a small difference; `drafter/README.md` has an
    offline chain simulator that removes the noise.
15. **A stale torch.compile cache bites anything that changes tensor shapes
    behind vLLM's back.** The compiled graph bakes in e.g. the Marlin workspace
    size; a new env knob that changes it must be registered in `envs.py`
    (`patches/speed-knobs-envs.patch`) or you get `assert_size_stride ...
    expected size 328==82` from a cached artifact.

## 262k context: the KVarN KV cache

With fp8 KV the pool holds ~200k tokens, so 150k is the shipped max and ~195k
the ceiling; the model's full 262,144 is out of reach because 16 attention
layers × 4 KV heads × 256 dims × 2 bytes (K+V, fp8) is 2 KB per token. The
way past that is a smaller cache, not a different engine, and
[KVarN](https://github.com/huawei-csl/KVarN) (Huawei CSL) has the best one we
know of: Hadamard rotation + iterative variance normalization + 4-bit keys /
2-bit values per 128-token tile, at ~840 B/token/layer here. It ships as a
fork of vLLM 0.23; [kvarn/](kvarn/) is our port of its dense backend onto the
0.27.1 this repo runs (`bash kvarn/install.sh`, then `KV=kvarn` in batch mode
or `CTX=huge` in single-user mode).

Measured on the 3090 (`--kv-cache-dtype kvarn_k4v2_g128 --block-size 128`,
fp16 recurrent state, batch defaults otherwise):

| | fp8 KV (default) | KVarN k4v2 |
|---|---|---|
| KV pool, batch mode | ~205-225k tokens (150k max, ~195k ceiling) | 302-344k tokens with 64 slots, **420k with 4 slots — 262k fits with room for 1.6 such requests** |
| KV pool, single-user mode (MTP-3, `CTX=long`/`huge`) | 150k max | 200k max |
| needle-in-a-haystack, greedy | — | correct at 4k / 16k / 30k / 100k / 240k, both depths |
| perplexity (en/da/code, 33k tokens) | 8.223 | 8.236 (+0.16%) |
| prefill, 1k / 16k / 100k inputs | 1,812 / 1,595 / 997 tok/s | 1,741 / 1,569 / 1,050 tok/s (same within ±5%) |
| single stream at 100k context | TTFT 99 s, 27 ms/token | TTFT 94 s, 33 ms/token |
| 4 × 60k-token requests, 1,024 out | only 3 fit → 256 s total, ITL 33 ms | all 4 resident → 242 s total, ITL 49 ms |
| 64 concurrent short requests (128/512) | 876 tok/s | 692 tok/s (38 resident: 2048-token blocks cost as much per short request as fp8's 800-token block) |
| MTP-3 single stream, real prompts (base variant, earlier draft vocab) | 84 / 89 tok/s | 79 / 88 tok/s |

So: same VRAM, 1.6-2× the tokens, full 262k context, quality intact, prefill
unchanged, and a speed tax of ~20% on long-context decode and more on
short-request throughput (the fp16 staging pool for tiles-in-progress and the
2048-token block granularity are the costs). Which is why it's a mode and not
the default — nothing changes unless you set `KV=kvarn` (batch) or `CTX=huge`
(single-user); the KV-cache format is an engine-level choice in vLLM, so it
can't be switched per request. Port notes and what to watch when bumping vLLM
are in [kvarn/README.md](kvarn/README.md).

(vLLM 0.27.1 also has TurboQuant built in — `--kv-cache-dtype turboquant_4bit_nc`
gives a similar 413k-token pool here and about 15% slower decode, but its
chunked-prefill path allocates O(context) scratch outside the memory profile
and OOMs at 32k+ prompts on this card at 0.972 utilization, and at 128k even
at 0.90. KVarN's prefill path is bounded and did 240k.)

### The built-in per-token-head modes

vLLM 0.27.1 also ships `int8_per_token_head`, `fp8_per_token_head` and
`int4_per_token_head` (dynamic per-token, per-head scales; the int4 one with a
rotation and asymmetric zero-points), all only in the Triton attention
backend. Measured on the 3090 in the batch config at 0.93 utilization, same
script for every column (`fp8_per_token_head` does not start on sm86: Triton's
fp8 KV needs SM89+):

| | fp8 (FlashInfer) | int8_per_token_head (Triton) | int4_per_token_head (Triton) | KVarN k4v2 |
|---|---|---|---|---|
| KV pool at 0.93 util | 164k tokens | 178k | **355k — 262k fits (1.35×)** | 302-420k |
| perplexity (same battery) | 8.235 | 8.231 | 8.257 (+0.3%) | +0.16% |
| needle, greedy | 100k ok | 100k ok | 100k ok, **240k ok** | 4k…240k ok |
| prefill 1k / 16k | 1,773 / 1,601 tok/s | 1,739 / 1,187 | 1,710 / 1,194 | 1,741 / 1,569 |
| 100k context, single stream | TTFT 100 s, 26.8 ms/token | 231 s, 40.8 ms | 220 s, 41.4 ms | 94 s, 33 ms |
| 64 concurrent short (128/512) | 839 tok/s | 850 | 835 | 692 |

Reading: `int8_per_token_head` buys nothing over fp8 here (same byte per
element, quality already neutral) and costs the Triton backend's long-context
speed. `int4_per_token_head` is a genuine zero-install alternative to KVarN for
the 262k use case — it fits, passes the 240k needle, and keeps short-request
throughput that KVarN's 2048-token blocks lose — at 2.3× the prefill time and
1.5× the decode time at 100k, because vLLM's Triton attention is that much
slower than FlashInfer/FlashAttention on this card at long context (the same
backend tax the single-user mode avoids by staying on FlashAttention). If the
Triton backend catches up, it becomes the simpler choice; today KVarN is
faster at long context and `int4_per_token_head` is faster on many short
requests. To try it: `--kv-cache-dtype int4_per_token_head --attention-backend
TRITON_ATTN --max-model-len 262144` (batch/start_qwen.sh: `KV=int4pth`).

## License

Apache-2.0, same as the model.
