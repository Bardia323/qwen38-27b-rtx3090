# Qwen3.8-27B on one RTX 3090

Serving setup for [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) on a
single 24 GB consumer GPU with vLLM. 150k token context, OpenAI-compatible
API with key auth, and two ready-made configs depending on what you're doing:

| | [batch/](batch/) | [single-user/](single-user/) |
|---|---|---|
| for | API backends, pipelines, many concurrent requests | one or a few people chatting |
| aggregate, 64 concurrent (128 in / 512 out) | **876 tok/s** end-to-end, ~1,050 steady-state decode (1,025 / ~1,150 with all layers int8) | n/a (8 slots) |
| single-stream, realistic prompts | 46 tok/s | **84 tok/s** at default sampling, 89 tok/s greedy (90 / 98 with `CTX=fast`, 64k context) |
| trick | 16-bit recurrent state + int8 tensor-core GEMMs | MTP speculation with 3 cheap drafts |

Both share the same install; the mode is just which launch script you run.
The crossover is around 8 concurrent users: below that, speculation wins;
above, plain batching. All numbers measured with `vllm bench serve` on an
RTX 3090 at a 250 W power limit (stock is 350 W, so probably conservative).
Full tables in each mode's README.

Where these numbers came from, in one line each: fixing a per-request memory
cost that silently capped the batch server at 37 running requests (2.4×), an
int8-activation kernel path that vLLM already ships but that produces garbage
on this checkpoint until a sign bug is worked around (1.4×), and making
speculative drafts cheap enough that three of them pay off (1.3×). All of it is
in `patches/` and the two `quant_*.py` / `build_draft_vocab.py` scripts.

Prefill is its own budget, independent of mode and of concurrency: ~1,210
tok/s of prompt processing at 1k inputs with the W4A16 kernels (single-user
mode), roughly 40% more on batch mode's int8 tensor-core path, degrading
gently with length because only 16 of 64 layers pay quadratic attention. A
100k prompt costs ~2 minutes of TTFT, and concurrent prompts queue linearly
behind each other. Full matrix in [batch/README.md](batch/README.md#prefill).

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
| C1 | 70.19 tok/s | 45.4 tok/s | **83.4 tok/s** | +19% |
| C2 | 89.43 tok/s | 81.8 tok/s | **146.6 tok/s** | +64% |
| C4 | 97.89 tok/s | 153.8 tok/s | **256.8 tok/s** | +162% |
| C8 | 161.28 tok/s | 298.4 tok/s | **327.9 tok/s** | +103% |
| C64, batch mode (128 in / 512 out) | not supported | 876 tok/s | | |

Peak VRAM is comparable (23.0 GiB vs their 22.1 GiB at C8). Their engine is
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
speculative decoding is exact by construction, so its quality is the W4A16 row.

## Why this isn't just `vllm serve`

Six things in this repo that stock vLLM doesn't give you:

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
5. **Cheap speculative drafts.** The shipped MTP draft module is bf16 (850 MB)
   and every draft token also runs the full 248k-row lm_head (1.3 GB), so each
   extra draft cost ~3 ms and MTP-3 was already slower than MTP-2.
   `quant_mtp.py` requantizes the draft module to int8, `build_draft_vocab.py`
   builds a 40k-token draft head from the frequency of tokens in Danish,
   English and code text (`draft_vocab_ids.json` is the list we ship), and
   `patches/qwen3_5-mtp-draft-vocab.patch` makes the drafter use it. A draft
   now costs ~1 ms, three of them pay off, and with `draft_sample_method:
   probabilistic` the acceptance at temperature 1.0 is nearly what greedy gets.
6. **Tuned flags that are easy to get wrong**, each documented in the launch
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
| + probabilistic draft sampling | 90 / 98 | 2.6 / 2.7 | 69% / 70% |
| same with 3 drafts (**shipped**, see below) | **84 / 89** | 2.5 / 2.4 | 69% / 71% |

Cheaper drafts lose ~10 points of acceptance (the int8 draft module and the
truncated vocabulary each cost a few) and still win, because a step went from
~32 ms to ~27 ms while carrying more drafts. Going deeper (k=6) loses again.
k=4 is the knee, but on vLLM 0.27.1's FlashInfer backend (needed for fp8 KV,
i.e. for 150k context) four drafts crash the engine with an illegal memory
access as soon as one request finishes while another is mid-generation —
club-3090 reports the same "n=4 eventually dies, n=3 stable" pattern — so the
default config drafts 3 and gives up ~7%; `CTX=fast` (FlashAttention, bf16 KV,
~64k context) keeps k=4.

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

# patch vllm (all written against 0.27.1; reapply after upgrades)
for p in patches/*.patch; do
  patch -p1 -d venv/lib/python3.12/site-packages/vllm < $p
done

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

## Full 256k context?

Doesn't fit, and no serving engine changes that: fp8 KV for 262k tokens is
8.4 GB on its own, plus 14.3 GB of weights, plus state pages. After the
embedding requant the cache pool holds 200k tokens, so the real ceiling is
~195k for a single request; we ship 150k as the default because a max-length
request at the ceiling monopolizes the whole pool while it runs. Raise
`MAX_LEN` toward 195000 if you want it. The remaining gap to 262k is ~2.5 GB
that a 24 GB card simply doesn't have — that's what 32 GB cards are for.
(A 2-4 bit KV cache such as [KVarN](https://github.com/huawei-csl/KVarN) would
close it, but it lives in a vLLM 0.23 fork; not ported here.)

## License

Apache-2.0, same as the model.
