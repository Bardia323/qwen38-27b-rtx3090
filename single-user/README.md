# Single-user mode

For one person (or a handful) chatting with the model: coding assistant,
local chat UI, anything where you're watching tokens stream in.

The difference from batch mode is MTP speculative decoding. Qwen ships a
multi-token-prediction head with the model and this checkpoint keeps it, so
the model drafts ahead and verifies the drafts in a single forward pass.
Speculative decoding is exact: the sampled distribution is the same as without
it, only the speed changes.

## Benchmarks

Realistic chat prompts (8 mixed English/Danish/code tasks in
[bench/prompts_real.jsonl](../bench/prompts_real.jsonl), 1,024-token answers),
`vllm bench serve --dataset-name custom`, RTX 3090 at 250 W:

**`CTX=fast` + fast variant (the default; 64k context)**, as reproduced by
`bash bench/run_benchmarks.sh single`:

| Cohort | e2e, model-default sampling (T 1.0, top-p 0.95, top-k 20) | decode | e2e, greedy | decode | tokens per step | mean TTFT |
|---|---|---|---|---|---|---|
| C1 | **111.1 tok/s** | 113.6 | **115.3 tok/s** | 118.3 | 2.82 / 2.88 | 172 ms |
| C2 | 173.1 tok/s | 194.0 | 175.9 tok/s | 194.9 | 2.87 / 2.83 | 278 ms |
| C4 | 220.9 tok/s | 258.6 | 256.2 tok/s | 292.6 | 2.71 / 2.98 | 341 ms |
| C8 | 309.2 tok/s | 379.9 | 321.9 tok/s | 404.0 | 2.71 / 2.79 | 1,006 ms |

Decode throughput is C × 1000 / mean TPOT; e2e includes prefill and the tail.
Per-position draft acceptance at C1: 74% / 50% / 34% / 24% (T 1.0), 77% / 55%
/ 40% / 30% (greedy). The best C1 repeats read 115 / 124 tok/s decode: greedy
generation is deterministic for a given server and request order, but a
different drafter config or a prefix-cache hit changes the text at near-ties
and with it the acceptance, so expect ±3-5% between runs. Quality of the fast
variant: perplexity 8.095 vs 8.045 for the base requantization (+0.6%,
en/da/code), GSM8K 96.5% (200 questions, greedy), same as the base.

**`CTX=long` (150k context, FlashInfer/fp8 KV, k=3)** with the fast variant:
95.3 / 100.3 tok/s at C1 (T default / greedy, 2.58 / 2.61 tokens per step).
With the base requantization and the earlier draft vocabulary, all cohorts:

| Cohort | e2e, model-default sampling | decode | e2e, greedy | decode | tokens per step | mean TTFT |
|---|---|---|---|---|---|---|
| C1 | 83.4 tok/s | 84.7 | 87.1 tok/s | 89.3 | 2.46 / 2.43 | 179 ms |
| C2 | 146.6 tok/s | 168.2 | 160.3 tok/s | 177.6 | 2.47 / 2.42 | 233 ms |
| C4 | 256.8 tok/s | 289.2 | 256.0 tok/s | 303.7 | 2.46 / 2.47 | 358 ms |
| C8 | 327.9 tok/s | 409.0 | 364.2 tok/s | 450.5 | 2.37 / 2.50 | 1,069 ms |

Four drafts win up to two concurrent users; from C4 up the three-draft config
is ahead (rejected drafts cost more when the verify batch is bigger), so for
a shared box `CTX=long` or `DRAFT_TOKENS=3` is the better single-user config.
Batch mode does 45-46 tok/s single-stream on the same prompts, and overtakes
this mode from C8 up.

The same server measured on the random-token protocol used by
[ninfer-3090](https://github.com/Don-Chad/ninfer-3090) (256 random tokens in,
1,024 out) reads anywhere from 35 to 151 tok/s depending on what the model
makes of the noise — this is why the tables above use real prompts. Their MTP3
number on that protocol is 70.19 tok/s at C1.

### How the draft got cheap

Every draft token is one pass through the MTP module plus a full lm_head
projection, k times per step. As shipped that is ~3 ms per draft on this card
(850 MB bf16 module + 1.3 GB int8 head), so MTP-2 was the sweet spot and MTP-3
already lost. Three changes, measured on the prompts above (T = default /
greedy):

| config | tok/s | tokens/step | acceptance pos 0 |
|---|---|---|---|
| no speculation (batch mode) | 46 / 46 | 1.0 | — |
| MTP-2 as shipped: bf16 module, full head, fp32 state | 66 / 79 | 2.1 / 2.4 | 65% / 80% |
| MTP-4: int8 module (`quant_mtp.py`), 40k-token draft head (`build_draft_vocab.py`), fp16 state | 78 / 99 | 2.2 / 2.7 | 58% / 70% |
| + `draft_sample_method: probabilistic` | 90 / 98 | 2.6 / 2.7 | 69% / 70% |
| same, k=3 (`CTX=long`) | 84 / 89 | 2.5 / 2.4 | 69% / 71% |
| same, k=6 | 76 / 94 | 2.3 / 2.7 | |
| k=4, full 248k head instead of 40k | 85 / 91 | 2.85 / 3.0 | 74% / 76% |
| k=4, bf16 `mtp.fc` (rest int8) | 88 / 96 | 2.6 / 2.6 | 67% / 70% |
| + sampler patch + split-KV verify attention | 93 / 99 | 2.6 / 2.6 | 69% / 70% |
| + draft vocabulary counted over the model's own outputs | 107 / 109 | 2.9 / 2.9 | 74% / 74% |
| + GPTQ-int4 lm_head, int4 draft head | 109 / 112 | 2.8 / 2.8 | 73% / 73% |
| **+ GPTQ-int4 MTP module (fast variant, shipped)** | **~114 / ~124** | 2.8 / 3.0 | 74% / 77% |
| same, k=5 | 106 / 105 | 3.0 / 2.9 | |
| same, 49k draft vocab | 109 / 115 | 2.7 / 2.8 | |
| same, `draft_sample_method: greedy` | 97 / 124 | 2.3 / 3.0 | |

The cheap drafter loses a few points of acceptance to int8/int4 (GPTQ with a
calibration set from the model's own hidden states loses none) and wins
because a step dropped from ~32 to ~24 ms while carrying more drafts. The
truncated vocabulary only costs acceptance if it's the wrong vocabulary: the
list counted over the model's own outputs (97.5% coverage of what it
generates) is the largest single step in the table; the earlier web-text list
(92%) had been silently capping acceptance at every position. Probabilistic
drafting samples the draft from the MTP distribution instead of taking its
argmax, which is what rejection sampling wants at temperature > 0; at greedy
it changes nothing. See [drafter/README.md](../drafter/README.md) for how the
vocabulary and the calibrated int4 tensors are made (and for the fine-tuning
attempt that did not help).

k=4 is the fastest but not the default: on the FlashInfer attention backend
(the only one that supports fp8 KV on Ampere, and fp8 KV is what makes 150k
context fit) vLLM 0.27.1 dies with an illegal memory access as soon as one
request finishes while another is mid-generation with 4 drafts (with or
without our patches; the vendored PR #50021 bounds fix does not cure it;
club-3090 sees the same "n=4 eventually dies, n=3 stable" on their rigs, and
vLLM has a family of open MTP illegal-memory-access reports on Qwen3.5/3.6,
e.g. [#40756](https://github.com/vllm-project/vllm/issues/40756),
[#36498](https://github.com/vllm-project/vllm/issues/36498)). The same k=4
config on the FlashAttention backend (bf16 KV) runs clean at C2/C4, so the
bug is in the FlashInfer spec-decode path. Hence two configs:

- `CTX=fast` (default): FlashAttention, bf16 KV, **~64k context**, k=4, split-KV
  verify attention → ~114 / ~124 tok/s with the fast variant
- `CTX=long`: FlashInfer, fp8 KV, **150k context**, k=3 → 95 / 100 tok/s with
  the fast variant (84 / 89 with the base requantization)

k=3 passed every concurrency soak we ran (C2/C4/C8 with staggered finishes,
100k-token prompt, 4×6k-token generations); if you see the crash anyway,
`DRAFT_TOKENS=2` costs ~5% and is the most conservative setting.

Why not 150? The verify pass alone reads ~13 GB of weights (~17 ms at what
this card actually delivers on 16-92 MB reads; see `marlin-tune/`) plus ~4 ms
of drafts and sampling, and Qwen's MTP head agrees with the target on ~75-77%
of first drafts on real text once it can propose the right tokens, so ~3
accepted tokens per ~24 ms step is where a single-layer chain drafter tops
out. Random-token benchmarks that show 150+ are measuring how repetitive
noise is. Fine-tuning the head on the model's own outputs did not raise its
top-1 agreement (`drafter/README.md`); a tree drafter would, but the
DeltaNet layers can't verify a tree.

## Setup

Do the [common setup](../README.md#setup) first (venv, model download,
requantization, draft head, the fast variant via `fetch_fast_variant.py`, vLLM
patches; `bash verify.sh --no-server` checks all of it). Then:

```bash
bash single-user/start_qwen.sh
bash bench/run_benchmarks.sh single    # reproduces the tables above
```

Or as a service:

```bash
mkdir -p ~/.config/systemd/user
cp single-user/qwen-serving.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now qwen-serving
loginctl enable-linger $USER
```

Point your chat client at `http://<host>:18020/v1` with the key from
`api_key.txt`. Works with anything that speaks the OpenAI API.

## Knobs

| var | default | notes |
|---|---|---|
| `MODEL` | `models/Qwen3.8-27B-W4A16-AutoRound-fast` if present, else the base dir | the fast variant (`fetch_fast_variant.py`) is +15% |
| `CTX` | `fast` | `fast`: bf16 KV / FlashAttention / 64k / 4 drafts / split-KV attention. `long`: fp8 KV / FlashInfer / 150k / 3 drafts, ~15% slower at C1, faster from C4 up. `huge`: KVarN 4/2-bit KV / 200k / 3 drafts (needs `bash kvarn/install.sh`; main README "262k context") |
| `DRAFT_TOKENS` | 4 (3 for `CTX=long`/`huge`) | speculative depth; 5 and 6 are slower |
| `SPEC_ATTN` | 1 (`CTX=fast` only) | split-KV Triton attention for the verify step (`patches/spec-decode-attn.patch`); 0 = FlashAttention-2 |
| `DRAFT_SAMPLE` | `probabilistic` | `greedy` drafts: same speed at T=0, ~15% slower at T>0 |
| `MAX_SEQS` | 8 | plenty for a few users; each request holds k+1 recurrent-state slots |
| `MAX_LEN` | 65536 (`fast`) / 150000 (`long`) | 150k needs `GPU_UTIL` 0.93 |
| `GPU_UTIL` | 0.93 | soak-tested with a 100k prompt and 4×6k-token generations; batch mode's 0.972 OOMs in the MTP path (main README, gotcha 4) |
| `MTP_DRAFT_VOCAB` | 1 | set 0 to draft with the full lm_head (more acceptance, slower per draft) |
| `PORT` | 18020 | |

## Switching modes

Only one mode can run at a time (one GPU). Swap by replacing the unit file:

```bash
systemctl --user stop qwen-serving
cp batch/qwen-serving.service ~/.config/systemd/user/   # or single-user/
systemctl --user daemon-reload
systemctl --user start qwen-serving
```
