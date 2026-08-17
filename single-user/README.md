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

| Cohort | e2e, model-default sampling (T 1.0, top-p 0.95) | decode | e2e, greedy | decode | tokens per step | mean TTFT |
|---|---|---|---|---|---|---|
| C1 | **83.4 tok/s** | 84.7 | 87.1 tok/s | 89.3 | 2.46 / 2.43 | 179 ms |
| C2 | 146.6 tok/s | 168.2 | 160.3 tok/s | 177.6 | 2.47 / 2.42 | 233 ms |
| C4 | 256.8 tok/s | 289.2 | 256.0 tok/s | 303.7 | 2.46 / 2.47 | 358 ms |
| C8 | 327.9 tok/s | 409.0 | 364.2 tok/s | 450.5 | 2.37 / 2.50 | 1,069 ms |

Decode throughput is C × 1000 / mean TPOT; e2e includes prefill and the tail.
Per-position draft acceptance at C1: 69% / 45% / 29% (T 1.0), 71% / 47% / 31%
(greedy). Batch mode does 45-46 tok/s single-stream on the same prompts, and
overtakes this mode from C8 up.

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
| **same, k=3 (shipped)** | **84 / 89** | 2.5 / 2.4 | 69% / 71% |
| same, k=6 | 76 / 94 | 2.3 / 2.7 | |
| k=4, full 248k head instead of 40k | 85 / 91 | 2.85 / 3.0 | 74% / 76% |
| k=4, bf16 `mtp.fc` (rest int8) | 88 / 96 | 2.6 / 2.6 | 67% / 70% |

The cheap drafter loses ~10 points of acceptance (int8 module ~4, truncated
vocabulary ~6) and wins anyway because a step dropped from ~32 to ~27 ms while
carrying more drafts. Probabilistic drafting samples the draft from the MTP
distribution instead of taking its argmax, which is what rejection sampling
wants at temperature > 0; at greedy it changes nothing.

k=4 is the fastest but not shipped: on vLLM 0.27.1 the engine dies with an
illegal memory access in the DeltaNet spec-decode path as soon as one request
finishes while another is mid-generation (any k=4 config, with or without
our patches; the vendored PR #50021 bounds fix does not cure it; club-3090
sees the same "n=4 eventually dies, n=3 stable" on their rigs). k=3 passed
every concurrency soak we ran (C2/C4/C8 with staggered finishes, 100k-token
prompt, 4×6k-token generations).

Why not 150? The verify pass alone reads 13.9 GB of weights (~21 ms at this
card's bandwidth) and the shipped MTP head agrees with the target on only ~70%
of first drafts on real text, so ~2.7 accepted tokens per ~27 ms step is where
this drafter tops out. Random-token benchmarks that show 150+ are measuring
how repetitive noise is. A stronger drafter (DFlash/EAGLE-style, trained for
this model) is the next lever, not engine work.

## Setup

Do the [common setup](../README.md#setup) first (venv, model download,
requantization, draft head, vLLM patches). Then:

```bash
bash single-user/start_qwen.sh
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
| `DRAFT_TOKENS` | 3 | speculative depth; 4 is ~7% faster single-stream but crashes when requests churn (see above), 6 is slower |
| `MAX_SEQS` | 8 | plenty for a few users; each request holds k+1 recurrent-state slots |
| `MAX_LEN` | 150000 | 150k needs `GPU_UTIL` 0.93 |
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
