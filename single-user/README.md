# Single-user mode

For one person (or a handful) chatting with the model: coding assistant,
local chat UI, anything where you're watching tokens stream in.

The difference from batch mode is MTP speculative decoding. Qwen ships a
multi-token-prediction head with the model and this checkpoint keeps it, so
the model drafts ahead and verifies the draft in a single forward pass.

## Benchmarks

Cohort protocol (same as ninfer-3090's published tables): C concurrent
requests, 256-token prompts, 1,024 output tokens each. `vllm bench serve`,
RTX 3090 at 250 W:

| Cohort | Total output | End-to-end throughput | Decode throughput* | MTP acceptance | Mean TTFT | Peak VRAM |
|---|---|---|---|---|---|---|
| C1 | 1,024 tokens | 81.85 tok/s | 83.47 tok/s | 81.8% | 260 ms | 23,035 MiB |
| C2 | 2,048 tokens | 119.38 tok/s | 136.99 tok/s | 59.6% | 657 ms | 23,035 MiB |
| C4 | 4,096 tokens | 243.62 tok/s | 281.69 tok/s | 79.4% | 943 ms | 23,035 MiB |
| C8 | 8,192 tokens | 212.75 tok/s | 375.76 tok/s | 62.1% | 1,825 ms | 23,035 MiB |

*Decode throughput derived as C × 1000 / mean TPOT. Acceptance is per drafted
token and swings with the sampled content, hence the C2/C8 dips. Peak VRAM is
flat because vLLM preallocates its pool at startup.

Same protocol on ninfer-3090's published numbers, same model, same card:

| Cohort | ninfer e2e | this repo e2e | ninfer decode | this repo decode |
|---|---|---|---|---|
| C1 | 70.19 | **81.85** (+17%) | 71.00 | **83.47** |
| C2 | 89.43 | **119.38** (+33%) | 90.66 | **136.99** |
| C4 | 97.89 | **243.62** (+149%) | 100.28 | **281.69** |
| C8 | 161.28 | **212.75** (+32%) | 165.33 | **375.76** |

(And from C8 up you'd switch to [batch mode](../batch/) anyway, which does
289.6 tok/s e2e on this same C8 protocol.)

### Choosing a speculation method

Earlier shootout with shorter outputs (256/256), including the community
[DSpark](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark) draft model for
this exact target:

| config | 1 concurrent | median TPOT | 8 concurrent | draft acceptance | extra VRAM |
|---|---|---|---|---|---|
| no speculation (batch mode) | 45.0 tok/s | 21.4 ms | 249.7 tok/s | — | — |
| **MTP, 2 drafts (this mode)** | **63.2 tok/s** | **14.9 ms** | 240.6 tok/s | 57% | none |
| MTP, 3 drafts | 56.1 tok/s | 14.0 ms | 242.0 tok/s | 46% | none |
| DSpark, 7 drafts | 61.3 tok/s | 17.9 ms | 108.6 tok/s | 16% | 2.7 GB, ctx capped ~62k |
| DSpark, 3 drafts | 56.5 tok/s | 16.6 ms | 177.6 tok/s | 30% | 2.7 GB, ctx capped ~62k |

MTP-2 wins: +40% single-stream over no speculation, full 150k context kept,
zero extra memory, and speculative decoding is exact (the output distribution
is identical to normal decoding). The DSpark draft head sounds appealing on
paper (block-7 drafts) but its acceptance collapses against this int4 target
and it burns 2.7 GB that this architecture would rather spend on cache pages.
To try it anyway: point `--speculative-config` at the DSpark model with
`{"method":"dspark",...}` and set the arch in its config.json to
`Qwen3DSparkModel` so vLLM dispatches the right implementation.

Don't put this mode behind a busy API: the no-speculation column wins from
roughly 8 concurrent users up, and batch mode scales to 64.

## Setup

Do the [common setup](../README.md#setup) first (venv, model download,
requantization, vLLM patch — the requant step also fixes the quant config for
the MTP head, so don't skip it). Then:

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
| `DRAFT_TOKENS` | 2 | speculative depth. 3 measured slightly worse on general text (see table); might pay off on repetitive/code-heavy prompts |
| `MAX_SEQS` | 8 | plenty for a few users; keeps state-slot reservations small |
| `MAX_LEN` | 150000 | |
| `GPU_UTIL` | 0.90 | deliberately lower than batch mode's 0.972 — the MTP decode path OOMs on long generations above this (main README, gotcha 3). Don't raise |
| `PORT` | 18020 | |

## Switching modes

Only one mode can run at a time (one GPU). Swap by replacing the unit file:

```bash
systemctl --user stop qwen-serving
cp batch/qwen-serving.service ~/.config/systemd/user/   # or single-user/
systemctl --user daemon-reload
systemctl --user start qwen-serving
```
