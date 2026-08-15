# Single-user mode

For one person (or a handful) chatting with the model: coding assistant,
local chat UI, anything where you're watching tokens stream in.

The difference from batch mode is MTP speculative decoding. Qwen ships a
multi-token-prediction head with the model and this checkpoint keeps it, so
the model drafts ahead and verifies the draft in a single forward pass.

Measured on an RTX 3090:

- ~25 ms per token (~40 tok/s) streaming to one user, vs ~60 ms in batch mode
- same 150k context, same quality (speculative decoding is exact — the output
  distribution is identical to normal decoding)

Don't put this config behind a busy API: at high concurrency the drafting
overhead wins and aggregate throughput drops to ~145 tok/s, roughly a third of
batch mode.

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
| `DRAFT_TOKENS` | 2 | speculative depth. Try 3 for code-heavy use — acceptance is higher there, so deeper drafts pay off. Beyond 3 hasn't helped us |
| `MAX_SEQS` | 8 | plenty for a few users; keeps state-slot reservations small |
| `MAX_LEN` | 150000 | |
| `PORT` | 18020 | |

## Switching modes

Only one mode can run at a time (one GPU). Swap by replacing the unit file:

```bash
systemctl --user stop qwen-serving
cp batch/qwen-serving.service ~/.config/systemd/user/   # or single-user/
systemctl --user daemon-reload
systemctl --user start qwen-serving
```
