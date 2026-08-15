# Batch mode

For serving many concurrent requests: API backends, data processing pipelines,
eval runs. Tuned for aggregate tokens per second, not per-request latency.

Measured on an RTX 3090 (256 in / 256 out, 64 concurrent):

- 416 tok/s sustained aggregate, 672 tok/s peak
- 66 ms median time per output token
- ~1000 tok/s total (prompt + completion) on input-heavy workloads

The tradeoff: a single request on an idle server decodes at ~17 tok/s. If
that's your main use case, use [single-user mode](../single-user/) instead.

## Setup

Do the [common setup](../README.md#setup) first (venv, model download,
requantization, vLLM patch). Then:

```bash
bash batch/start_qwen.sh
```

Or as a service:

```bash
mkdir -p ~/.config/systemd/user
cp batch/qwen-serving.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now qwen-serving
loginctl enable-linger $USER
```

First start takes a few minutes (torch.compile, CUDA graph capture, flashinfer
JIT). Watch `qwen.log` in the repo root; it's up when you see
"HTTP server started".

## Knobs

All overridable as env vars, defaults in the script:

| var | default | notes |
|---|---|---|
| `MAX_SEQS` | 64 | scheduler slots. Actual running batch is bounded by the cache page pool (~30 for 512-token requests), so raising this further does nothing |
| `MAX_LEN` | 150000 | max context. Lower it if you want a bigger safety margin; raising it much past this fails startup, the pool can't hold a longer request |
| `PORT` | 18020 | |
| `GPU_UTIL` | 0.972 | do not raise, see gotchas in the main README |

## Verify you're getting the numbers

```bash
OPENAI_API_KEY=$(cat api_key.txt) venv/bin/vllm bench serve \
  --host 127.0.0.1 --port 18020 \
  --model models/Qwen3.8-27B-W4A16-AutoRound \
  --served-model-name qwen3.8-27b \
  --dataset-name random --random-input-len 256 --random-output-len 256 \
  --num-prompts 256 --max-concurrency 64
```

Run it twice, keep the second number. The first run after a restart includes
JIT warmup and reads 30-50% low.
