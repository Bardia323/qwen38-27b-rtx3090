# Batch mode

For serving many concurrent requests: API backends, data processing pipelines,
eval runs. Tuned for aggregate tokens per second, not per-request latency.

## Benchmarks

`vllm bench serve`, random dataset, RTX 3090 at 250 W:

| workload | output tok/s | total tok/s (in+out) | median TPOT | median TTFT |
|---|---|---|---|---|
| 256/256, 1 concurrent | 45.0 | 90 | 21.4 ms | 229 ms |
| 256/256, 8 concurrent | 249.7 | 499 | 25.4 ms | — |
| 256/256, 64 concurrent | 417.5 | 835 | 65.4 ms | 22.3 s* |
| 2048/256, 32 concurrent | 115.0 | 1034.6 | 203.7 ms | 15.6 s* |
| 8192/512, 16 concurrent | 59.1 | 1004.3 | 209.1 ms | 24.8 s* |

*TTFT at saturation is queue time — the bench fires all requests at once.
Under real traffic at lower utilization it's sub-second.

Cohort protocol (C requests, 1,024 output tokens each — comparable to the
[single-user tables](../single-user/README.md)):

| Cohort | Total output | End-to-end throughput | Decode throughput | Mean TTFT | Peak VRAM |
|---|---|---|---|---|---|
| C1 | 1,024 tokens | 45.68 tok/s | 46.08 tok/s | 223 ms | 23,961 MiB |
| C2 | 2,048 tokens | 82.44 tok/s | 83.75 tok/s | 406 ms | 23,961 MiB |
| C4 | 4,096 tokens | 155.13 tok/s | 159.87 tok/s | 806 ms | 23,961 MiB |
| C8 | 8,192 tokens | 289.60 tok/s | 307.81 tok/s | 1,693 ms | 23,961 MiB |

Below ~C8, single-user mode's speculative decoding is faster; batch mode pulls
ahead from C8 and keeps scaling to C64.

Two readings from the table: long-input workloads are prefill-bound (the
machine sustains ~1000 total tok/s of prompt processing no matter the shape),
and single-stream on an idle server is a respectable 45 tok/s even without
speculation. If sub-20 ms tokens for one user is the goal, use
[single-user mode](../single-user/) instead.

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
