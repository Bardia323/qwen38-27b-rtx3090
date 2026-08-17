# Batch mode

For serving many concurrent requests: API backends, data processing pipelines,
eval runs. Tuned for aggregate tokens per second, not per-request latency.

## Benchmarks

`vllm bench serve`, random dataset, 256 requests, RTX 3090 at 250 W, default
config (fp16 recurrent state, int8 activations on the MLP GEMMs):

| workload | output tok/s (e2e) | steady-state decode | median TPOT | median TTFT |
|---|---|---|---|---|
| 256/256, 1 concurrent | 46 | 46 | 21.7 ms | 230 ms |
| 128/512, 64 concurrent | **876** | ~1,050 | 61.3 ms | 3.7 s* |
| 256/256, 64 concurrent | 642 | ~1,050 | 80.5 ms | 4.2 s* |

*TTFT at saturation is queue time — the bench fires all requests at once.

"Steady-state decode" is what the engine logs as generation throughput once
all 64 requests are resident and no prefill is interleaved; the e2e number
includes everybody's prefill and the ramp. Older configs, same protocol:

| config | e2e 128/512 | e2e 256/256 | steady decode |
|---|---|---|---|
| W4A16, fp32 state (only 37 of 64 requests could run) | 516 | 393 | ~585 |
| W4A16, fp16 state | 707 | 491 | ~830 |
| int8 activations, gate/up only (`INT8_LAYERS=gate_up`) | 787 | 572 | ~930 |
| int8 activations, MLP (default) | 876 | 642 | ~1,050 |
| int8 activations, all linears (`INT8_LAYERS=.`) | 1,025 | 752 | ~1,150 |

Quality of each row is in the [main README](../README.md#quality).

Cohort protocol on realistic prompts (C concurrent chat requests, 1,024-token
answers, model-default sampling; comparable to the
[single-user tables](../single-user/README.md)):

| Cohort | e2e throughput | decode throughput | mean TTFT |
|---|---|---|---|
| C1 | 45.4 tok/s | 45.6 tok/s | 110 ms |
| C2 | 81.8 tok/s | 83.7 tok/s | 197 ms |
| C4 | 153.8 tok/s | 162.7 tok/s | 343 ms |
| C8 | 298.4 tok/s | 321.0 tok/s | 638 ms |

Below ~C8, single-user mode's speculative decoding is faster; batch mode pulls
ahead from C8 and keeps scaling to C64.

### Prefill

Measured with 1-token outputs so it's pure prompt processing. W4A16 numbers
(single-user mode, and batch mode with `INT8_ACT=` unset); the int8 tensor-core
path is faster still (~+40% at 1k, e.g. 4.2 s TTFT for 64×256 tokens vs 5.6 s):

| input length | conc 1 | conc 4 | conc 8-16 | single-request TTFT |
|---|---|---|---|---|
| 1k | 1,210 tok/s | 1,215 | 1,213 | 0.85 s |
| 4k | 1,185 | 1,185 | 1,183 | 3.5 s |
| 16k | 1,112 | 1,117 | 1,116 | 14.7 s |
| 64k | 906 | 908 | — | 72 s |
| 100k | 795 | — | — | 129 s |

Two things to plan capacity around. First, concurrency does nothing for
prefill: chunked prefill feeds everything through the same 2,048-token
per-step budget, so prompt processing is a fixed resource the whole server
shares, and queueing is linear (four 16k prompts at once means the last one
waits ~48 s). Second, the falloff with length is mild — only ~34% from 1k to
100k — because just 16 of 64 layers pay quadratic attention; this is one of
the places the hybrid architecture genuinely helps.

## Setup

Do the [common setup](../README.md#setup) first (venv, model download,
requantization, vLLM patches). Then:

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
| `INT8_ACT` | `int8` | int8 activations on the Marlin GEMMs (int8 tensor cores, weights stay int4). Empty string = plain W4A16 |
| `INT8_LAYERS` | `mlp` | regex on the layer name that gets int8 activations. `gate_up` for the gentle variant, `.` for everything, or a hand-picked list from `bench/act_calib.py` |
| `MAX_SEQS` | 64 | scheduler slots; with fp16 state ~70 short requests fit the page pool |
| `MAX_LEN` | 150000 | max context. Raising it much past this fails startup, the pool can't hold a longer request |
| `PORT` | 18020 | |
| `GPU_UTIL` | 0.972 | do not raise, see gotchas in the main README. Use 0.93 when you want `prompt_logprobs` (quality checks) |

## Verify you're getting the numbers

```bash
OPENAI_API_KEY=$(cat api_key.txt) venv/bin/vllm bench serve \
  --host 127.0.0.1 --port 18020 \
  --model models/Qwen3.8-27B-W4A16-AutoRound \
  --served-model-name qwen3.8-27b \
  --dataset-name random --random-input-len 128 --random-output-len 512 \
  --num-prompts 256 --max-concurrency 64
```

Run it twice, keep the second number. The first run after a restart includes
JIT warmup and reads 30-50% low. Then run `bench/quality_battery.py` — a
throughput number from a server that emits garbage is worth nothing, and the
int8 path taught us that the hard way.
