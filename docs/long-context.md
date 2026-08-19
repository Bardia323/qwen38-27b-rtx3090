# Long context: 262k with KVarN, and the built-in KV modes

How to get past the ~200k the fp8 KV cache allows, and what vLLM's own per-token-head quantization modes are worth on this card.

[← back to the main README](../README.md)

## 262k context with KVarN

With fp8 KV the pool holds ~200k tokens, so 150k is the shipped max and ~195k
the ceiling; the model's full 262,144 is out of reach because 16 attention
layers × 4 KV heads × 256 dims × 2 bytes (K+V, fp8) is 2 KB per token. The
way past that is a smaller cache, not a different engine, and
[KVarN](https://github.com/huawei-csl/KVarN) (Huawei CSL) has the best one we
know of: Hadamard rotation + iterative variance normalization + 4-bit keys /
2-bit values per 128-token tile, at ~840 B/token/layer here. It ships as a
fork of vLLM 0.23; [kvarn/](../kvarn/) is our port of its dense backend onto the
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
are in [kvarn/README.md](../kvarn/README.md).

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
