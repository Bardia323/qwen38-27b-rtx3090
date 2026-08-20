# Gotchas

Things that each cost us hours, in rough order of pain. Worth skimming before you debug something that looks like a vLLM bug.

[← back to the main README](../README.md)

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
    KV only) is a 180-line Triton fix. Watch its query cap: the kernel used to
    handle at most `BLOCK_M / (heads per kv head)` = 10 query tokens and fall
    back silently past that, which doubled the step at 25k context the moment
    the verify block grew to 16. It now tiles the query rows instead.
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
16. **The very first start gets a smaller KV pool.** vLLM sizes the pool from
    the peak memory of a profiling forward pass, and on a cold torch.compile
    cache that pass also runs inductor's autotuning: batch mode profiles a
    1.96 GiB activation peak instead of 1.09 GiB and comes up with 196k KV
    tokens instead of 224k (`Maximum concurrency ... 1.31x` in the log instead
    of 1.49x). Restart once after the cache is warm (venv: `~/.cache/vllm`,
    Docker: the `qwen-cache` volume) and the pool is back to the README numbers.
    (The WSL2 notes above pin it the other way round — record the cold-start
    `--kv-cache-memory` recommendation and pass it via `EXTRA_ARGS` — if you
    prefer the extra transient headroom to the extra KV pages.)
17. **vLLM picks the speculative method from the model *path*.** `"dflash" in
    model_path` switches `method` to dflash — for the *target* too, since MTP
    uses the target path as its draft model. A checkout under a directory with
    "dflash" in its name turns `SPEC=mtp` into a crash in `EAGLEConfig`
    (`'Qwen3_5Config' object has no attribute 'vocab_size'`). Name your
    directories accordingly.
18. **The V2 model runner (`SPEC=dflash2`) does not count its CUDA graphs when
    sizing the KV pool** (~1.2 GiB on top of whatever `--gpu-memory-utilization`
    you asked for), the hybrid allocator sizes KV groups by the smallest layer
    bucket, and the profiled activation peak varies by ~1 GiB between starts of
    the *same* config — three ways to get a server that either wastes a quarter
    of its pool or dies mid-request. `patches/hybrid-kv-groups-v2-cudagraph.patch`
    fixes the first two; for the third, pin the pool in bytes
    (`--kv-cache-memory`, what `KV_MEM` does) instead of tuning utilization. That
    runner also answers `thinking_token_budget` with 400, and the first request
    after a cold start JIT-compiles four Triton kernels (~5 s once; cached in
    `~/.triton`).
19. **`INT8_LAYERS=.` needs `GPU_UTIL=0.95`.** Quantizing the activations of every linear
    layer (rather than just the MLP) is worth ~11% throughput — 1,042 vs 942 tok/s at 64
    concurrent — but the extra per-layer scratch no longer fits batch mode's 0.972: the
    engine dies with `torch.OutOfMemoryError` inside `chunk_fwd_o` once ~17 requests are
    resident, which reads as every request returning 500 while `/health` still answers.
20. **A Triton kernel's scratch buffers may not grow after CUDA graph capture.** The
    split-KV verify attention sizes its partial buffers from the longest query block it has
    been asked for. Once the block got longer than the drafter's — and once a small prefill
    chunk could land on the same kernel — that "longest so far" changed mid-run, the buffers
    were reallocated, and the captured decode graph went on reading the freed ones:
    `CUDA error: an illegal memory access was encountered`, a few hundred tokens into the
    first request. `VLLM_SPEC_DECODE_ATTN_QMAX` (set by `single-user/start_qwen.sh` from
    `DFLASH_TOKENS`) fixes the size at startup instead.
21. **Async scheduling pins the number of speculative tokens.** vLLM only feeds draft token
    ids — and therefore the *count* the worker wants verified — back to the scheduler on the
    synchronous path (`EngineCore.post_step`). With async scheduling on, every decode step is
    padded to `num_speculative_tokens` and a worker asking for fewer is ignored, silently.
    Adaptive block length (`LOOKUP=1` with `DFLASH_TOKENS > 7`) needs `ASYNC_SCHED=0`; at
    batch 1 that costs under 1%.
22. **`--async-scheduling` is already the default in 0.27.1.** The flag exists and passing it
    changes nothing; `--no-async-scheduling` is what turns it off. Two hours of "the adaptive
    block isn't working" was this.
23. **A longer verify block costs KV pool per request slot, not per token.**
    `--mamba-cache-mode align` reserves `2 + num_speculative_blocks` recurrent-state pages
    per slot, so `DFLASH_TOKENS=31` with 8 slots wants 5.3 GiB before a single token of
    context and refuses to start. Single-user mode drops to 4 slots when the block is long,
    which is what makes the long block affordable at all.
