# marlin-tune — a tunable standalone build of vLLM 0.27.1's GPTQ-Marlin for sm86

Compiles a modified copy of vLLM 0.27.1's Marlin CUDA kernel as a standalone
torch extension so its small-batch (M ≤ 16) tile configuration can be tuned for
the RTX 3090. Nothing is installed into, or modified inside, the vLLM venv —
the extension registers its own torch library `_C_marlin_tune` and can be
loaded side by side with vLLM's `_C`.

Everything lives in `~/marlin-tune/src` on `syv`.

```
~/marlin-tune/
  vllm-src/            sparse git checkout of vllm v0.27.1 (reference only)
  src/
    csrc.pristine/     unmodified copy of the marlin sources
    csrc/              patched copy that actually gets built
    gen_kernels.py     replacement for vLLM's generate_kernels.py
    patch_marlin.py    applies our changes to csrc/ (idempotent-checked)
    binding.cpp        pybind bindings for the tuning knobs
    setup.py, build.sh
    marlin_common.py   shared test/bench helpers
    test_correctness.py
    bench_marlin.py
    marlin_best.py     per-(N,K) best-config table + selector
```

---

## 1. CUDA 13 toolchain (no root, no venv changes)

The system `nvcc` is 12.0, too old for torch 2.13+cu130 extensions.

```bash
python3 -m venv ~/cuda13-env
~/cuda13-env/bin/pip install --upgrade pip
~/cuda13-env/bin/pip install \
    nvidia-cuda-nvcc==13.0.88   nvidia-cuda-runtime==13.0.96 \
    nvidia-cuda-cccl==13.0.85   nvidia-cuda-crt==13.0.88 \
    nvidia-cuda-nvrtc==13.0.88  nvidia-cuda-profiler-api==13.0.85 \
    nvidia-nvvm==13.0.88

# verify
C=~/cuda13-env/lib/python3.12/site-packages/nvidia/cu13
$C/bin/nvcc --version            # -> release 13.0, V13.0.88
printf '#include <cstdio>\n__global__ void k(){printf("hi %%d\\n",blockIdx.x);}\nint main(){k<<<2,1>>>();cudaDeviceSynchronize();}\n' > /tmp/h.cu
$C/bin/nvcc -arch=sm_86 -I$C/include -L$C/lib /tmp/h.cu -o /tmp/h -lcudart \
    -Xlinker -rpath=$C/lib && /tmp/h
```

Three traps, all of which cost a build cycle:

1. **`nvidia-*-cu13` package names are deprecated stubs.** Their sdists
   deliberately fail to build with "THIS PROJECT IS DEPRECATED". The
   *unsuffixed* names (`nvidia-cuda-nvcc`, …) are the CUDA 13 wheels.
2. **Pin to 13.0.x.** `torch.version.cuda` is `13.0`; a 13.3 nvcc rejects the
   13.0 CCCL headers with
   `"CUDA compiler and CUDA toolkit headers are incompatible"`.
3. **Pin `nvidia-nvvm` too.** It is only a transitive dependency, so pip
   happily resolves it to 13.3 even when everything else is 13.0. Its `cicc`
   then emits PTX ISA 9.3 and the 13.0 `ptxas` fails with
   `Unsupported .version 9.3; current version is '9.0'`.

Note: the torch cu130 wheel ships its own `site-packages/nvidia/cu13`, but it
is internally **mismatched** (nvcc 13.3 on top of 13.0 headers), so it cannot
be used as `CUDA_HOME` as-is — it hits trap 2.

`build.sh` builds a symlink shim at `src/cuda13-home/` with
`bin/ include/ lib64/ nvvm/` because `torch.utils.cpp_extension` expects
`lib64`, while the pip layout uses `lib`. The venv is never written to.

## 2. Source

```bash
mkdir -p ~/marlin-tune && cd ~/marlin-tune
git clone --filter=blob:none --no-checkout --depth 1 --branch v0.27.1 \
    https://github.com/vllm-project/vllm.git vllm-src
cd vllm-src && git sparse-checkout init --cone \
  && git sparse-checkout set csrc cmake && git checkout
```

In 0.27.1 the kernel lives at `csrc/libtorch_stable/quantization/marlin/` and
is built against the **torch stable ABI** (`torch::stable::Tensor`), not ATen.
The op is registered as `torch.ops._C.marlin_gemm` — *not* `gptq_marlin_gemm`,
which was the pre-0.27 name. Files copied into `src/csrc/`:

```
core/scalar_type.hpp
libtorch_stable/torch_utils.h
libtorch_stable/core/math.hpp
libtorch_stable/quantization/marlin/{marlin.cu,marlin.cuh,marlin_dtypes.cuh,
    marlin_template.h,marlin_mma.h,dequant.h,kernel.h}
```

## 3. Build

```bash
cd ~/marlin-tune/src
./build.sh                 # full tuning grid, ~20 s on 12 cores
./build.sh --stock-only    # only the 4 stock tiles at stages=4
```

Required nvcc flags that are easy to miss:

* `-DUSE_CUDA` — `torch_utils.h` calls `aoti_torch_get_current_cuda_stream`,
  which `aoti_torch/c/shim.h` only declares inside `#ifdef USE_CUDA`.
* `-static-global-template-stub=false` — CUDA 13 defaults this to `true`,
  which gives explicit `__global__` template instantiations *internal*
  linkage. `marlin.cu` takes their address from a different TU, so the link
  fails with `relocation ... against undefined hidden symbol`. vLLM's own
  CMakeLists sets the same flag.
* `-DMARLIN_NAMESPACE_NAME=marlin_tune` — puts every symbol in its own
  namespace so the extension cannot collide with the vLLM `_C.so` loaded in
  the same process.
* No `--use_fast_math`: vLLM applies it only to the Kimi-K3 kernels, not to
  Marlin, so we don't either.

`torch_utils.h` also `#include <cublas_v2.h>` for a handle helper Marlin never
uses; the pip CUDA wheels don't ship cuBLAS headers, so `patch_marlin.py`
strips both.

## 4. What was changed relative to stock

`patch_marlin.py` makes exactly six edits, each guarded by a
"must match exactly once" check so a future vLLM version fails loudly instead
of being silently mis-patched:

1. `namespace marlin` → `namespace MARLIN_NAMESPACE_NAME`.
2. Adds a process-global `MarlinTuneConfig` + `marlin_tune_set_config()` /
   `marlin_tune_get_last_config()`.
3. Applies the overrides inside `marlin_mm()`:
   `thread_k`, `thread_n`, `threads`, `stages`, `blocks_per_sm`, `max_par`,
   `smem_mode`, `force_m_block_size_8`.
4. Checks `cudaGetLastError()` after the kernel launch. Stock never does, so a
   config that exceeds the per-block register/thread limit fails **silently**
   and leaves `C` untouched — which benchmarks as an impossibly fast kernel.
5. Sizes `C_tmp` for `sms * blocks_per_sm` blocks instead of `sms` (see 6.3).
6. Registers into its own `STABLE_TORCH_LIBRARY(_C_marlin_tune)`.

`gen_kernels.py` replaces vLLM's generator. 300 instantiations, ~28 s to build
(the stock generator emits thousands and takes tens of minutes). What is
emitted:

| set | a_type | b_type | group_blocks | thread_m_blocks | tiles × stages |
|---|---|---|---|---|---|
| tuning grid | bf16 | uint4b8, uint8b128 | 8 | 0.5, 1 | 10 tiles × stages 2-6 |
| drop-in | bf16 | uint4b8, uint8b128 | -1, 2, 4, 8 | 0.5, 1, 2, 3, 4 | 4 stock tiles, stages 4 |
| W4A8 | **int8** | uint4b8 | -1, 2, 4, 8 | 1, 2, 3, 4 | 4 stock tiles, stages 4 |

The four stock tiles are the union of `small_batch_thread_configs` and
`large_batch_thread_configs` from `marlin.cu`, so **prefill resolves to exactly
the kernel stock would have picked** — verified bit-identical up to M=512.
The W4A8 set is the `VLLM_MARLIN_INPUT_DTYPE=int8` path (int8 activations are
only supported against 4-bit weights: *"W8A8-INT8 is not supported by marlin
kernel"*).

Not emitted, so not a drop-in for these: fp16 activations, fp8 activations
(sm89+ only), AWQ `kU4` / zero-points, nvfp4 / mxfp4 / fp8 weights, act-order
with `is_k_full=False` (`group_blocks == 0`), and `is_zp_float`. Those raise
"Unsupported shapes" rather than computing anything wrong.

### API

The per-shape choice happens **inside the op**, driven by a table installed
once at import. Nothing branches on `size_m` in Python — under `torch.compile`
the M dimension is dynamic, so a Python `if size_m <= 16` would add guards or a
graph break, and CUDA graph replay does not run Python at all.

```python
import sys; sys.path.insert(0, "<repo>/marlin-tune")
import marlin_tune            # loads the ext, registers the fake, installs the table

out = torch.ops._C_marlin_tune.marlin_gemm(   # identical signature to _C
    a, c_or_none, b_q_weight, b_bias_or_none, b_scales, a_scales,
    global_scale, b_zeros_or_none, g_idx_or_none, perm_or_none, workspace,
    b_type.id, size_m, size_n, size_k, is_k_full, use_atomic_add,
    use_fp32_reduce, is_zp_float)
```

Table rows are 10 ints:

```
size_n, size_k, num_bits, m_max, thread_k, thread_n, threads, stages,
blocks_per_sm, smem_mode
```

`m_max` is the largest `size_m` the row covers; the op picks the matching row
with the smallest `m_max`, so an M≤8 row and an M≤16 row can coexist for one
shape. `marlin_best.table()` builds the rows from `marlin_best.BEST`.

```python
import marlin_tune_ext
marlin_tune_ext.set_table(rows)          # install (once)
marlin_tune_ext.get_table_size()
marlin_tune_ext.lookup(n, k, num_bits, m, a_bits=16)  # what the table would pick
marlin_tune_ext.get_last_config()        # what the last launch actually used
marlin_tune_ext.set_config(...)          # global override, benchmarking only
marlin_tune_ext.reset_config()
```

Two guards in the C++ lookup:

* **16-bit activations only.** The W4A8 path halves the k-slice per pipeline
  stage, which changes the tile constraints; it was never tuned. Without this
  guard a `(16384, 5120, 4)` row tuned for bf16 would be applied to W4A8 and
  return wrong results — that is exactly what the integration test caught.
* **`blocks_per_sm` is clamped by the caller's workspace.** The kernel indexes
  `locks[]` by block; vLLM's `marlin_make_workspace_new(device)` allocates only
  `sms` entries. A caller that did not widen it degrades to 1 block/SM
  (correct, just not faster) instead of writing past the buffer.

### Dropping it into vLLM

`vllm_hook.py` — import it before the model loads; it is a no-op unless
`VLLM_MARLIN_TUNE=1`:

```python
import sys; sys.path.insert(0, "<repo>/marlin-tune")
import vllm_hook  # noqa: F401
```

```
[marlin_tune] enabled: 8 table rows, workspace=sms*4, patched marlin_utils+mixed_precision.marlin
```

It rebinds `vllm._custom_ops.marlin_gemm` (`marlin_utils.py` does
`from vllm import _custom_ops as ops` and calls `ops.marlin_gemm(...)`, so the
module attribute is the right hook), and widens the Marlin lock workspace to
`sms * 4`. The workspace helper has to be rebound in **two** places —
`marlin_utils` and `vllm/model_executor/kernels/linear/mixed_precision/
marlin.py`, which imports it by name.

`marlin_tune.py` registers the fake/meta kernel for
`_C_marlin_tune::marlin_gemm`, mirroring vLLM's `_marlin_gemm_fake`, so dynamo
can trace through it.

## 5. Correctness

```bash
~/qwen-serving/venv/bin/python test_correctness.py --lm-head
```

Compares against `torch.ops._C.marlin_gemm` for
M ∈ {1, 4, 5, 8, 16} on all five production shapes. **Max abs diff is exactly
0.0 on every shape and every M** — the tuned build with default (auto) config
is bit-identical to stock.

The 4-bit shapes use real weights quantized with vLLM's own
`gptq_quantize_weights` / `marlin_weights` / `marlin_permute_scales` (done on
the CPU so a full bf16 copy of the weight never enters VRAM). The two int8
shapes use correctly-shaped random packed data, because a real CPU
quantization of a 248320×5120 weight needs ~8 GB of host RAM; both kernels read
identical bytes, so it still proves equivalence, and the 4-bit shapes validate
the packing itself.

## 6. Two bugs / footguns found in the stock kernel

### 6.1 `stages` must be in phase with the quantization group

When the quantization group spans more than one k-tile
(`group_blocks > thread_k_blocks`) the scale fetch only advances once every
`group_blocks / thread_k_blocks` pipeline steps. `stages` must be a multiple of
that ratio or the stage index and the scale index drift out of phase and the
kernel **silently returns garbage** (relative error ~0.4, no error raised).

Measured on sm86 at group_size 128 (`group_blocks = 8`):

| thread_k | tk_blocks | ratio | stages 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| 64  | 4 | 2 | ok | **WRONG** | ok | **WRONG** | ok |
| 128 | 8 | 1 | ok | ok | ok | ok | ok |
| 256 | 16 | 1 | ok | ok | ok | ok | ok |

Stock vLLM never hits this because it hardcodes `stages = 4`. Anyone tuning
this parameter will. `gen_kernels.stages_ok()` now refuses to emit those
instantiations, so an out-of-phase request raises "Unsupported shapes" instead
of computing nonsense.

### 6.2 The shared-memory request pins occupancy at 1 block/SM

`marlin_mm` launches with `max_shared_mem_new`, which for the auto path is the
device's **full** `cudaDevAttrMaxSharedMemoryPerBlockOptin` (99 KB on sm86),
regardless of how much the pipeline actually needs — and the kernel never reads
that value (`max_shared_mem` is an unused kernel parameter in
`marlin_template.h`). sm86 has 100 KB of shared memory per SM, so a 99 KB
request means **exactly one resident block per SM**.

The tile stock picks for every decode shape here is
`thread_k=128, thread_n=128, threads=256, stages=4`, which uses **96 registers
per thread** (`ptxas -v`): 256 × 96 = 24 576 registers, so the register file
would allow **2 blocks/SM**. Its actual pipeline footprint is ~48 KB. The 99 KB
request throws away half the achievable occupancy on a purely memory-bound
kernel.

The launch code *does* support more than one block per SM — it divides the
shared-memory request by `exec_cfg.blocks_per_sm` — but `determine_exec_config()`
unconditionally `return {1, th_config}`, so that path is dead code and
`blocks_per_sm` is always 1. Setting `blocks_per_sm=2` is therefore the single
highest-value knob here; `smem_mode=1` (request only
`get_kernel_cache_size()`, rounded up to 128 B) matters on top of that when you
want 3+ resident blocks, since the stock `max_shared_mem/blocks_per_sm - 1024`
formula only gets you to 2.
Registers per thread for the M ≤ 8 kernels (`ptxas -v`, 4-bit, group_blocks=8):

| tile (tk,tn,threads) | regs | blocks/SM allowed by registers |
|---|---|---|
| 64,128,128  | 96  | 5 |
| 128,64,128  | 96  | 5 |
| 64,256,256  | 96  | 2 |
| 128,128,256 | 96  | 2 |
| 256,64,256  | 95  | 2 |
| 128,256,512 | 96  | 1 |
| 256,128,512 | 128 | 1 |
| 512,64,512  | 96  | 1 |
| 256,256,1024 / 512,128,1024 | 96 | 0 — launch fails |

### 6.3 `blocks_per_sm > 1` writes out of bounds

`C_tmp` (the fp32 global-reduce scratch) is indexed inside the kernel as
`locks_off * c_size`, and `locks_off` runs up to `gridDim.x - 1`, i.e.
`sms * blocks_per_sm - 1`. But `marlin_gemm` sizes it for `sms` blocks only:

```cpp
int max_c_tmp_size = sms * max_m_block_size * marlin::max_thread_n;
```

Upstream this is latent — `determine_exec_config()` always returns
`blocks_per_sm == 1` — but it becomes an out-of-bounds write the moment
anything sets it higher. The slack in the formula (`max_thread_n` is 256 while
`thread_n` is often 128, and `max_m_block_size` is 16 while `m_block_size_8`
tiles use 8 rows) hides it for some shapes and not others: it first showed up
here as a `cudaErrorIllegalAddress` on the int8 lm_head, *after* the
`thread_n=128, blocks_per_sm=4` configs had already run at M=16 and silently
scribbled ~1.3 MB past the buffer.

`patch_marlin.py` multiplies the allocation by `blocks_per_sm`. Note the
`workspace`/`locks` buffer has the same requirement, and vLLM already
parameterizes that one — `marlin_make_workspace_new(device, max_blocks_per_sm)`
must be called with a matching value.

## 7. Benchmark

```bash
~/qwen-serving/venv/bin/python bench_marlin.py --grid --m 5 \
    --iters 1 --rounds 400
```

Shapes are the Qwen3.8-27B W4A16-AutoRound decode GEMMs (group_size 128
throughout, including the int8 lm_head/embed).

### Measurement methodology

The box shares one 3090 with other jobs, so the benchmark does two things:

* **Round-robin.** Every candidate is timed once per round, then the next
  round. Timing candidates to completion one after another bakes the clock ramp
  (210 MHz idle → 2130 MHz) and thermal droop into whoever ran first, which is
  worth several percent — more than most of the differences being measured.
* **`--iters 1`, minimum over many rounds.** GPU contexts are time-sliced at
  roughly millisecond granularity, so a single ~100 µs kernel usually lands
  entirely inside one of our slices; the minimum over a few hundred single-
  launch samples recovers the uncontended time. A 200-iteration inner loop
  always straddles context switches and ends up measuring the *other* job — with
  it, stock and the (bit-identical) tuned-auto build differed by ±8 % run to
  run; with single-launch timing they agree to the last decimal.

### Results

RTX 3090, vLLM 0.27.1, bf16 activations, group_size 128, no act-order, no
zero-points, `use_fp32_reduce=True`, `use_atomic_add=False`. `us` is the
minimum over 600 single-launch samples; GB/s counts weight bytes
(`N*K*bits/8 + (K/128)*N*2`). The GPU was shared with another vLLM job
throughout — see the methodology note above for why the numbers are still
usable, and note the ~1.02 us CUDA-event quantum, which is a full 3.5% on the
26 us o_proj.

| shape (N x K) | M | stock us | stock GB/s | best config (tk,tn,threads) | best us | best GB/s | gain |
|---|---|---|---|---|---|---|---|
| qkv 16384 x 5120 | 1 | 62.46 | 692 | `64,128,128 st=4 tight b/sm=2` | 59.39 | 728 | **+5.2%** |
| qkv 16384 x 5120 | 5 | 61.44 | 704 | `64,128,128 st=4 tight b/sm=2` | 59.39 | 728 | **+3.4%** |
| qkv 16384 x 5120 | 16 | 68.61 | 630 | `64,128,128 st=4 tight b/sm=2` | 65.54 | 660 | **+4.7%** |
| o_proj 5120 x 6144 | 1 | 29.70 | 546 | `256,64,256 st=3 tight b/sm=1` | 27.65 | 587 | **+7.4%** |
| o_proj 5120 x 6144 | 5 | 28.67 | 566 | `256,64,256 st=3 tight b/sm=1` | 27.65 | 587 | **+3.7%** |
| o_proj 5120 x 6144 | 16 | 30.72 | 528 | `256,64,256 st=3 tight b/sm=1` | 29.70 | 546 | **+3.4%** |
| gate_up 34816 x 5120 | 1 | 121.86 | 754 | `64,128,128 st=4 tight b/sm=2` | 116.74 | 787 | **+4.4%** |
| gate_up 34816 x 5120 | 5 | 120.83 | 761 | `64,128,128 st=4 tight b/sm=2` | 115.71 | 794 | **+4.4%** |
| gate_up 34816 x 5120 | 16 | 133.12 | 690 | `64,128,128 st=4 tight b/sm=3` | 124.93 | 736 | **+6.6%** |
| down 5120 x 17408 | 1 | 63.49 | 724 | `128,64,128 st=3 tight b/sm=2` | 62.46 | 736 | +1.6% |
| down 5120 x 17408 | 5 | 63.49 | 724 | (stock) | 63.49 | 724 | +0.0% |
| down 5120 x 17408 | 16 | 69.63 | 660 | `128,64,128 st=4 tight b/sm=2` | 68.61 | 670 | +1.5% |
| lm_head/2 124160 x 5120 int8 | 1 | 738.30 | 874 | (stock) | 737.28 | 876 | +0.1% |
| lm_head/2 124160 x 5120 int8 | 5 | 743.42 | 868 | (stock) | 741.38 | 871 | +0.3% |
| lm_head/2 124160 x 5120 int8 | 16 | 754.69 | 856 | (stock) | 751.62 | 859 | +0.4% |

`tight` = `smem_mode=1`, `b/sm` = `blocks_per_sm`. `threads` is always
`thread_k * thread_n / 64`.

### What the table says

* **The 4-bit decode GEMMs gain 3-7%.** Almost all of it comes from one thing:
  getting a second block resident per SM, which stock cannot do because
  `determine_exec_config()` hardcodes `blocks_per_sm = 1` and the launch then
  asks for all 99 KB of shared memory (6.2). Where 2 blocks/SM does not help
  (o_proj), a deeper k-tile with fewer, larger blocks does.
* **`stages` is worth little on its own.** 4 wins nearly everywhere; 3 wins on
  o_proj by about one event quantum. It is mostly a lever for fitting a tile
  into less shared memory so more blocks can be resident.
* **The int8 lm_head has no headroom.** It already runs at 856-876 GB/s, i.e.
  91-94% of the 3090's 936 GB/s, and every config in the sweep lands within
  ±0.5% of stock. `marlin_best.py` deliberately has no entry for it.
* **`down` (17408 -> 5120) is not reproducibly tunable at M ≤ 8.** The winner
  swaps between runs and the spread is about one event quantum, so the table
  keeps stock there.

Two measurement caveats worth repeating:

* The full 248320 x 5120 int8 lm_head (1.29 GB packed) does not fit next to a
  running vLLM server; it was verified for correctness when the GPU was briefly
  free (bit-identical at every M) but benchmarked via the half-width 124160
  stand-in. Same K, same dtype, same tile behaviour, so per-byte numbers carry
  over.
* An earlier pass using a 200-iteration inner loop reported the int8 lm_head at
  768 GB/s and showed 2-4% "gains" there. That was measuring the other job's
  time slices. At `--iters 1` the same shape measures 874 GB/s and the gains
  vanish. If a result looks like free performance, check the inner loop length
  against the kernel duration first.

### Shipped table path vs stock

`python bench_marlin.py --table --m 1,4,5,8,16 --iters 1 --rounds 600` — this
is the exact path serving takes: no `set_config`, the C++ table decides. It
measures identically to the manual override, as it should.

| shape | M=1 | M=4 | M=5 | M=8 | M=16 |
|---|---|---|---|---|---|
| qkv 16384×5120 | +3.4% | +3.4% | +3.4% | +3.4% | +4.7% |
| o_proj 5120×6144 | +7.2% | +7.7% | +3.7%\* | +7.7% | +0.0% |
| gate_up 34816×5120 | +4.4% | +5.3% | +5.3% | +4.4% | +5.7% |
| down 5120×17408 | +0.0% | +0.0% | +0.0% | +0.0% | +3.0% |
| lm_head int8 | +0.0% | +0.1% | +0.1% | +0.0% | +0.1% |

\* o_proj at M=5 landed one 1.02 µs event tick worse than its neighbours;
26.6 µs (+7.7%) is the representative number.

### Estimated saving per decode step

64 layers × [qkv, o_proj, gate_up, down] = 256 Marlin calls, plus the lm_head
(measured on the half-width shape and doubled, which is fair because it is
bandwidth-bound and linear in N).

| M | Marlin/layer stock → tuned | 256 GEMMs | + lm_head | saving |
|---|---|---|---|---|
| 1 | 274.4 → 265.3 µs | 17.56 → 16.98 ms | 19.04 → 18.46 ms | **0.58 ms (3.0%)** |
| 4 | 275.5 → 265.2 µs | 17.63 → 16.97 ms | 19.12 → 18.46 ms | **0.66 ms (3.4%)** |
| 5 | 275.5 → 266.2 µs | 17.63 → 17.04 ms | 19.12 → 18.53 ms | **0.59 ms (3.1%)** |
| 8 | 275.5 → 266.2 µs | 17.63 → 17.04 ms | 19.12 → 18.53 ms | **0.59 ms (3.1%)** |
| 16 | 300.0 → 287.8 µs | 19.20 → 18.42 ms | 20.71 → 19.92 ms | **0.79 ms (3.8%)** |

The stock column cross-checks against the production profile: 256 calls at
66.8 µs avg = 17.1 ms there, 17.6 ms here.

**This is below the 1.5-2.5 ms the task hoped for, and the gap is structural,
not a tuning failure.** Hitting 1.5-2.5 ms needs ~850 GB/s across all four
GEMMs. Tuned, they reach 728 / 609 / 794 / 724 GB/s (qkv / o_proj / gate_up /
down). The int8 lm_head shows what saturation actually looks like on this card
— 866-874 GB/s, 93% of peak — and it is the one GEMM where no config beats
stock. The others cannot get there because their weights are small enough that
fixed launch and tail cost is a large fraction of the kernel: o_proj moves
16 MB in 27 µs, so a few µs of ramp-up is ~15% of its runtime. Closing that
gap means fewer, larger GEMMs (fusing qkv+gate_up across layers, or a
persistent kernel), not a better tile.

### Reproduce

```bash
cd ~/marlin-tune/src
./run_all.sh                       # build + correctness + shortlist benchmark
python bench_marlin.py --grid --m 1,5,16 --iters 1 --rounds 600   # full sweep
python bench_marlin.py --table --m 1,4,5,8,16 --iters 1 --rounds 600
python test_correctness.py              # tuned-auto is bit-identical to stock
python test_correctness.py --use-best   # the shipped BEST table
python test_integration.py              # table dispatch, prefill, W4A8, clamping
```

## 8. Numerical differences

At the default (auto) config and on every path the table does not cover —
prefill (M ≥ 17, up to 512), W4A8, the int8 lm_head, `down` at M ≤ 8 — the
output is **bit-identical** to `torch.ops._C.marlin_gemm`.

Where the table does pick a different tile, results differ slightly, as
expected: `thread_k` changes how K is split across the pipeline, and
`blocks_per_sm` changes how many partial results go through the fp32 global
reduce, so the fp32 accumulation order changes and the bf16 output rounds
differently. Measured max abs diff across all shapes and M ∈ {1,4,5,8,16}:

| shape | max abs diff | |out| max | relative |
|---|---|---|---|
| qkv | 9.8e-4 | ~0.55 | 1.8e-3 |
| o_proj | 9.8e-4 | ~0.52 | 1.9e-3 |
| gate_up | 2.0e-3 | ~0.59 | 3.3e-3 |
| down (M=16) | 2.0e-3 | ~0.58 | 3.4e-3 |

That is one to two ulp of bf16 (eps ≈ 7.8e-3), i.e. the same order as changing
the reduction order of the *stock* kernel would produce. Nothing accumulates
in fp16.

## 9. CUDA-core int4 GEMV for M <= 8 -- attempted, does not win

`gemv_4bit.cu` is a complete, correct, non-tensor-core int4 skinny GEMM that
reads the Marlin-repacked weights and scales **in place** (no second copy).
It ties Marlin at M=1 and loses badly from M=2 up, so it ships **disabled**
(`marlin_tune_ext.set_gemv(True)` to try it). The numbers and the reason are
below because the reason is the useful part.

### The repacked layout (decode_layout.py proves this)

`B` is `uint32[K/16][2N]`. Row `r` covers `k in [16r, 16r+16)`. For word `w` of
a row, with `group = w/128`, `i = (w%128)/4`, `j = w%4`:

```
col   = i/4                                     in [0,8)
kset  = i%4
klocs = {2*kset, 2*kset+1, 2*kset+8, 2*kset+9}   = k0,k1,k2,k3
n0    = 16*(4*group + j) + col
n1    = n0 + 8
```

and the eight nibbles of that word are, in order,

```
(k0,n0) (k2,n0) (k0,n1) (k2,n1) (k1,n0) (k3,n0) (k1,n1) (k3,n1)
```

So a word feeds **two** output columns over four k -- not one column, which is
what you would guess from the packing code. Scales are `bf16[K/128][N]`
permuted along n by the 8x8 transpose within each 64-wide chunk (its own
inverse), so column `n` reads `(n/64)*64 + (n%8)*8 + ((n%64)/8)`.

`decode_layout.py` verifies both by tracing an index array through the exact
`marlin_permute_weights` / `get_weight_perm` pipeline and by round-tripping
real packed data: **0 mismatches**.

### Kernel

128 threads/block, 256 output columns per block. Each thread loads one `int4`
(4 packed words, 16 B); a warp's 32 threads cover 512 contiguous bytes. The
four words of a thread share `i`, hence the same `kset` and `col`, so **one set
of four activation values feeds 32 FMAs**. Nibbles become floats with one
`PRMT` each (`0x4B000000 | byte` reinterpreted is `8388608.0f + byte`) -- no
`I2F`, and no `half->float` (`F2F` is only 16/clk/SM on Ampere and would
bottleneck on its own). fp32 accumulate, group scale folded in per group of
128 k, `__shfl_xor_sync` by 1 and 2 to reduce the four threads sharing a
column, split-K across blocks with disjoint fp32 partial slots and a
last-block-reduces counter in the existing Marlin workspace.

Correctness: max abs diff vs `torch.ops._C.marlin_gemm` is <= 3.9e-3 on
|out| ~ 0.55 (rel <= 7.5e-3, i.e. one bf16 ulp) for all four 4-bit shapes at
M in {1,2,4,5,8}, on real quantized weights.

### Results (us, min of 500 single-launch samples)

| shape | M | tuned Marlin | GEMV | GEMV vs Marlin |
|---|---|---|---|---|
| qkv 16384x5120 | 1 | 59.4 (728 GB/s) | 60.4 (716) | -1.7% |
| | 2 | 59.4 | 66.6 (650) | -10.8% |
| | 4 | 59.4 | 88.1 (491) | -32.6% |
| | 5 | 59.4 | 92.2 (469) | -35.6% |
| | 8 | 59.4 | 128.0 (338) | -53.6% |
| o_proj 5120x6144 | 1 | 27.7 (587) | 28.7 (566) | -3.6% |
| | 5 | 27.7 | 48.1 (337) | -42.6% |
| gate_up 34816x5120 | 1 | 115.7 (794) | **114.7 (801)** | **+0.9%** |
| | 2 | 115.7 | 122.9 (748) | -5.8% |
| | 4 | 115.7 | 143.4 (641) | -19.3% |
| | 5 | 116.7 | 169.0 (544) | -30.9% |
| | 8 | 116.7 | 225.3 (408) | -48.2% |
| down 5120x17408 | 1 | 63.5 (724) | 64.5 (712) | -1.6% |
| | 5 | 63.5 | 103.4 (444) | -38.6% |

### Why -- two separate findings

**1. The M dimension is nearly free on tensor cores and linear on CUDA cores.**
Marlin from M=1 to M=8 on gate_up: 115.7 -> 116.7 us, **+0.9%**. The GEMV:
114.7 -> 225.3 us, **+96%**. One `m16n8k16` bf16 `mma` does 2048 MACs per warp
instruction; a warp-wide fp32 `FFMA` does 32. That 64x gap is the entire story
and no amount of tuning closes it.

Measured CUDA-core throughput, from the slope between M=4 and M=8:

| shape | us per unit M | FMA/s |
|---|---|---|
| qkv | 9.99 | 8.4e12 |
| o_proj | 5.17 | 6.1e12 |
| gate_up | 20.48 | 8.7e12 |
| down | 10.24 | 8.7e12 |

~8.5e12 FMA/s is **49% of the 17.8e12 fp32 peak** (82 SM x 128 lanes x
1.695 GHz). That is the single-pipe rate: each GA10x SM partition issues one
instruction per clock into either a 16-wide fp32-only datapath or a 16-wide
fp32/int32 one, and reaching 128 FMA/clk/SM needs both saturated by
back-to-back fp32 instructions. Our stream interleaves ~72 dequant
instructions per four words (8 masks + 32 PRMT = 40 INT, plus 32 fp32 SUB)
with 32*M FMAs, so the INT work steals issue slots and we sit at 64/clk.
At 8.7e12 FMA/s gate_up at M=5 needs 890e6/8.7e12 = 102 us of FMA alone, which
already equals the 102 us DRAM floor -- there is no room left for the dequant,
the shared-memory traffic or the split-K reduction, let alone to beat Marlin's
116 us.

**2. Marlin has no removable fixed overhead.** I went in expecting to reclaim
the ~9-12 us intercept implied by section 7 (fitting Marlin's time vs weight
bytes), on the theory that it was the 4-stage cp.async smem pipeline prologue.
It is not. A kernel with no smem pipeline, no tensor cores and a two-instruction
prologue lands on the *same* efficiency-vs-size curve:

| weight | Marlin | GEMV (M=1) |
|---|---|---|
| 16.2 MB (o_proj) | 587 GB/s | 566 GB/s |
| 43.3 MB (qkv) | 728 GB/s | 716 GB/s |
| 46.0 MB (down) | 724 GB/s | 712 GB/s |
| 91.9 MB (gate_up) | 794 GB/s | 801 GB/s |
| 645.6 MB (lm_head, int8) | 874 GB/s | -- |

Two unrelated kernels tracing the same curve means the "fixed cost" is a
property of the memory system -- it takes a while to get enough requests in
flight to saturate 936 GB/s, and a 16 MB read is over before that happens --
not of Marlin. The 850 GB/s target is reachable at 645 MB and not at 16-92 MB,
by any kernel.

### One thing that did not work, worth recording

Folding the `+8388616` dequant bias out of the inner loop (replacing 32 fp32
SUBs per row with `3*M` adds and one correction per group) is numerically
dead: the accumulator would hold `~2^23 * 128 * |a| ~ 1e9` while the answer is
`~1`, so fp32 cancellation destroys every digit -- measured rel err 0.3. It was
also slower, because the `asum` chain serializes per m. The subtract has to
happen per nibble.

### Conclusion

At the operating point that matters (M=4-5 for MTP decode) the CUDA-core GEMV
is 20-40% *slower* than tuned Marlin, so it is not shipped. Getting past
Marlin here needs fewer, larger GEMMs -- not a different kernel for the same
GEMM.

## 10. Not done

Not measured end to end: the tuned op has not been run inside a live vLLM
server, so there is no token/s number, only the per-GEMM arithmetic above.
`vllm_hook.py` is the one-line wiring for that.

Not covered by the build (these raise "Unsupported shapes" rather than
silently doing the wrong thing): fp16 activations, fp8 activations, AWQ /
zero-points, act-order with `is_k_full=False`, nvfp4 / mxfp4 / fp8 weights.

Not attempted: an int8-weight GEMV for the lm_head. It would be pointless --
that GEMM already runs at 874 GB/s (93% of peak) under Marlin, which is the
ceiling the 4-bit shapes cannot reach.

Not tuned: the W4A8 path. The table is guarded to 16-bit activations, so W4A8
runs stock. Tuning it would need its own sweep — the int8-activation kernel
halves the k-slice per pipeline stage, so the tile constraints differ, and
applying a bf16-tuned row to it produced wrong results before the guard went
in.
