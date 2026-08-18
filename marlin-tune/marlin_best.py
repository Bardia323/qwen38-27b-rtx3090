"""Per-shape best Marlin tile configuration for the RTX 3090 (sm86).

The table is keyed by (size_n, size_k, weight_bits) and, within that, by a
batch bucket, because the small-batch path behaves differently at M<=8
(m_block_size_8 kernels) than at 8<M<=16.

Usage
-----
    import marlin_best
    marlin_best.select(size_n, size_k, num_bits, size_m)   # before the gemm
    torch.ops._C_marlin_tune.marlin_gemm(...)

`select()` applies the tuned override when the shape is in the table and
otherwise resets to stock auto-selection, so it is safe to call for every
layer.  Regenerate the table with:

    python bench_marlin.py --grid --m 5,16 --rounds 9

on an *idle* GPU and paste the winning rows in.
"""

import marlin_tune_ext

# Knobs: thread_k, thread_n, stages, smem_mode, blocks_per_sm.
#   threads is always thread_k * thread_n / 64 (enforced by the extension).
#   smem_mode 1 = request only the shared memory the pipeline needs instead of
#   the full 99KB, which is what lets >1 block/SM actually be resident.
#
# Constraint baked into the generator: when thread_k < group_size (128), only
# `stages` that are a multiple of group_blocks/thread_k_blocks are correct --
# for thread_k=64 that means even stages only.
STOCK = dict(thread_k=128, thread_n=128, stages=4, smem_mode=0, blocks_per_sm=1)

# (size_n, size_k, num_bits) -> {m_bucket: cfg}
# m_bucket: 8 covers M<=8, 16 covers 8<M<=16.
# Measured on an RTX 3090 (sm86), vLLM 0.27.1, bf16 activations, group_size
# 128, no act-order, no zero-points.  Percentages are vs stock at the same M.
# Shapes whose best config beat stock by less than ~1.5% are deliberately left
# out: that is inside the run-to-run spread on this box, so stock is kept.
BEST = {
    # qkv:      5120 -> 16384
    (16384, 5120, 4): {
        8:  dict(thread_k=64, thread_n=128, stages=4, smem_mode=1,
                 blocks_per_sm=2),                  # +5.2% M=1, +3.4% M=5
        16: dict(thread_k=64, thread_n=128, stages=4, smem_mode=1,
                 blocks_per_sm=2),                  # +4.7%
    },
    # o_proj:   6144 -> 5120   (smallest weight, most latency-bound)
    (5120, 6144, 4): {
        8:  dict(thread_k=256, thread_n=64, stages=3, smem_mode=1,
                 blocks_per_sm=1),                  # +7.4% M=1, +3.7% M=5
        16: dict(thread_k=256, thread_n=64, stages=3, smem_mode=1,
                 blocks_per_sm=1),                  # +3.4%
    },
    # gate_up:  5120 -> 34816
    (34816, 5120, 4): {
        8:  dict(thread_k=64, thread_n=128, stages=4, smem_mode=1,
                 blocks_per_sm=2),                  # +4.4% M=1, +4.4% M=5
        16: dict(thread_k=64, thread_n=128, stages=4, smem_mode=1,
                 blocks_per_sm=3),                  # +6.6%
    },
    # down:     17408 -> 5120.  Nothing reproducible at M<=8 (the best config
    # swaps between runs and the spread is ~1 CUDA-event quantum), so the M<=8
    # bucket is an explicit copy of STOCK -- without it the M<=16 row below
    # would also apply to M<=8.
    (5120, 17408, 4): {
        8:  dict(STOCK),
        16: dict(thread_k=128, thread_n=64, stages=4, smem_mode=1,
                 blocks_per_sm=2),                  # +1.5%
    },
    # No entry for the int8 lm_head / embed (N=124160 or 248320, K=5120).
    # That GEMM already runs at 855-873 GB/s, i.e. 91-93% of the 3090's
    # 936 GB/s, and every config in the sweep lands within +-0.5% of stock.
    # There is nothing left to tune there; it is bandwidth-saturated.
}


def table():
    """BEST as flat rows for marlin_tune_ext.set_table().

    Row layout (10 ints):
        size_n, size_k, num_bits, m_max,
        thread_k, thread_n, threads, stages, blocks_per_sm, smem_mode

    `m_max` is the largest size_m the row applies to; the op picks the matching
    row with the smallest m_max, so an M<=8 row and an M<=16 row can coexist
    for the same shape.  `threads` is always thread_k*thread_n/64.
    """
    rows = []
    for (size_n, size_k, num_bits), buckets in sorted(BEST.items()):
        for m_max, cfg in sorted(buckets.items()):
            rows.append([
                size_n, size_k, num_bits, m_max,
                cfg["thread_k"], cfg["thread_n"],
                cfg["thread_k"] * cfg["thread_n"] // 64,
                cfg["stages"], cfg["blocks_per_sm"], cfg["smem_mode"],
            ])
    return rows


def lookup(size_n, size_k, num_bits, size_m):
    entry = BEST.get((size_n, size_k, num_bits))
    if entry is None or size_m > 16:
        return None
    bucket = 8 if size_m <= 8 else 16
    return entry.get(bucket)


def select(size_n, size_k, num_bits, size_m):
    """Apply the tuned config for this shape, or fall back to stock."""
    cfg = lookup(size_n, size_k, num_bits, size_m)
    if cfg is None:
        marlin_tune_ext.reset_config()
    else:
        marlin_tune_ext.set_config(**cfg)
    return cfg
