"""Integration layer for the tuned Marlin op.

Importing this module:
  * loads the extension (which registers torch.ops._C_marlin_tune.marlin_gemm),
  * registers a fake/meta kernel so dynamo can trace through it,
  * installs the per-shape config table from marlin_best.py.

After that the op is a single, shape-agnostic torch op: the (M, N, K, num_bits)
-> tile-config decision happens inside C++, so nothing branches on the dynamic
M dimension in Python (no torch.compile guards or graph breaks) and CUDA graph
replay records the right kernel.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import marlin_best  # noqa: E402
import marlin_tune_ext  # noqa: E402  (registers torch.ops._C_marlin_tune)

try:
    from torch.library import register_fake
except ImportError:  # torch < 2.4
    from torch.library import impl_abstract as register_fake


# Mirrors vllm._custom_ops._marlin_gemm_fake.
@register_fake("_C_marlin_tune::marlin_gemm")
def _marlin_gemm_fake(
    a: torch.Tensor,
    c: torch.Tensor | None,
    b_q_weight: torch.Tensor,
    b_bias: torch.Tensor | None,
    b_scales: torch.Tensor,
    a_scales: torch.Tensor | None,
    global_scale: torch.Tensor | None,
    b_zeros: torch.Tensor | None,
    g_idx: torch.Tensor | None,
    perm: torch.Tensor | None,
    workspace: torch.Tensor,
    b_q_type_id: int,
    size_m: torch.SymInt,
    size_n: torch.SymInt,
    size_k: torch.SymInt,
    is_k_full: bool = True,
    use_atomic_add: bool = False,
    use_fp32_reduce: bool = False,
    is_zp_float: bool = False,
) -> torch.Tensor:
    dtype = a.dtype
    if dtype not in [torch.half, torch.bfloat16]:
        dtype = b_scales.dtype
    return torch.empty((size_m, size_n), device=a.device, dtype=dtype)


def marlin_gemm(
    a: torch.Tensor,
    c: torch.Tensor | None,
    b_q_weight: torch.Tensor,
    b_bias: torch.Tensor | None,
    b_scales: torch.Tensor,
    a_scales: torch.Tensor | None,
    global_scale: torch.Tensor | None,
    b_zeros: torch.Tensor | None,
    g_idx: torch.Tensor | None,
    perm: torch.Tensor | None,
    workspace: torch.Tensor,
    b_q_type,
    size_m: int,
    size_n: int,
    size_k: int,
    is_k_full: bool = True,
    use_atomic_add: bool = False,
    use_fp32_reduce: bool = False,
    is_zp_float: bool = False,
) -> torch.Tensor:
    """Signature-compatible replacement for vllm._custom_ops.marlin_gemm."""
    return torch.ops._C_marlin_tune.marlin_gemm(
        a,
        c,
        b_q_weight,
        b_bias,
        b_scales,
        a_scales,
        global_scale,
        b_zeros,
        g_idx,
        perm,
        workspace,
        b_q_type.id,
        size_m,
        size_n,
        size_k,
        is_k_full,
        use_atomic_add,
        use_fp32_reduce,
        is_zp_float,
    )


def install_table(rows=None):
    """Install the per-shape table. Returns the number of rows installed."""
    return marlin_tune_ext.set_table(
        marlin_best.table() if rows is None else rows
    )


# Installed on import so a plain `import marlin_tune` is enough.
N_TABLE_ROWS = install_table()
