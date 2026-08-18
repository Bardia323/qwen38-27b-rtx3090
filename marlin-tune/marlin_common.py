"""Shared helpers for the marlin_tune correctness test and benchmark.

Memory note: this box shares a 24GB 3090 with a running vLLM server that holds
~21.6GB, so everything here is built to keep GPU allocations small.  The 4-bit
weights are quantized on the CPU and only the packed result is moved to the
GPU; the 248320x5120 int8 lm_head weight (1.27GB packed) is only touched when
there is demonstrably enough free VRAM.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import marlin_tune_ext  # noqa: F401  (registers torch.ops._C_marlin_tune)

from vllm.model_executor.layers.quantization.utils.marlin_utils import (  # noqa: E402
    marlin_permute_scales,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (  # noqa: E402
    get_weight_perm,
    marlin_weights,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (  # noqa: E402
    gptq_quantize_weights,
)
from vllm.scalar_type import scalar_types  # noqa: E402

TILE = 16
GROUP_SIZE = 128

UINT4B8 = scalar_types.uint4b8
UINT8B128 = scalar_types.uint8b128

# (name, N, K, weight scalar type) -- the Qwen3.8-27B W4A16 decode GEMMs
# that dominate a single-stream step, plus the int8 lm_head.
SHAPES = [
    ("qkv      5120->16384", 16384, 5120, UINT4B8),
    ("o_proj   6144->5120", 5120, 6144, UINT4B8),
    ("gate_up  5120->34816", 34816, 5120, UINT4B8),
    ("down     17408->5120", 5120, 17408, UINT4B8),
    # Half-width stand-in for the lm_head: the full 248320x5120 int8 weight is
    # 1.27GB packed and does not fit next to the running vLLM server.  This is
    # the same K, dtype and tile behaviour, so per-byte numbers extrapolate.
    ("lm_head/2 5120->124160 (int8)", 124160, 5120, UINT8B128),
    ("lm_head  5120->248320 (int8)", 248320, 5120, UINT8B128),
]

# Shapes that need more VRAM than is available while the server runs.
BIG = {"lm_head  5120->248320 (int8)"}


def free_vram_mb():
    free, _total = torch.cuda.mem_get_info()
    return free / 1024 / 1024


def num_sms(device=0):
    return torch.cuda.get_device_properties(device).multi_processor_count


def make_workspace(device, max_blocks_per_sm=4):
    return torch.zeros(
        num_sms(device.index if device.index is not None else 0) * max_blocks_per_sm,
        dtype=torch.int,
        device=device,
    )


def pack_real(n, k, wtype, device, group_size=GROUP_SIZE, seed=0):
    """Quantize a random weight on the CPU, return GPU marlin tensors.

    Returns (b_q_weight, b_scales, w_ref_cpu).  w_ref stays on the CPU so we
    never pay for a full bf16 copy of the weight in VRAM.
    """
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(k, n, generator=g, dtype=torch.float32) / (k**0.5)

    w_ref, q_w, s, _g_idx, _perm = gptq_quantize_weights(
        w, wtype, group_size, act_order=False
    )
    perm = get_weight_perm(wtype.size_bits, False)
    b_q = marlin_weights(q_w, k, n, wtype.size_bits, perm, is_a_8bit=False)
    b_s = marlin_permute_scales(s, k, n, group_size, is_a_8bit=False)
    del w, q_w, s
    return b_q.to(device), b_s.to(device).to(torch.bfloat16), w_ref


def pack_synthetic(n, k, wtype, device, group_size=GROUP_SIZE, seed=0):
    """Build correctly-shaped *random* marlin tensors straight on the GPU.

    Used for the 248320x5120 lm_head, where a real CPU quantization would need
    ~8GB of host RAM and a float reference is not affordable in VRAM anyway.
    Both kernels read the identical bytes, so this still validates that the
    tuned build is bit-equivalent to stock -- it just does not additionally
    validate the packing itself (the 4-bit shapes do that).
    """
    pack_factor = 32 // wtype.size_bits
    g = torch.Generator(device=device).manual_seed(seed)
    b_q = torch.randint(
        -(2**31), 2**31 - 1, (k // TILE, n * TILE // pack_factor),
        generator=g, dtype=torch.int32, device=device,
    )
    b_s = (
        torch.rand((k // group_size, n), generator=g, dtype=torch.float32,
                   device=device)
        * 0.01
        + 0.001
    ).to(torch.bfloat16)
    return b_q, b_s, None


def call(op, a, b_q, b_s, workspace, wtype, m, n, k,
         use_atomic_add=False, use_fp32_reduce=True, c=None):
    """Invoke either torch.ops._C.marlin_gemm or the marlin_tune build.

    Argument order is vLLM 0.27.1's:
      a, c_or_none, b_q_weight, b_bias_or_none, b_scales, a_scales,
      global_scale, b_zeros_or_none, g_idx_or_none, perm_or_none, workspace,
      b_type_id, size_m, size_n, size_k, is_k_full, use_atomic_add,
      use_fp32_reduce, is_zp_float
    """
    return op(
        a, c, b_q, None, b_s, None, None, None, None, None, workspace,
        wtype.id, m, n, k, True, use_atomic_add, use_fp32_reduce, False,
    )


STOCK = torch.ops._C.marlin_gemm
TUNED = torch.ops._C_marlin_tune.marlin_gemm


def weight_bytes(n, k, wtype, group_size=GROUP_SIZE):
    return n * k * wtype.size_bits // 8 + (k // group_size) * n * 2


def describe_cfg():
    tk, tn, thr, st, blocks, smem, tmb, m8 = marlin_tune_ext.get_last_config()
    return (
        "tk=%d tn=%d thr=%d stages=%d blocks=%d smem=%dB mblk=%d m8=%d"
        % (tk, tn, thr, st, blocks, smem, tmb, m8)
    )
