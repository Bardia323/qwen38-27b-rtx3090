"""Route vLLM's Marlin GEMM through the tuned build.

One-line integration -- import this before the model is loaded, e.g. at the top
of the serving entrypoint:

    import sys; sys.path.insert(0, "<repo>/marlin-tune")
    import vllm_hook  # noqa: F401

It is a no-op unless VLLM_MARLIN_TUNE=1, so it is safe to leave in place.

What it does
------------
1. Replaces ``vllm._custom_ops.marlin_gemm`` with a wrapper around
   ``torch.ops._C_marlin_tune.marlin_gemm``.  ``marlin_utils.py`` does
   ``from vllm import _custom_ops as ops`` and calls ``ops.marlin_gemm(...)``,
   so rebinding the module attribute is enough and the call still goes through
   one custom op -- no Python branching on the (dynamic) M dimension.
2. Widens the Marlin lock workspace to ``sms * 4``.  Tuned entries with
   ``blocks_per_sm > 1`` need ``sms * blocks_per_sm`` lock slots; the linear
   kernel allocates only ``sms``.  The op clamps itself to whatever workspace
   it is handed, so *without* this step the tuned entries silently degrade to
   1 block/SM (correct, just not faster).  ``vllm/model_executor/kernels/
   linear/mixed_precision/marlin.py`` imports the helper by name, so the
   rebinding has to happen there as well as in ``marlin_utils``.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

# sms * WORKSPACE_BLOCKS lock slots. 4 covers every blocks_per_sm we tuned.
WORKSPACE_BLOCKS = 4

enabled = False
info = "disabled (set VLLM_MARLIN_TUNE=1)"


def install():
    global enabled, info

    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    import marlin_tune

    import vllm._custom_ops as ops
    from vllm.model_executor.layers.quantization.utils import marlin_utils

    ops.marlin_gemm = marlin_tune.marlin_gemm

    _orig_workspace = marlin_utils.marlin_make_workspace_new

    def _wide_workspace(device, max_blocks_per_sm=1, existing=None):
        return _orig_workspace(
            device, max(max_blocks_per_sm, WORKSPACE_BLOCKS), existing
        )

    patched = []
    marlin_utils.marlin_make_workspace_new = _wide_workspace
    patched.append("marlin_utils")
    try:
        from vllm.model_executor.kernels.linear.mixed_precision import (
            marlin as mp_marlin,
        )

        mp_marlin.marlin_make_workspace_new = _wide_workspace
        patched.append("mixed_precision.marlin")
    except ImportError:
        pass

    enabled = True
    info = "enabled: %d table rows, workspace=sms*%d, patched %s" % (
        marlin_tune.N_TABLE_ROWS,
        WORKSPACE_BLOCKS,
        "+".join(patched),
    )
    return info


if os.environ.get("VLLM_MARLIN_TUNE") == "1":
    print("[marlin_tune] " + install(), flush=True)
else:
    print("[marlin_tune] " + info, flush=True)
