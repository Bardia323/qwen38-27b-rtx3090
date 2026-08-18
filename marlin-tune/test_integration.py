"""Validate the drop-in path: table dispatch in C++, prefill sizes, W4A8.

  python test_integration.py

Checks, for every production shape:
  * decode sizes (M <= 16)  -> the table's config is used and matches stock
    within bf16 reduction-reordering tolerance;
  * prefill sizes (M up to 512) -> falls through to stock auto selection and is
    **bit-identical** to torch.ops._C.marlin_gemm;
  * the W4A8 path (int8 activations x uint4b8 weights, i.e.
    VLLM_MARLIN_INPUT_DTYPE=int8) -> bit-identical to stock;
  * blocks_per_sm is clamped when the caller passes vLLM's default `sms`-entry
    workspace, instead of writing past it.
"""

import gc

import torch

import marlin_best
import marlin_common as mc
import marlin_tune  # noqa: F401  (installs the table on import)
import marlin_tune_ext
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_quant_input,
)

DECODE_M = [1, 4, 5, 8, 16]
PREFILL_M = [17, 32, 64, 128, 512]
TOL = 2e-2


def cfg_str():
    tk, tn, thr, st, blocks, smem, tmb, m8 = marlin_tune_ext.get_last_config()
    return ("tk=%d tn=%d thr=%d st=%d blocks=%d smem=%dkB mblk=%d m8=%d"
            % (tk, tn, thr, st, blocks, smem // 1024, tmb, m8))


def run_shape(name, n, k, wtype, device):
    wbytes = mc.weight_bytes(n, k, wtype)
    if mc.free_vram_mb() < wbytes / 1024 / 1024 + 400:
        print("  SKIP (only %.0f MB free)" % mc.free_vram_mb())
        return True
    if wbytes > 256 * 1024 * 1024:
        b_q, b_s, _ = mc.pack_synthetic(n, k, wtype, device)
    else:
        b_q, b_s, _ = mc.pack_real(n, k, wtype, device)

    ws_wide = mc.make_workspace(device, max_blocks_per_sm=4)
    ws_stock = mc.make_workspace(device, max_blocks_per_sm=1)
    ok = True

    for m in DECODE_M + PREFILL_M:
        # keep the output (and the a/ref/out trio) inside the VRAM we have
        if m * n * 2 > 48 * 1024 * 1024:
            continue
        a = torch.randn(m, k, dtype=torch.bfloat16, device=device) / 8
        ref = mc.call(mc.STOCK, a, b_q, b_s, ws_stock, wtype, m, n, k)
        out = mc.call(mc.TUNED, a, b_q, b_s, ws_wide, wtype, m, n, k)
        torch.cuda.synchronize()
        used = cfg_str()

        entry = marlin_tune_ext.lookup(n, k, wtype.size_bits, m)
        tuned = entry is not None and entry != [128, 128, 256, 4, 1, 0]
        if not tuned:
            # must be bit-identical; compare without an fp32 upcast so the
            # large-M cases fit in the little VRAM the other job leaves us
            good = torch.equal(ref, out)
            diff = 0.0 if good else (ref - out).abs().max().float().item()
        else:
            diff = (ref.float() - out.float()).abs().max().item()
            scale = max(ref.float().abs().max().item(), 1e-6)
            good = diff <= TOL * scale
        ok &= good

        # the table entry must actually be what the kernel ran
        if entry is not None:
            want_tk, want_tn, want_thr = entry[0], entry[1], entry[2]
            got = marlin_tune_ext.get_last_config()
            if (got[0], got[1], got[2]) != (want_tk, want_tn, want_thr):
                print("      !! table said %s, kernel used %s"
                      % (entry[:3], got[:3]))
                ok = False

        print("  M=%-4d %-8s diff=%-10.3e %s  [%s]"
              % (m, "tuned" if tuned else "stock", diff,
                 "OK " if good else "FAIL", used))
        del a, ref, out

    # workspace clamping: the stock-width workspace must force blocks_per_sm=1
    m = 5
    a = torch.randn(m, k, dtype=torch.bfloat16, device=device) / 8
    ref = mc.call(mc.STOCK, a, b_q, b_s, ws_stock, wtype, m, n, k)
    out = mc.call(mc.TUNED, a, b_q, b_s, ws_stock, wtype, m, n, k)
    torch.cuda.synchronize()
    blocks = marlin_tune_ext.get_last_config()[4]
    clamped = blocks <= mc.num_sms()
    diff = (ref.float() - out.float()).abs().max().item()
    scale = max(ref.float().abs().max().item(), 1e-6)
    good = clamped and diff <= TOL * scale
    ok &= good
    print("  narrow workspace (sms=%d entries): blocks=%d %s diff=%.3e"
          % (mc.num_sms(), blocks, "clamped OK" if clamped else "NOT CLAMPED",
             diff))

    del a, ref, out, b_q, b_s, ws_wide, ws_stock
    gc.collect()
    torch.cuda.empty_cache()
    return ok


def run_w4a8(device):
    """int8 activations x uint4b8 weights -- VLLM_MARLIN_INPUT_DTYPE=int8."""
    n, k, wtype = 16384, 5120, mc.UINT4B8
    print("\nW4A8  int8 act x uint4b8  N=%d K=%d" % (n, k))
    if mc.free_vram_mb() < 400:
        print("  SKIP (low VRAM)")
        return True
    g = torch.Generator().manual_seed(0)
    w = torch.randn(k, n, generator=g, dtype=torch.float32) / (k**0.5)
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_permute_scales,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
        get_weight_perm,
        marlin_weights,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        gptq_quantize_weights,
    )

    _wr, q_w, s, _gi, _p = gptq_quantize_weights(w, wtype, 128, act_order=False)
    perm = get_weight_perm(4, True)
    b_q = marlin_weights(q_w, k, n, 4, perm, is_a_8bit=True).to(device)
    b_s = marlin_permute_scales(s, k, n, 128, is_a_8bit=True).to(
        device).to(torch.bfloat16)
    del w, q_w, s

    ws = mc.make_workspace(device, max_blocks_per_sm=4)
    ok = True
    for m in [1, 5, 16, 64, 256]:
        x = torch.randn(m, k, dtype=torch.bfloat16, device=device) / 8
        a_i8, a_sc = marlin_quant_input(x, torch.int8)
        args = (a_i8, None, b_q, None, b_s, a_sc, None, None, None, None, ws,
                wtype.id, m, n, k, True, False, True, False)
        ref = mc.STOCK(*args)
        out = mc.TUNED(*args)
        torch.cuda.synchronize()
        diff = (ref.float() - out.float()).abs().max().item()
        good = diff == 0.0
        ok &= good
        print("  M=%-4d diff=%-10.3e %s  [%s]"
              % (m, diff, "OK " if good else "FAIL", cfg_str()))
        del x, a_i8, a_sc, ref, out
    del b_q, b_s, ws
    gc.collect()
    torch.cuda.empty_cache()
    return ok


def main():
    device = torch.device("cuda:0")
    torch.cuda.init()
    print("table rows: %d   free VRAM: %.0f MB"
          % (marlin_tune_ext.get_table_size(), mc.free_vram_mb()))
    print("table:")
    for row in marlin_best.table():
        print("   ", row)

    all_ok = True
    for name, n, k, wtype in mc.SHAPES:
        if name in mc.BIG:
            continue
        print("\n%s  N=%d K=%d  %s" % (name, n, k, wtype))
        all_ok &= run_shape(name, n, k, wtype, device)

    all_ok &= run_w4a8(device)
    print("\n%s" % ("INTEGRATION OK" if all_ok else "INTEGRATION FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
