"""Verify marlin_tune matches vLLM 0.27.1's stock torch.ops._C.marlin_gemm.

  python test_correctness.py            # 4-bit shapes only (small VRAM)
  python test_correctness.py --lm-head  # also the 1.27GB int8 lm_head weight
"""

import argparse
import gc

import torch

import marlin_best
import marlin_common as mc
import marlin_tune_ext

M_VALUES = [1, 4, 5, 8, 16]


def check_shape(name, n, k, wtype, device, args):
    wbytes = mc.weight_bytes(n, k, wtype)
    if name in mc.BIG and not args.lm_head:
        print("SKIP  (needs --lm-head, %.2f GB of VRAM)" % (wbytes / 1e9))
        return True
    if mc.free_vram_mb() < wbytes / 1024 / 1024 + 300:
        print("SKIP  (only %.0f MB free, need ~%.0f MB; the vLLM server holds "
              "the rest)" % (mc.free_vram_mb(), wbytes / 1024 / 1024 + 300))
        return True

    if wbytes > 256 * 1024 * 1024:
        # CPU-side real quantization of these would need ~8GB of host RAM.
        b_q, b_s, w_ref = mc.pack_synthetic(n, k, wtype, device)
        ref_kind = "synthetic packing (equivalence-only)"
    else:
        b_q, b_s, w_ref = mc.pack_real(n, k, wtype, device)
        ref_kind = "real quantized weights"

    ws_a = mc.make_workspace(device)
    ws_b = mc.make_workspace(device)
    ok = True

    for m in M_VALUES:
        a = torch.randn(m, k, dtype=torch.bfloat16, device=device) / 8

        marlin_tune_ext.reset_config()
        out_stock = mc.call(mc.STOCK, a, b_q, b_s, ws_a, wtype, m, n, k)
        if args.use_best:
            marlin_best.select(n, k, wtype.size_bits, m)
        elif args.config:
            tk, tn, st, sm, bps = (int(x) for x in args.config.split(","))
            marlin_tune_ext.set_config(thread_k=tk, thread_n=tn, stages=st,
                                       smem_mode=sm, blocks_per_sm=bps)
        out_tuned = mc.call(mc.TUNED, a, b_q, b_s, ws_b, wtype, m, n, k)
        cfg = mc.describe_cfg()
        marlin_tune_ext.reset_config()
        torch.cuda.synchronize()

        diff = (out_stock.float() - out_tuned.float()).abs().max().item()
        scale = out_stock.float().abs().max().item()
        # A different tile config reorders the k-reduction, so bf16 rounding
        # differences are expected; the default (auto) config must be exact.
        tuned = args.config or args.use_best
        good = diff <= (args.tol * max(scale, 1e-6) if tuned else 0.0)
        ok &= good

        line = "  M=%-3d maxabsdiff=%-12.3e (|out|max=%.3f) %s   [%s]" % (
            m, diff, scale, "OK " if good else "FAIL", cfg)
        print(line)

        # extra: for the smallest 4-bit shape, also compare against a float
        # matmul of the dequantized reference, proving the packing is right.
        if w_ref is not None and m == 5 and args.float_ref:
            ref = (a.cpu().float() @ w_ref.float()).to(torch.bfloat16)
            rel = ((ref.float() - out_stock.cpu().float()).abs().max()
                   / max(scale, 1e-6)).item()
            print("        float-ref rel err vs stock: %.3e" % rel)

        del a, out_stock, out_tuned

    del b_q, b_s, w_ref, ws_a, ws_b
    gc.collect()
    torch.cuda.empty_cache()
    print("  (%s reference)" % ref_kind)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm-head", action="store_true",
                    help="also test the 248320x5120 int8 weight (1.27GB VRAM)")
    ap.add_argument("--use-best", action="store_true",
                    help="validate the shipped marlin_best.BEST table itself")
    ap.add_argument("--config", default="",
                    help='validate a specific tuned config "tk,tn,stages,smem,bps"')
    ap.add_argument("--tol", type=float, default=2e-2,
                    help="relative tolerance when --config is given")
    ap.add_argument("--float-ref", action="store_true",
                    help="additionally cross-check stock against a CPU float matmul")
    args = ap.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.init()
    print("free VRAM: %.0f MB, SMs: %d" % (mc.free_vram_mb(), mc.num_sms()))

    all_ok = True
    for name, n, k, wtype in mc.SHAPES:
        print("\n%s  N=%d K=%d  %s" % (name, n, k, wtype))
        all_ok &= check_shape(name, n, k, wtype, device, args)

    if args.config or args.use_best:
        print("\n%s" % ("ALL SHAPES WITHIN %.0e OF STOCK" % args.tol if all_ok
                        else "MISMATCH"))
    else:
        print("\n%s" % ("ALL SHAPES BIT-IDENTICAL TO STOCK" if all_ok
                        else "MISMATCH"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
