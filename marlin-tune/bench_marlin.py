"""Benchmark stock vLLM 0.27.1 Marlin against the tunable marlin_tune build.

  python bench_marlin.py                       # stock vs tuned-default
  python bench_marlin.py --grid                # full tile/stages/smem sweep
  python bench_marlin.py --grid --m 1,4,5,8,16
  python bench_marlin.py --shapes gate_up --grid --top 20

All candidates for a given (shape, M) are timed **round-robin**: round 1 times
every candidate once, then round 2, and so on.  Measuring each candidate to
completion in turn instead would bake the 3090's clock ramp (210MHz idle ->
2130MHz) and thermal droop into whichever candidate ran first, which is worth
several percent -- more than most of the differences we are looking for.

Both the median and the minimum over rounds are reported.  If the box is
sharing the GPU with a vLLM server, trust the minimum.
"""

import argparse
import gc
import itertools

import torch

import marlin_common as mc
import marlin_tune_ext

# (thread_k, thread_n); threads is always thread_k*thread_n/64 -- see
# gen_kernels.py for why nothing else is safe to generate.
TILES = [
    (64, 128), (64, 256), (128, 64), (128, 128),
    (128, 256), (256, 64), (256, 128), (256, 256),
    (512, 64), (512, 128),
]
STAGES = [2, 3, 4, 5, 6]
SMEM_MODES = [0, 1]        # 0 = stock 99KB request, 1 = only what is needed
BLOCKS_PER_SM = [1, 2]


class Candidate:
    def __init__(self, label, op, cfg=None, gemv=False):
        self.label = label
        self.op = op
        self.cfg = cfg          # None => reset_config()
        self.gemv = gemv
        self.samples = []
        self.note = ""

    def apply(self):
        marlin_tune_ext.set_gemv(self.gemv)
        if self.cfg is None:
            marlin_tune_ext.reset_config()
        else:
            marlin_tune_ext.set_config(**self.cfg)

    @property
    def med(self):
        s = sorted(self.samples)
        return s[len(s) // 2]

    @property
    def best(self):
        return min(self.samples)

    @property
    def p10(self):
        s = sorted(self.samples)
        return s[max(0, len(s) // 10)]


def round_robin_time(cands, run, iters, rounds, warmup=20):
    """Time every candidate round-robin.

    With --iters 1 each sample covers a single kernel launch.  That matters
    when another process is using the GPU: contexts are time-sliced at roughly
    millisecond granularity, so a lone ~100us kernel often lands entirely
    inside one of our slices and the *minimum* over many samples recovers the
    uncontended time.  A long inner loop (iters=200) always straddles context
    switches and measures the other job instead.

    CUDA events quantize to ~1.024us, so iters=1 costs ~1.7% resolution at
    60us.  iters=4 divides that by four and still fits in a slice for kernels
    up to ~200us; for the ~800us lm_head use iters=1.
    """
    ev = [(torch.cuda.Event(enable_timing=True),
           torch.cuda.Event(enable_timing=True)) for _ in cands]
    for c in cands:
        c.apply()
        for _ in range(warmup):
            run(c.op)
    torch.cuda.synchronize()
    for _ in range(rounds):
        for i, c in enumerate(cands):
            c.apply()
            start, end = ev[i]
            start.record()
            for _ in range(iters):
                run(c.op)
            end.record()
            torch.cuda.synchronize()
            c.samples.append(start.elapsed_time(end) * 1000.0 / iters)
    marlin_tune_ext.reset_config()


def build_candidates(n, k, ref, run, poison, args):
    """Stock + tuned-auto + every grid config that reproduces stock's output."""
    cands = [Candidate("stock (vLLM _C)", mc.STOCK),
             Candidate("tuned auto", mc.TUNED)]
    if args.table:
        # exactly what serving runs: no set_config, the C++ table decides
        cands.append(Candidate("tuned TABLE (shipped)", mc.TUNED))
    if args.gemv:
        cands.append(Candidate("CUDA-core GEMV", mc.TUNED, gemv=True))
    if not args.grid:
        return cands, []

    rejected = []
    scale = max(ref.float().abs().max().item(), 1e-6)
    if args.only_configs:
        combos = []
        for spec in args.only_configs.split(";"):
            tk, tn, st, smem, bps = (int(x) for x in spec.split(","))
            combos.append(((tk, tn), st, smem, bps))
    else:
        combos = list(itertools.product(TILES, STAGES, SMEM_MODES,
                                        BLOCKS_PER_SM))
    for (tk, tn), st, smem, bps in combos:
        if n % tn or k % tk:
            continue
        # group_size 128 -> group_blocks 8; a k-tile smaller than the group
        # needs `stages` to be a multiple of group_blocks/thread_k_blocks or
        # the scale pipeline goes out of phase (see gen_kernels.stages_ok).
        if (tk // 16) < 8 and st % (8 // (tk // 16)):
            continue
        cfg = dict(thread_k=tk, thread_n=tn, stages=st,
                   smem_mode=smem, blocks_per_sm=bps)
        label = "%d,%d,%d st=%d %s b/sm=%d" % (
            tk, tn, tk * tn // 64, st, "tight" if smem else "stock", bps)
        marlin_tune_ext.set_config(**cfg)
        # Poison the output buffer first: a kernel that fails to launch would
        # otherwise leave the previous (correct) result in place and look like
        # a 20TB/s kernel.
        poison()
        try:
            out = run(mc.TUNED)
            torch.cuda.synchronize()
        except Exception as e:
            rejected.append((label, str(e).split("\n")[0][:90]))
            continue
        err = (out.float() - ref.float()).abs().max().item() / scale
        if err > 2e-2:
            rejected.append((label, "WRONG rel=%.2e" % err))
            continue
        c = Candidate(label, mc.TUNED, cfg)
        got = marlin_tune_ext.get_last_config()
        c.note = "smem=%dkB blocks=%d relerr=%.0e" % (
            got[5] // 1024, got[4], err)
        cands.append(c)
    marlin_tune_ext.reset_config()
    return cands, rejected


def bench_shape(name, n, k, wtype, device, args):
    wbytes = mc.weight_bytes(n, k, wtype)
    print("\n=== %s   N=%d K=%d  %s  (weight %.1f MB) ===" % (
        name, n, k, wtype, wbytes / 1e6))

    if mc.free_vram_mb() < wbytes / 1024 / 1024 + 300:
        print("SKIP: only %.0f MB free, need ~%.0f MB"
              % (mc.free_vram_mb(), wbytes / 1024 / 1024 + 300))
        return

    if wbytes > 256 * 1024 * 1024:
        b_q, b_s, _ = mc.pack_synthetic(n, k, wtype, device)
    else:
        b_q, b_s, _ = mc.pack_real(n, k, wtype, device)
    ws = mc.make_workspace(device)

    for m in args.m:
        a = torch.randn(m, k, dtype=torch.bfloat16, device=device) / 8
        c_out = torch.empty(m, n, dtype=torch.bfloat16, device=device)

        def run(op):
            return mc.call(op, a, b_q, b_s, ws, wtype, m, n, k,
                           use_atomic_add=args.atomic,
                           use_fp32_reduce=not args.no_fp32_reduce, c=c_out)

        marlin_tune_ext.reset_config()
        ref = run(mc.STOCK).clone()
        run(mc.TUNED)
        auto_cfg = mc.describe_cfg()

        def poison():
            c_out.fill_(float("nan"))

        cands, rejected = build_candidates(n, k, ref, run, poison, args)
        round_robin_time(cands, run, args.iters, args.rounds)
        marlin_tune_ext.set_gemv(False)

        stock = cands[0]
        if args.refine_top and len(cands) > args.refine_top:
            # CUDA events quantize to ~1.024us, which is 1.7% at 60us.  Re-time
            # the survivors with a few launches per event pair: still short
            # enough to fit inside one context-switch slice, but the
            # quantization is divided by `--refine-iters`.
            cands.sort(key=lambda c: c.best)
            keep = cands[: args.refine_top]
            if stock not in keep:
                keep.append(stock)
            for c in keep:
                c.samples = []
            round_robin_time(keep, run, args.refine_iters, args.rounds)
            cands = keep
        cands.sort(key=lambda c: c.best)
        print("\n M=%d   (%d candidates, %d rejected)   stock picks: %s"
              % (m, len(cands), len(rejected), auto_cfg))
        print("   %-28s %9s %9s %9s %9s %9s  %s"
              % ("config", "us(min)", "us(p10)", "us(med)", "GB/s", "vs stock",
                 "note"))
        for cand in cands[: args.top]:
            print("   %-28s %9.2f %9.2f %9.2f %9.1f %+8.1f%%  %s"
                  % (cand.label, cand.best, cand.p10, cand.med,
                     wbytes / cand.best / 1e3,
                     100 * (stock.best / cand.best - 1), cand.note))
        if stock not in cands[: args.top]:
            print("   %-28s %9.2f %9.2f %9.2f %9.1f %+8.1f%%  <- stock"
                  % (stock.label, stock.best, stock.p10, stock.med,
                     wbytes / stock.best / 1e3, 0.0))
        if rejected and args.show_invalid:
            print("   -- rejected (%d) --" % len(rejected))
            for label, why in rejected[:12]:
                print("      %-28s %s" % (label, why))
        del a, c_out, ref

    del b_q, b_s, ws
    gc.collect()
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", default="5", help="comma-separated batch sizes")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--grid", action="store_true", help="sweep the tuning grid")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--show-invalid", action="store_true")
    ap.add_argument("--gemv", action="store_true",
                    help="also time the CUDA-core int4 GEMV (M<=8, 4-bit only)")
    ap.add_argument("--table", action="store_true",
                    help="also time the shipped marlin_best table path")
    ap.add_argument("--only-configs", default="",
                    help='explicit shortlist "tk,tn,stages,smem,bps;..."')
    ap.add_argument("--refine-top", type=int, default=0,
                    help="re-time the N best candidates with --refine-iters")
    ap.add_argument("--refine-iters", type=int, default=8)
    ap.add_argument("--atomic", action="store_true", help="use_atomic_add=True")
    ap.add_argument("--no-fp32-reduce", action="store_true")
    ap.add_argument("--shapes", default="", help="substring filter on shape name")
    args = ap.parse_args()
    args.m = [int(x) for x in args.m.split(",")]
    if args.table:
        import marlin_tune  # noqa: F401  (installs the table on import)

        print("installed table rows: %d" % marlin_tune.N_TABLE_ROWS)

    device = torch.device("cuda:0")
    torch.cuda.init()
    props = torch.cuda.get_device_properties(0)
    print("%s  SMs=%d  smem/block(optin)=%d B  free VRAM=%.0f MB"
          % (props.name, props.multi_processor_count,
             props.shared_memory_per_block_optin, mc.free_vram_mb()))
    print("iters=%d rounds=%d  use_atomic_add=%s use_fp32_reduce=%s"
          % (args.iters, args.rounds, args.atomic, not args.no_fp32_reduce))

    for name, n, k, wtype in mc.SHAPES:
        if args.shapes and args.shapes not in name:
            continue
        bench_shape(name, n, k, wtype, device, args)


if __name__ == "__main__":
    main()
