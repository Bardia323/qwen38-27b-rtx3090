# marlin-tune (experiment, not shipped)

A standalone, tunable build of vLLM 0.27.1's GPTQ-Marlin kernel for sm86, produced while
chasing the single-user target. Result: the small-batch (M ≤ 16) tiles can be re-tuned
for the RTX 3090 for 3-7% per GEMM in isolation (the stock launch pins occupancy at one
block per SM by requesting all 99 KB of shared memory), but that did not translate into a
measurable end-to-end gain in the server, and a CUDA-core int4 GEMV written for M ≤ 8
ties Marlin at M=1 and loses from M=2 up. The useful part is the analysis in README.md
(sections 6-9): the remaining distance to peak bandwidth on these 16-92 MB weight reads
is the memory system's ramp-up, not the kernel, and the two latent stock bugs (`stages`
must be in phase with the quantization group; `blocks_per_sm > 1` writes out of bounds).

Paths in these files assume the repo layout (`~/qwen-serving`); adjust `VENV` and the
`sys.path.insert` in `vllm_hook.py`. Requires a CUDA 13 nvcc (README section 1: pip wheels,
no root). Nothing in the serving configs depends on this directory.
