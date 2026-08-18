#!/usr/bin/env python3
"""Patch the vLLM v0.27.1 marlin sources into the tunable `marlin_tune` build.

Run once against a pristine copy of
  csrc/libtorch_stable/quantization/marlin/marlin.cu
  csrc/libtorch_stable/torch_utils.h
Every replacement must match exactly once, otherwise the script aborts, so a
future vLLM version fails loudly instead of being silently mis-patched.

What it changes
---------------
marlin.cu
  1. `namespace marlin` -> `namespace MARLIN_NAMESPACE_NAME` so -D can rename
     the whole extension and it cannot collide with the vLLM _C.so that is
     already loaded in the same process.
  2. Adds a tuning-config **table** keyed by (size_n, size_k, num_bits, m_max)
     plus a manual override for benchmarking.  The table is consulted inside
     marlin_mm(), so there is no Python-level branching on size_m -- important
     because under torch.compile the M dimension is dynamic and a Python
     `if size_m <= 16` would add guards / graph breaks, and because CUDA graph
     replay does not run Python at all.
  3. Applies the resolved config inside marlin_mm().
  4. Checks cudaGetLastError() after the kernel launch (stock never does, so a
     config that exceeds the per-block register/thread limit fails silently and
     leaves C untouched -- which benchmarks as an impossibly fast kernel).
  5. Sizes C_tmp for sms * blocks_per_sm blocks instead of sms, and clamps
     blocks_per_sm by the caller's workspace, so a caller that allocated the
     stock `sms`-entry lock buffer degrades to 1 block/SM instead of scribbling
     past it.
  6. Registers the op into its own torch library `_C_marlin_tune`.

torch_utils.h
  7. Drops the cuBLAS include + handle helper (marlin never uses them, and
     cublas_v2.h is not shipped by the nvidia-cuda-* pip wheels).
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MARLIN_DIR = os.path.join(HERE, "csrc", "libtorch_stable", "quantization", "marlin")
MARLIN_CU = os.path.join(MARLIN_DIR, "marlin.cu")
TORCH_UTILS = os.path.join(HERE, "csrc", "libtorch_stable", "torch_utils.h")


def sub_once(text, old, new, tag):
    n = text.count(old)
    if n != 1:
        sys.exit("patch_marlin: pattern %r matched %d times (expected 1)" % (tag, n))
    return text.replace(old, new)


TUNE_GLOBALS = r"""
// ---------------------------------------------------------------------------
// marlin_tune
//
// Two ways to override the tile configuration:
//
//   * set_config(...)  -- one global override, for benchmarking.
//   * set_table([...]) -- a (size_n, size_k, num_bits, m_max) -> config table,
//                         installed once from Python at import time.  The
//                         lookup happens *here*, inside marlin_mm, so the
//                         serving path stays a single torch op with no Python
//                         branching on the (dynamic) M dimension.
//
// set_config wins over the table.  A value <= 0 means "keep stock behaviour".
// ---------------------------------------------------------------------------
struct MarlinTuneConfig {
  int thread_k = -1;
  int thread_n = -1;
  int threads = -1;
  int stages = -1;
  int blocks_per_sm = -1;
  int max_par = -1;
  int smem_mode = 0;              // 0 = stock (always request the full 99KB),
                                  // 1 = request only what the pipeline needs
  int force_m_block_size_8 = -1;  // -1 auto, 0 off, 1 on
};

MarlinTuneConfig g_marlin_tune_cfg;

// [thread_k, thread_n, threads, stages, blocks, dyn_smem_bytes,
//  thread_m_blocks, m_block_size_8]
int g_marlin_tune_last[8] = {0, 0, 0, 0, 0, 0, 0, 0};

struct MarlinTuneEntry {
  int size_n;
  int size_k;
  int num_bits;
  int m_max;  // entry applies when size_m <= m_max
  int thread_k;
  int thread_n;
  int threads;
  int stages;
  int blocks_per_sm;
  int smem_mode;
};

static const int MARLIN_TUNE_ENTRY_INTS = 10;
static const int MARLIN_TUNE_MAX_ENTRIES = 256;
MarlinTuneEntry g_marlin_tune_table[MARLIN_TUNE_MAX_ENTRIES];
int g_marlin_tune_table_size = 0;

// Returns the matching entry with the smallest m_max, so a table can carry
// e.g. an M<=8 row and an M<=16 row for the same shape.
//
// `a_bits` is the *activation* width.  The table only ever applies to 16-bit
// activations: the W4A8 path (int8 activations, VLLM_MARLIN_INPUT_DTYPE=int8)
// halves the k-slice per pipeline stage, which changes the tile constraints,
// and it was never tuned or measured.  Without this guard a (N, K, 4) row
// tuned for bf16 activations would silently be applied to W4A8 and produce
// wrong results.
const MarlinTuneEntry* marlin_tune_lookup(int size_n, int size_k, int num_bits,
                                          int size_m, int a_bits) {
  if (a_bits != 16) return nullptr;
  const MarlinTuneEntry* best = nullptr;
  for (int i = 0; i < g_marlin_tune_table_size; i++) {
    const MarlinTuneEntry& e = g_marlin_tune_table[i];
    if (e.size_n != size_n || e.size_k != size_k || e.num_bits != num_bits)
      continue;
    if (size_m > e.m_max) continue;
    if (best == nullptr || e.m_max < best->m_max) best = &e;
  }
  return best;
}

// The blocks-per-SM marlin_mm will actually use.  Clamped by the caller's
// `locks` workspace: the kernel indexes locks[] by block, and vLLM's default
// marlin_make_workspace_new(device) only allocates `sms` entries.
int marlin_tune_blocks_per_sm(int size_n, int size_k, int num_bits, int size_m,
                              int a_bits, int sms, int64_t workspace_numel) {
  int bps = g_marlin_tune_cfg.blocks_per_sm;
  if (bps <= 0) {
    const MarlinTuneEntry* te =
        marlin_tune_lookup(size_n, size_k, num_bits, size_m, a_bits);
    bps = (te != nullptr) ? te->blocks_per_sm : 1;
  }
  if (bps < 1) bps = 1;
  int max_bps = (sms > 0) ? (int)(workspace_numel / sms) : 1;
  if (max_bps < 1) max_bps = 1;
  if (bps > max_bps) bps = max_bps;
  return bps;
}

extern "C" void marlin_tune_set_config(int thread_k, int thread_n, int threads,
                                       int stages, int blocks_per_sm,
                                       int max_par, int smem_mode,
                                       int force_m_block_size_8) {
  g_marlin_tune_cfg.thread_k = thread_k;
  g_marlin_tune_cfg.thread_n = thread_n;
  g_marlin_tune_cfg.threads = threads;
  g_marlin_tune_cfg.stages = stages;
  g_marlin_tune_cfg.blocks_per_sm = blocks_per_sm;
  g_marlin_tune_cfg.max_par = max_par;
  g_marlin_tune_cfg.smem_mode = smem_mode;
  g_marlin_tune_cfg.force_m_block_size_8 = force_m_block_size_8;
}

extern "C" void marlin_tune_get_last_config(int* out) {
  for (int i = 0; i < 8; i++) out[i] = g_marlin_tune_last[i];
}

// `data` is n_entries * 10 ints, laid out as MarlinTuneEntry.
// Returns the number of entries stored, or -1 if the table is too large.
extern "C" int marlin_tune_set_table(const int* data, int n_entries) {
  if (n_entries < 0 || n_entries > MARLIN_TUNE_MAX_ENTRIES) return -1;
  for (int i = 0; i < n_entries; i++) {
    const int* e = data + i * MARLIN_TUNE_ENTRY_INTS;
    MarlinTuneEntry& t = g_marlin_tune_table[i];
    t.size_n = e[0];
    t.size_k = e[1];
    t.num_bits = e[2];
    t.m_max = e[3];
    t.thread_k = e[4];
    t.thread_n = e[5];
    t.threads = e[6];
    t.stages = e[7];
    t.blocks_per_sm = e[8];
    t.smem_mode = e[9];
  }
  g_marlin_tune_table_size = n_entries;
  return n_entries;
}

extern "C" int marlin_tune_get_table_size() { return g_marlin_tune_table_size; }

// --- CUDA-core GEMV fast path (gemv_4bit.cu) -------------------------------
int g_marlin_gemv_enabled = 0;
int g_marlin_gemv_last[2] = {0, 0};  // {used_gemv, split_k}

bool gemv_supported(int prob_m, int prob_n, int prob_k, int num_bits,
                    int a_bits, int group_size, bool has_act_order,
                    bool has_zp, bool has_bias);
int gemv_split_k(int prob_n, int n_groups, int sms, int64_t workspace_numel);
bool gemv_4bit_launch(const void* A, const void* B, const void* S, void* C,
                      void* C_part, int* locks, int prob_m, int prob_n,
                      int prob_k, int lda, int split_k, cudaStream_t stream);

extern "C" void marlin_tune_set_gemv(int on) { g_marlin_gemv_enabled = on; }
extern "C" int marlin_tune_get_gemv() { return g_marlin_gemv_enabled; }
extern "C" void marlin_tune_get_gemv_last(int* out) {
  out[0] = g_marlin_gemv_last[0];
  out[1] = g_marlin_gemv_last[1];
}

// out receives {thread_k, thread_n, threads, stages, blocks_per_sm, smem_mode}
extern "C" int marlin_tune_lookup_c(int size_n, int size_k, int num_bits,
                                    int size_m, int a_bits, int* out) {
  const MarlinTuneEntry* e =
      marlin_tune_lookup(size_n, size_k, num_bits, size_m, a_bits);
  if (e == nullptr) return 0;
  out[0] = e->thread_k;
  out[1] = e->thread_n;
  out[2] = e->threads;
  out[3] = e->stages;
  out[4] = e->blocks_per_sm;
  out[5] = e->smem_mode;
  return 1;
}

"""

NEW_REGISTRATION = r"""STABLE_TORCH_LIBRARY(_C_marlin_tune, m) {
  m.def(
      "marlin_gemm(Tensor a, Tensor? c_or_none, Tensor b_q_weight, "
      "Tensor? b_bias_or_none,Tensor b_scales, "
      "Tensor? a_scales, Tensor? global_scale, Tensor? b_zeros_or_none, "
      "Tensor? "
      "g_idx_or_none, Tensor? perm_or_none, Tensor workspace, int b_type_id, "
      "SymInt size_m, SymInt size_n, SymInt size_k, bool is_k_full, "
      "bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(_C_marlin_tune, CUDA, m) {
  m.impl("marlin_gemm", TORCH_BOX(&marlin_gemm));
}
"""

# The block that resolves set_config / table into one effective config.
EFF_BLOCK = r"""
  // marlin_tune: resolve the effective tile config once, here in C++, so the
  // caller never has to branch on size_m in Python.
  MarlinTuneConfig eff = g_marlin_tune_cfg;
  if (eff.thread_k <= 0 || eff.thread_n <= 0) {
    const MarlinTuneEntry* te = marlin_tune_lookup(
        prob_n, prob_k, num_bits, prob_m, a_type.size_bits());
    if (te != nullptr) {
      eff.thread_k = te->thread_k;
      eff.thread_n = te->thread_n;
      if (eff.threads <= 0) eff.threads = te->threads;
      if (eff.stages <= 0) eff.stages = te->stages;
      if (eff.smem_mode == 0) eff.smem_mode = te->smem_mode;
    }
  }
  eff.blocks_per_sm =
      marlin_tune_blocks_per_sm(prob_n, prob_k, num_bits, prob_m,
                                a_type.size_bits(), sms, workspace_numel);
"""


GEMV_DISPATCH = r"""  // marlin_tune: CUDA-core GEMV fast path for M <= 8.  Reads the same
  // Marlin-repacked b_q_weight / b_scales in place, so it can be swapped in
  // per call while marlin keeps handling prefill.
  MARLIN_NAMESPACE_NAME::g_marlin_gemv_last[0] = 0;
  if (MARLIN_NAMESPACE_NAME::g_marlin_gemv_enabled && !has_act_order &&
      !has_zp && !has_bias && !global_scale_or_none.has_value() &&
      a.scalar_type() == torch::headeronly::ScalarType::BFloat16 &&
      c.scalar_type() == torch::headeronly::ScalarType::BFloat16 &&
      MARLIN_NAMESPACE_NAME::gemv_supported(
          size_m, size_n, size_k, b_type.size_bits(), a_type.size_bits(),
          group_size, has_act_order, has_zp, has_bias)) {
    int split_k = MARLIN_NAMESPACE_NAME::gemv_split_k(
        size_n, size_k / 128, sms, workspace.numel());
    torch::stable::Tensor part;
    void* part_ptr = nullptr;
    if (split_k > 1) {
      part = torch::stable::empty({(int64_t)split_k * size_m * size_n},
                                  torch::headeronly::ScalarType::Float,
                                  std::nullopt, device);
      part_ptr = part.mutable_data_ptr();
    }
    if (MARLIN_NAMESPACE_NAME::gemv_4bit_launch(
            a.const_data_ptr(), b_q_weight.const_data_ptr(),
            b_scales.const_data_ptr(), c.mutable_data_ptr(), part_ptr,
            (int*)workspace.mutable_data_ptr(), size_m, size_n, size_k,
            a.stride(0), split_k, get_current_cuda_stream(device_index))) {
      cudaError_t gemv_err = cudaGetLastError();
      STD_TORCH_CHECK(gemv_err == cudaSuccess,
                      "marlin_tune gemv launch failed: ",
                      cudaGetErrorString(gemv_err), " (M=", size_m,
                      ", N=", size_n, ", K=", size_k, ", split_k=", split_k,
                      ")");
      MARLIN_NAMESPACE_NAME::g_marlin_gemv_last[0] = 1;
      MARLIN_NAMESPACE_NAME::g_marlin_gemv_last[1] = split_k;
      return c;
    }
  }

  MARLIN_NAMESPACE_NAME::marlin_mm("""


def patch_marlin_cu():
    with open(MARLIN_CU) as f:
        t = f.read()

    # 1. namespace rename ---------------------------------------------------
    t = sub_once(t, "namespace marlin {", "namespace MARLIN_NAMESPACE_NAME {",
                 "namespace-open")
    assert t.count("}  // namespace marlin") == 2
    t = t.replace("}  // namespace marlin", "}  // namespace MARLIN_NAMESPACE_NAME")
    t = sub_once(t, "  marlin::marlin_mm(", "  MARLIN_NAMESPACE_NAME::marlin_mm(",
                 "marlin_mm-call")

    # 2. tuning globals + table ----------------------------------------------
    t = sub_once(
        t,
        "using MarlinFuncPtr = void (*)(MARLIN_KERNEL_PARAMS);\n",
        "using MarlinFuncPtr = void (*)(MARLIN_KERNEL_PARAMS);\n" + TUNE_GLOBALS,
        "tune-globals",
    )

    # 3. marlin_mm needs to know how big the lock workspace is ---------------
    t = sub_once(
        t,
        "               int lda, void* workspace, vllm::ScalarType const& a_type,",
        "               int lda, void* workspace, int64_t workspace_numel,\n"
        "               vllm::ScalarType const& a_type,",
        "marlin_mm-signature",
    )
    t = sub_once(
        t,
        "      workspace.mutable_data_ptr(), a_type, b_type, c_type, s_type, has_bias,",
        "      workspace.mutable_data_ptr(), workspace.numel(), a_type, b_type,\n"
        "      c_type, s_type, has_bias,",
        "marlin_mm-callsite",
    )

    # 4. resolve the effective config ----------------------------------------
    t = sub_once(
        t,
        "  int num_bits = b_type.size_bits();\n  const int4* A_ptr = (const int4*)A;",
        "  int num_bits = b_type.size_bits();\n" + EFF_BLOCK +
        "  const int4* A_ptr = (const int4*)A;",
        "eff-block",
    )

    # 5a. stages -------------------------------------------------------------
    t = sub_once(
        t,
        "  int stages = 4;\n  if (major_capability == 7 && minor_capability == 5) {",
        "  int stages = 4;\n"
        "  if (eff.stages > 0) stages = eff.stages;\n"
        "  if (major_capability == 7 && minor_capability == 5) {",
        "stages-override",
    )

    # 5b. max_par ------------------------------------------------------------
    t = sub_once(
        t,
        "  int max_par = 16;\n  if (prob_n <= 4096) max_par = 16 * 8;\n",
        "  int max_par = 16;\n"
        "  if (prob_n <= 4096) max_par = 16 * 8;\n"
        "  if (eff.max_par > 0) max_par = eff.max_par;\n",
        "max_par-override",
    )

    # 5c. tile / threads / m_block_size_8 ------------------------------------
    old = """    int thread_k = thread_k_init;
    int thread_n = thread_n_init;

    int thread_m_blocks = min(div_ceil(prob_m_split, 16), max_thread_m_blocks);
    int m_block_size_8 = prob_m_split <= 8 && a_type.size_bits() == 16;

    // Set thread config
    exec_config_t exec_cfg;
    thread_config_t thread_tfg;
    if (thread_k != -1 && thread_n != -1) {
      thread_tfg = thread_config_t{thread_k, thread_n, default_threads};
"""
    new = """    int thread_k = thread_k_init;
    int thread_n = thread_n_init;
    if (eff.thread_k > 0 && eff.thread_n > 0) {
      thread_k = eff.thread_k;
      thread_n = eff.thread_n;
    }

    int thread_m_blocks = min(div_ceil(prob_m_split, 16), max_thread_m_blocks);
    int m_block_size_8 = prob_m_split <= 8 && a_type.size_bits() == 16;
    if (eff.force_m_block_size_8 >= 0 && thread_m_blocks == 1 &&
        a_type.size_bits() == 16)
      m_block_size_8 = eff.force_m_block_size_8;

    // Set thread config
    exec_config_t exec_cfg;
    thread_config_t thread_tfg;
    if (thread_k != -1 && thread_n != -1) {
      // Stock hardcodes default_threads(=256) here, which is wrong for e.g.
      // (128, 64).  Every valid marlin tile obeys threads == tk * tn / 64.
      int forced_threads =
          eff.threads > 0 ? eff.threads : thread_k * thread_n / 64;
      thread_tfg = thread_config_t{thread_k, thread_n, forced_threads};
"""
    t = sub_once(t, old, new, "tile-override")

    # 5d. blocks_per_sm + shared-memory request ------------------------------
    old = """    int blocks = sms * exec_cfg.blocks_per_sm;
    if (exec_cfg.blocks_per_sm > 1)
      max_shared_mem_new = max_shared_mem / exec_cfg.blocks_per_sm - 1024;
"""
    new = """    exec_cfg.blocks_per_sm = eff.blocks_per_sm;
    int blocks = sms * exec_cfg.blocks_per_sm;
    if (exec_cfg.blocks_per_sm > 1)
      max_shared_mem_new = max_shared_mem / exec_cfg.blocks_per_sm - 1024;
    if (eff.smem_mode == 1) {
      // Stock marlin always requests the full 99KB of dynamic shared memory,
      // even though the kernel only ever addresses get_kernel_cache_size()
      // bytes (the `max_shared_mem` kernel argument is unused inside the
      // template).  That pins occupancy at 1 block/SM on sm86.  Ask for what
      // the pipeline actually needs instead.
      int need = get_kernel_cache_size(
          thread_tfg, thread_m_blocks, prob_m_split, prob_n, prob_k, num_bits,
          group_size, has_act_order, is_k_full, has_zp, is_zp_float, is_a_8bit,
          stages);
      need = div_ceil(need, 128) * 128;
      if (need < max_shared_mem_new) max_shared_mem_new = need;
    }
"""
    t = sub_once(t, old, new, "blocks-smem-override")

    # 5e. record what was actually used --------------------------------------
    old = """    int thread_k_blocks = thread_k / 16;
    int thread_n_blocks = thread_n / 16;
"""
    new = """    int thread_k_blocks = thread_k / 16;
    int thread_n_blocks = thread_n / 16;

    g_marlin_tune_last[0] = thread_k;
    g_marlin_tune_last[1] = thread_n;
    g_marlin_tune_last[2] = num_threads;
    g_marlin_tune_last[3] = stages;
    g_marlin_tune_last[4] = blocks;
    g_marlin_tune_last[5] = max_shared_mem_new;
    g_marlin_tune_last[6] = thread_m_blocks;
    g_marlin_tune_last[7] = m_block_size_8;
"""
    t = sub_once(t, old, new, "record-last")

    # 6. surface launch failures ---------------------------------------------
    old = """    // clang-format on

    bool is_a_8bit = a_type.size_bits() == 8;
"""
    new = """    // clang-format on

    cudaError_t launch_err = cudaGetLastError();
    STD_TORCH_CHECK(launch_err == cudaSuccess,
                    "marlin kernel launch failed: ",
                    cudaGetErrorString(launch_err), " (blocks = ", blocks,
                    ", threads = ", num_threads,
                    ", dyn_smem = ", max_shared_mem_new,
                    ", thread_k = ", thread_k, ", thread_n = ", thread_n,
                    ", stages = ", stages, ")");

    bool is_a_8bit = a_type.size_bits() == 8;
"""
    t = sub_once(t, old, new, "launch-error-check")

    # 7. size C_tmp for the real grid ----------------------------------------
    # The kernel indexes C_tmp as `locks_off * c_size`, and locks_off runs up
    # to gridDim.x - 1 == sms * blocks_per_sm - 1.  Stock sizes the buffer for
    # `sms` blocks only -- harmless upstream because determine_exec_config()
    # always returns blocks_per_sm == 1, an out-of-bounds write the moment
    # anything sets it higher.
    old_ctmp = """    int max_c_tmp_size =
        sms * max_m_block_size * MARLIN_NAMESPACE_NAME::max_thread_n;
"""
    new_ctmp = """    int tune_bps = MARLIN_NAMESPACE_NAME::marlin_tune_blocks_per_sm(
        size_n, size_k, b_type.size_bits(), size_m, a_type.size_bits(), sms,
        workspace.numel());
    int max_c_tmp_size = sms * tune_bps * max_m_block_size *
                         MARLIN_NAMESPACE_NAME::max_thread_n;
"""
    t = sub_once(t, old_ctmp, new_ctmp, "c_tmp-size")

    # 7.5 GEMV fast path ------------------------------------------------------
    t = sub_once(t, "  MARLIN_NAMESPACE_NAME::marlin_mm(", GEMV_DISPATCH,
                 "gemv-dispatch")

    # 8. own torch library ---------------------------------------------------
    t = sub_once(
        t,
        """STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("marlin_gemm", TORCH_BOX(&marlin_gemm));
}
""",
        NEW_REGISTRATION,
        "registration",
    )

    with open(MARLIN_CU, "w") as f:
        f.write(t)


def patch_torch_utils():
    with open(TORCH_UTILS) as f:
        t = f.read()
    t = sub_once(t, "#include <cublas_v2.h>\n", "", "cublas-include")
    old = """// Utility to get the current cuBLAS handle using stable APIs.
inline cublasHandle_t get_current_cuda_blas_handle() {
  void* blas_handle_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(torch_get_current_cuda_blas_handle(&blas_handle_ptr));
  return reinterpret_cast<cublasHandle_t>(blas_handle_ptr);
}"""
    t = sub_once(t, old, "// (cuBLAS handle helper removed: marlin does not use it)",
                 "cublas-handle")
    with open(TORCH_UTILS, "w") as f:
        f.write(t)


if __name__ == "__main__":
    patch_marlin_cu()
    patch_torch_utils()
    print("patch_marlin: ok")
