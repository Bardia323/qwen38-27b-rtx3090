// Python bindings for the marlin_tune knobs.
//
// The GEMM itself is registered with the torch dispatcher as
// torch.ops._C_marlin_tune.marlin_gemm (see marlin.cu), keeping exactly the
// signature/semantics of vLLM 0.27.1's torch.ops._C.marlin_gemm.  Only the
// tuning knobs go through pybind, and none of them are meant to be touched
// per call in the serving path -- install the table once at import and the
// per-shape choice happens inside the op.

#include <torch/extension.h>

#include <stdexcept>
#include <vector>

extern "C" void marlin_tune_set_config(int thread_k, int thread_n, int threads,
                                       int stages, int blocks_per_sm,
                                       int max_par, int smem_mode,
                                       int force_m_block_size_8);
extern "C" void marlin_tune_get_last_config(int* out);
extern "C" int marlin_tune_set_table(const int* data, int n_entries);
extern "C" int marlin_tune_get_table_size();
extern "C" int marlin_tune_lookup_c(int size_n, int size_k, int num_bits,
                                    int size_m, int a_bits, int* out);
extern "C" void marlin_tune_set_gemv(int on);
extern "C" int marlin_tune_get_gemv();
extern "C" void marlin_tune_get_gemv_last(int* out);

namespace {
constexpr int kEntryInts = 10;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Tunable standalone build of vLLM 0.27.1 GPTQ-Marlin for sm86";

  m.def(
      "set_config",
      [](int thread_k, int thread_n, int threads, int stages,
         int blocks_per_sm, int max_par, int smem_mode,
         int force_m_block_size_8) {
        marlin_tune_set_config(thread_k, thread_n, threads, stages,
                               blocks_per_sm, max_par, smem_mode,
                               force_m_block_size_8);
      },
      py::arg("thread_k") = -1, py::arg("thread_n") = -1,
      py::arg("threads") = -1, py::arg("stages") = -1,
      py::arg("blocks_per_sm") = -1, py::arg("max_par") = -1,
      py::arg("smem_mode") = 0, py::arg("force_m_block_size_8") = -1,
      "Global override of the tile config, for benchmarking.  Wins over the "
      "table.  <=0 keeps stock behaviour.  smem_mode: 0 = stock (request the "
      "full 99KB of dynamic smem), 1 = request only what the pipeline needs.");

  m.def(
      "reset_config",
      []() { marlin_tune_set_config(-1, -1, -1, -1, -1, -1, 0, -1); },
      "Drop the global override (the table, if any, still applies).");

  m.def(
      "get_last_config",
      []() {
        int out[8];
        marlin_tune_get_last_config(out);
        return std::vector<int>(out, out + 8);
      },
      "[thread_k, thread_n, threads, stages, blocks, dyn_smem_bytes, "
      "thread_m_blocks, m_block_size_8] of the most recent launch.");

  m.def(
      "set_table",
      [](const std::vector<std::vector<int>>& rows) {
        std::vector<int> flat;
        flat.reserve(rows.size() * kEntryInts);
        for (const auto& r : rows) {
          if (r.size() != kEntryInts)
            throw std::runtime_error(
                "marlin_tune.set_table: each row must be 10 ints "
                "(size_n, size_k, num_bits, m_max, thread_k, thread_n, "
                "threads, stages, blocks_per_sm, smem_mode)");
          flat.insert(flat.end(), r.begin(), r.end());
        }
        int rc = marlin_tune_set_table(flat.empty() ? nullptr : flat.data(),
                                       static_cast<int>(rows.size()));
        if (rc < 0) throw std::runtime_error("marlin_tune.set_table: too many rows");
        return rc;
      },
      py::arg("rows"),
      "Install the (size_n, size_k, num_bits, m_max) -> config table.  Call "
      "once at import; the lookup then happens inside the op, so nothing "
      "branches on size_m in Python (torch.compile guards, cudagraph replay).");

  m.def(
      "set_gemv", [](bool on) { marlin_tune_set_gemv(on ? 1 : 0); },
      py::arg("on"),
      "Enable the CUDA-core int4 GEMV fast path for M<=8 (bf16 act, uint4b8, "
      "group 128, no act-order/zp/bias).  Off by default.");
  m.def("get_gemv", []() { return marlin_tune_get_gemv() != 0; });
  m.def(
      "get_gemv_last",
      []() {
        int out[2];
        marlin_tune_get_gemv_last(out);
        return std::vector<int>(out, out + 2);
      },
      "[used_gemv, split_k] for the most recent call.");

  m.def("clear_table", []() { marlin_tune_set_table(nullptr, 0); });
  m.def("get_table_size", []() { return marlin_tune_get_table_size(); });

  m.def(
      "lookup",
      [](int size_n, int size_k, int num_bits, int size_m,
         int a_bits) -> py::object {
        int out[6];
        if (!marlin_tune_lookup_c(size_n, size_k, num_bits, size_m, a_bits,
                                  out))
          return py::none();
        return py::cast(std::vector<int>(out, out + 6));
      },
      py::arg("size_n"), py::arg("size_k"), py::arg("num_bits"),
      py::arg("size_m"), py::arg("a_bits") = 16,
      "What the table would pick: [thread_k, thread_n, threads, stages, "
      "blocks_per_sm, smem_mode], or None.  a_bits is the activation width; "
      "the table only applies to 16-bit activations (W4A8 was never tuned).");
}
