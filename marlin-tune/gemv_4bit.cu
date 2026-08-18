// CUDA-core (non tensor-core) int4 weight-only skinny GEMM for M <= 8, reading
// vLLM's Marlin-repacked weights and scales IN PLACE -- no second copy of the
// weights.
//
// Layout (verified by decode_layout.py against marlin_weights/get_weight_perm)
// ---------------------------------------------------------------------------
// B is uint32[K/16][2N].  Row r covers k in [16r, 16r+16).  For word w of a
// row, with  group = w/128,  i = (w%128)/4,  j = w%4:
//
//     col   = i/4                       in [0,8)
//     kset  = i%4
//     klocs = {2*kset, 2*kset+1, 2*kset+8, 2*kset+9}      (k0,k1,k2,k3)
//     n0    = 16*(4*group + j) + col
//     n1    = n0 + 8
//
// and the eight nibbles of the word are, in order,
//     (k0,n0) (k2,n0) (k0,n1) (k2,n1) (k1,n0) (k3,n0) (k1,n1) (k3,n1)
//
// so one word feeds exactly two output columns over four k.  A warp reading 32
// consecutive words gets a fully coalesced 128-byte transaction covering a
// 16k x 16n sub-block, and the four threads that share an output column differ
// only in bits 2 and 3 of the lane id -- so the cross-thread reduction is two
// __shfl_xor_sync steps.
//
// Scales are bf16[K/128][N], permuted along n by the 8x8 transpose within each
// 64-wide chunk (which is its own inverse):
//     scale for column n lives at  (n/64)*64 + (n%8)*8 + ((n%64)/8)
//
// Dequant: nibbles are turned into floats with one PRMT each
// (0x4B000000 | byte == 8388608.0f + byte), so no I2F conversions and no
// half->float F2F (which is only 16/clk/SM on Ampere and would bottleneck).

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#ifndef MARLIN_NAMESPACE_NAME
  #define MARLIN_NAMESPACE_NAME marlin
#endif

namespace MARLIN_NAMESPACE_NAME {

namespace gemv {

// One thread loads an int4 (4 packed words = 16 bytes).  Those four words all
// share the same `i`, hence the same `kset` and `col`, and differ only in
// j = 0..3 -- so ONE set of four activation values feeds 32 FMAs instead of 8.
// That ratio is the whole game: at 1 shared-memory load per FMA the kernel is
// LDS-bound (32 lanes/clk/SM vs 128 FMA lanes/clk/SM, i.e. a hard 4x cap),
// which is exactly where the first version died for M >= 4.
constexpr int THREADS = 128;
constexpr int WORDS_PER_THREAD = 4;
constexpr int BLOCK_N = 256;   // output columns per block
constexpr int WORDS_PER_BLOCK_ROW = THREADS * WORDS_PER_THREAD;  // == 2*BLOCK_N
constexpr int GROUP = 128;     // quantization group along k
constexpr int ROWS_PER_GROUP = GROUP / 16;  // 8 k-tile rows per group

// 0x4B000000 | b  reinterpreted as float is 8388608.0f + b, for b in [0,256).
// The weights are uint4b8, so the true value is (nibble - 8).
constexpr float DEQUANT_BIAS = 8388608.0f + 8.0f;

__device__ __forceinline__ float nib_f(uint32_t bytes, int sel) {
  return __int_as_float(__byte_perm(bytes, 0x4B000000u, 0x7650u | sel));
}

template <int M>
__global__ __launch_bounds__(THREADS) void gemv_4bit_kernel(
    const __nv_bfloat16* __restrict__ A, const uint4* __restrict__ B,
    const __nv_bfloat16* __restrict__ S, __nv_bfloat16* __restrict__ C,
    float* __restrict__ C_part, int* __restrict__ locks, int prob_n,
    int prob_k, int lda, int n_groups, int split_k) {
  const int n_slice = blockIdx.x;
  const int split_id = blockIdx.y;
  const int t = threadIdx.x;

  // --- this thread's slot -------------------------------------------------
  const int lane32 = t % 32;
  const int col = lane32 / 4;
  const int kset = lane32 % 4;
  const int k0 = 2 * kset, k1 = 2 * kset + 1;
  const int k2 = 2 * kset + 8, k3 = 2 * kset + 9;
  const int wgroup = t / 32;                    // 0..3, the 64-column group
  const int n_base = n_slice * BLOCK_N + 64 * wgroup + col;
  // the four words differ only in j: n0_q = n_base + 16*q, n1_q = n0_q + 8

  int sp0[WORDS_PER_THREAD], sp1[WORDS_PER_THREAD];
#pragma unroll
  for (int q = 0; q < WORDS_PER_THREAD; q++) {
    int a0 = n_base + 16 * q, a1 = a0 + 8;
    sp0[q] = (a0 / 64) * 64 + (a0 % 8) * 8 + ((a0 % 64) / 8);
    sp1[q] = (a1 / 64) * 64 + (a1 % 8) * 8 + ((a1 % 64) / 8);
  }

  const int gr_begin = (int)((long)n_groups * split_id / split_k);
  const int gr_end = (int)((long)n_groups * (split_id + 1) / split_k);

  __shared__ float ash[M][GROUP];

  float tot0[WORDS_PER_THREAD][M], tot1[WORDS_PER_THREAD][M];
#pragma unroll
  for (int q = 0; q < WORDS_PER_THREAD; q++)
#pragma unroll
    for (int m = 0; m < M; m++) {
      tot0[q][m] = 0.f;
      tot1[q][m] = 0.f;
    }

  // B in uint4 units: one row is 2*prob_n words == prob_n/2 uint4
  const size_t row_stride16 = (size_t)prob_n / 2;
  const uint4* Bp = B + (size_t)gr_begin * ROWS_PER_GROUP * row_stride16 +
                    (size_t)n_slice * (WORDS_PER_BLOCK_ROW / 4) + t;

  for (int gr = gr_begin; gr < gr_end; gr++) {
    for (int idx = t; idx < M * GROUP; idx += THREADS) {
      int m = idx / GROUP, kk = idx % GROUP;
      ash[m][kk] = __bfloat162float(A[(size_t)m * lda + gr * GROUP + kk]);
    }
    __syncthreads();

    float acc0[WORDS_PER_THREAD][M], acc1[WORDS_PER_THREAD][M];
#pragma unroll
    for (int q = 0; q < WORDS_PER_THREAD; q++)
#pragma unroll
      for (int m = 0; m < M; m++) {
        acc0[q][m] = 0.f;
        acc1[q][m] = 0.f;
      }

#pragma unroll
    for (int r = 0; r < ROWS_PER_GROUP; r++) {
      const uint4 v = Bp[(size_t)r * row_stride16];
      const uint32_t word[4] = {v.x, v.y, v.z, v.w};

      float w0n0[4], w1n0[4], w2n0[4], w3n0[4];
      float w0n1[4], w1n1[4], w2n1[4], w3n1[4];
#pragma unroll
      for (int q = 0; q < 4; q++) {
        const uint32_t lo = word[q] & 0x0F0F0F0Fu;
        const uint32_t hi = (word[q] >> 4) & 0x0F0F0F0Fu;
        // The +DEQUANT_BIAS must come off HERE, per nibble.  Deferring it to a
        // per-group correction (which would replace 32 SUBs with 3*M adds)
        // looks attractive on an issue-bound kernel but is numerically dead:
        // the accumulator would hold ~2^23 * 128 * |a| ~ 1e9 while the answer
        // is ~1, so fp32 cancellation destroys every significant digit.
        // Measured: rel err 0.3, and it was slower anyway (the asum chain
        // serializes per m).
        w0n0[q] = nib_f(lo, 0) - DEQUANT_BIAS;
        w2n0[q] = nib_f(hi, 0) - DEQUANT_BIAS;
        w1n0[q] = nib_f(lo, 2) - DEQUANT_BIAS;
        w3n0[q] = nib_f(hi, 2) - DEQUANT_BIAS;
        w0n1[q] = nib_f(lo, 1) - DEQUANT_BIAS;
        w2n1[q] = nib_f(hi, 1) - DEQUANT_BIAS;
        w1n1[q] = nib_f(lo, 3) - DEQUANT_BIAS;
        w3n1[q] = nib_f(hi, 3) - DEQUANT_BIAS;
      }

      const int base = 16 * r;
#pragma unroll
      for (int m = 0; m < M; m++) {
        // four shared-memory loads, then 32 FMAs off them
        const float a0 = ash[m][base + k0];
        const float a1 = ash[m][base + k1];
        const float a2 = ash[m][base + k2];
        const float a3 = ash[m][base + k3];
#pragma unroll
        for (int q = 0; q < 4; q++) {
          acc0[q][m] = fmaf(w0n0[q], a0, acc0[q][m]);
          acc1[q][m] = fmaf(w0n1[q], a0, acc1[q][m]);
          acc0[q][m] = fmaf(w1n0[q], a1, acc0[q][m]);
          acc1[q][m] = fmaf(w1n1[q], a1, acc1[q][m]);
          acc0[q][m] = fmaf(w2n0[q], a2, acc0[q][m]);
          acc1[q][m] = fmaf(w2n1[q], a2, acc1[q][m]);
          acc0[q][m] = fmaf(w3n0[q], a3, acc0[q][m]);
          acc1[q][m] = fmaf(w3n1[q], a3, acc1[q][m]);
        }
      }
    }

#pragma unroll
    for (int q = 0; q < WORDS_PER_THREAD; q++) {
      const float s0 = __bfloat162float(S[(size_t)gr * prob_n + sp0[q]]);
      const float s1 = __bfloat162float(S[(size_t)gr * prob_n + sp1[q]]);
#pragma unroll
      for (int m = 0; m < M; m++) {
        tot0[q][m] = fmaf(acc0[q][m], s0, tot0[q][m]);
        tot1[q][m] = fmaf(acc1[q][m], s1, tot1[q][m]);
      }
    }

    Bp += (size_t)ROWS_PER_GROUP * row_stride16;
    __syncthreads();
  }

  // --- reduce the four threads (kset 0..3) that share each column ---------
#pragma unroll
  for (int q = 0; q < WORDS_PER_THREAD; q++)
#pragma unroll
    for (int m = 0; m < M; m++) {
#pragma unroll
      for (int d = 1; d <= 2; d <<= 1) {
        tot0[q][m] += __shfl_xor_sync(0xffffffffu, tot0[q][m], d);
        tot1[q][m] += __shfl_xor_sync(0xffffffffu, tot1[q][m], d);
      }
    }
  const bool writer = (kset == 0);

  if (split_k == 1) {
    if (writer) {
#pragma unroll
      for (int q = 0; q < WORDS_PER_THREAD; q++)
#pragma unroll
        for (int m = 0; m < M; m++) {
          C[(size_t)m * prob_n + n_base + 16 * q] = __float2bfloat16(tot0[q][m]);
          C[(size_t)m * prob_n + n_base + 16 * q + 8] =
              __float2bfloat16(tot1[q][m]);
        }
    }
    return;
  }

  float* part = C_part + (size_t)split_id * M * prob_n;
  if (writer) {
#pragma unroll
    for (int q = 0; q < WORDS_PER_THREAD; q++)
#pragma unroll
      for (int m = 0; m < M; m++) {
        part[(size_t)m * prob_n + n_base + 16 * q] = tot0[q][m];
        part[(size_t)m * prob_n + n_base + 16 * q + 8] = tot1[q][m];
      }
  }
  __threadfence();
  __syncthreads();

  __shared__ bool last;
  if (t == 0) {
    int old = atomicAdd(&locks[n_slice], 1);
    last = (old == split_k - 1);
    if (last) locks[n_slice] = 0;  // leave the workspace as we found it
  }
  __syncthreads();
  if (!last) return;

  for (int idx = t; idx < M * BLOCK_N; idx += THREADS) {
    int m = idx / BLOCK_N;
    int n = n_slice * BLOCK_N + (idx % BLOCK_N);
    float acc = 0.f;
    for (int sp = 0; sp < split_k; sp++)
      acc += C_part[((size_t)sp * M + m) * prob_n + n];
    C[(size_t)m * prob_n + n] = __float2bfloat16(acc);
  }
}

}  // namespace gemv

// How many k-splits to use: aim for ~4 blocks/SM without splitting more than
// there are quantization groups.
int gemv_split_k(int prob_n, int n_groups, int sms, int64_t workspace_numel) {
  int n_slices = prob_n / gemv::BLOCK_N;
  if (n_slices <= 0) return 0;
  // Aim for ~4 blocks/SM.  Each extra split costs an M*N fp32 write + read,
  // but empirically the extra parallelism wins: dropping the target to 3/SM
  // (and the cap to 8) cost o_proj M=1 20%.
  int want = (sms * 4 + n_slices - 1) / n_slices;
  if (want < 1) want = 1;
  if (want > 16) want = 16;
  if (want > n_groups) want = n_groups;
  // locks live in the caller's workspace
  if (want > 1 && n_slices > workspace_numel) want = 1;
  return want;
}

bool gemv_supported(int prob_m, int prob_n, int prob_k, int num_bits,
                    int a_bits, int group_size, bool has_act_order,
                    bool has_zp, bool has_bias) {
  return prob_m >= 1 && prob_m <= 8 && num_bits == 4 && a_bits == 16 &&
         group_size == gemv::GROUP && !has_act_order && !has_zp && !has_bias &&
         prob_n % gemv::BLOCK_N == 0 && prob_k % gemv::GROUP == 0;
}

// Returns false if the shape is not handled (caller falls back to marlin).
bool gemv_4bit_launch(const void* A, const void* B, const void* S, void* C,
                      void* C_part, int* locks, int prob_m, int prob_n,
                      int prob_k, int lda, int split_k, cudaStream_t stream) {
  const int n_groups = prob_k / gemv::GROUP;
  dim3 grid(prob_n / gemv::BLOCK_N, split_k);
  dim3 block(gemv::THREADS);

  auto a = (const __nv_bfloat16*)A;
  auto b = (const uint4*)B;
  auto s = (const __nv_bfloat16*)S;
  auto c = (__nv_bfloat16*)C;
  auto cp = (float*)C_part;

#define LAUNCH(MM)                                                          \
  case MM:                                                                  \
    gemv::gemv_4bit_kernel<MM><<<grid, block, 0, stream>>>(                 \
        a, b, s, c, cp, locks, prob_n, prob_k, lda, n_groups, split_k);      \
    break;

  switch (prob_m) {
    LAUNCH(1)
    LAUNCH(2)
    LAUNCH(3)
    LAUNCH(4)
    LAUNCH(5)
    LAUNCH(6)
    LAUNCH(7)
    LAUNCH(8)
    default:
      return false;
  }
#undef LAUNCH
  return true;
}

}  // namespace MARLIN_NAMESPACE_NAME
