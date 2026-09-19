/*
Adapted from https://github.com/turboderp/exllamav2 and
https://github.com/qwopqwop200/GPTQ-for-LLaMa
*/

#include <cstdint>
#include <cstdio>
#include <cstdlib>  // [fa2_sm70] getenv for the VLLM_GPTQ_ZERO_C switch
#include <cstring>  // [fa2_sm70] strcmp for the VLLM_GPTQ_ZERO_C modes
#include <string>
#include <vector>  // [fa2_sm70] эталонная последовательность прохода (инвариант #92)

#include "../../torch_utils.h"
#include <torch/csrc/stable/ops.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "compat.cuh"
#include "matrix_view.cuh"
#include "qdq_2.cuh"
#include "qdq_3.cuh"
#include "qdq_4.cuh"
#include "qdq_8.cuh"

namespace vllm {
namespace gptq {

#define BLOCK_KN_SIZE 128
#define BLOCK_M_SIZE_MAX 8
#define MAX_GROUPS_IN_BLOCK (BLOCK_KN_SIZE / 32)
#define MAX_Q_GEMM_ROWS 50
// [ПОРОГ ПОДНЯТ 24 -> 33, ЗАМЕР НА VOLTA 02.08.2026]
// Значение 24 досталось от upstream exllama и на sm_70 НЕ МЕРИЛОСЬ НИ РАЗУ. Кривая времени сплошь
// по M (боевые формы при TP=2, мкс) показывает, что СТУПЕНЬ В ТОЧКЕ ПОРОГА НАПРАВЛЕНА ВВЕРХ:
//   17408x5120, 8 бит:  M=24 -> 616 (поплиточно)   M=25 -> 835 (reconstruct)   +36 %
//   5120x17408, 8 бит:  M=24 -> 617                M=25 -> 841                 +36 %
//   5120x5120,  8 бит:  M=24 -> 191                M=25 -> 270                 +41 %
// То есть на M=25..33 движок уходил на ветку, которая ХУЖЕ. Ветка reconstruct плоская по M
// (835 -> 856 при M=25 -> 64), поплиточная растёт на 24.4 мкс/строку, пересечение на M ~ 33
// (три формы дают 33/33/36).
// ЧЕСТНО ПРО ЭКСТРАПОЛЯЦИЮ: выше M=24 поплиточная ветка была НЕДОСТИЖИМА без правки этой самой
// константы, поэтому точка 33 вычислена по наклону, а не снята. Правка ОДНОВРЕМЕННО является
// экспериментом: после пересборки надо снять прямой A/B на M=25..40 и, если пересечение окажется
// раньше, опустить значение сюда же.
// Валюта: батчевый декод и спекулятивные пачки. На префилле НОЛЬ (там M=4096, обе ветки reconstruct).
// MAX_Q_GEMM_ROWS (4 бита) НЕ ТРОГАЕМ: замеренное пересечение ~55 против стоящих 50 даёт всего +8 %.
#define MAX_Q_GEMM_ROWS_8BIT 33
#define MAX_ALT_GEMM_ROWS 8
#define THREADS_X 32
#define THREADS_Y 32
#define DIVIDE(x, size) (((x) + (size) - 1) / (size))

#if defined(USE_ROCM)
  #include <hipblas/hipblas.h>
__host__ __forceinline__ hipblasStatus_t __compat_hipblasHgemm(
    hipblasHandle_t handle, hipblasOperation_t transA,
    hipblasOperation_t transB, int m, int n, int k, const half* alpha,
    const half* AP, int lda, const half* BP, int ldb, const half* beta,
    half* CP, int ldc) {
  return hipblasHgemm(handle, transA, transB, m, n, k,
                      reinterpret_cast<const hipblasHalf*>(alpha),
                      reinterpret_cast<const hipblasHalf*>(AP), lda,
                      reinterpret_cast<const hipblasHalf*>(BP), ldb,
                      reinterpret_cast<const hipblasHalf*>(beta),
                      reinterpret_cast<hipblasHalf*>(CP), ldc);
}
  #define hipblasHgemm __compat_hipblasHgemm

  // Previous version of PyTorch were converting to rocBLAS instead of hipBLAS.
  #define rocblas_operation_none HIPBLAS_OP_N
  #define rocblas_hgemm __compat_hipblasHgemm
#endif

__forceinline__ __device__ half2 dot22_8(half2 (&dq)[4], const half* a_ptr,
                                         const half2 g_result) {
  half2 result = {};
  const half2* a2_ptr = (const half2*)a_ptr;
#pragma unroll
  for (int i = 0; i < 4; i++) result = __hfma2(dq[i], *a2_ptr++, result);
  return __hadd2(result, g_result);
}

__forceinline__ __device__ float dot22_8_f(half2 (&dq)[4], const half* a_ptr) {
  half2 result = {};
  const half2* a2_ptr = (const half2*)a_ptr;
#pragma unroll
  for (int i = 0; i < 4; i++) result = __hfma2(dq[i], *a2_ptr++, result);
  return __half2float(__low2half(result)) + __half2float(__high2half(result));
}

__forceinline__ __device__ half2 dot22_8(half2 (&dq)[4], const half* a_ptr,
                                         const half2 g_result,
                                         const half qs_h) {
  half2 result = {};
  const half2* a2_ptr = (const half2*)a_ptr;
#pragma unroll
  for (int i = 0; i < 4; i++) result = __hfma2(dq[i], *a2_ptr++, result);
  return __hfma2(result, __halves2half2(qs_h, qs_h), g_result);
}

__forceinline__ __device__ half2 dot22_16(half2 (&dq)[8], const half* a_ptr,
                                          const half2 g_result,
                                          const half qs_h) {
  half2 result = {};
  const half2* a2_ptr = (const half2*)a_ptr;
#pragma unroll
  for (int i = 0; i < 8; i++) result = __hfma2(dq[i], *a2_ptr++, result);
  return __hfma2(result, __halves2half2(qs_h, qs_h), g_result);
}

__forceinline__ __device__ half2 dot22_32(half2 (&dq)[16], const half* a_ptr,
                                          const half2 g_result,
                                          const half qs_h) {
  half2 result = {};
  const half2* a2_ptr = (const half2*)a_ptr;
#pragma unroll
  for (int i = 0; i < 16; i += 1) result = __hfma2(dq[i], *a2_ptr++, result);
  return __hfma2(result, __halves2half2(qs_h, qs_h), g_result);
}

__forceinline__ __device__ float dot22_8_f(half2 (&dq)[4], const half* a_ptr,
                                           const float g_result,
                                           const float qs_f) {
  half2 result = {};
  const half2* a2_ptr = (const half2*)a_ptr;
#pragma unroll
  for (int i = 0; i < 4; i++) result = __hfma2(dq[i], *a2_ptr++, result);
  float result_f =
      __half2float(__low2half(result)) + __half2float(__high2half(result));
  return fma(result_f, qs_f, g_result);
}

__forceinline__ __device__ float dot22_16_f(half2 (&dq)[8], const half* a_ptr,
                                            const float g_result,
                                            const float qs_f) {
  half2 result = {};
  const half2* a2_ptr = (const half2*)a_ptr;
#pragma unroll
  for (int i = 0; i < 8; i++) result = __hfma2(dq[i], *a2_ptr++, result);
  float result_f =
      __half2float(__low2half(result)) + __half2float(__high2half(result));
  return fma(result_f, qs_f, g_result);
}

__forceinline__ __device__ float dot22_32_f(half2 (&dq)[16], const half* a_ptr,
                                            const float g_result,
                                            const float qs_f) {
  half2 result = {};
  const half2* a2_ptr = (const half2*)a_ptr;
#pragma unroll
  for (int i = 0; i < 16; i += 1) result = __hfma2(dq[i], *a2_ptr++, result);
  float result_f =
      __half2float(__low2half(result)) + __half2float(__high2half(result));
  return fma(result_f, qs_f, g_result);
}

__forceinline__ __device__ half dot22_8_h(half2 (&dq)[4], const half* a_ptr,
                                          const half g_result,
                                          const half qs_h) {
  // Use FP32 accumulator to avoid potential overflow since unscaled weights are
  // in the range -128..127

  float result = {};
#pragma unroll
  for (int i = 0; i < 4; i++) {
    half2 w01 = dq[i];
    float w0 = __low2float(w01);
    float w1 = __high2float(w01);
    float x0 = __half2float(*a_ptr++);
    float x1 = __half2float(*a_ptr++);
    result = fma(w0, x0, result);
    result = fma(w1, x1, result);
  }
  float qs = __half2float(qs_h);
  result *= qs;
  half result_h = __float2half_rn(result);
  return __hadd(result_h, g_result);
}

__forceinline__ __device__ half dot22_16_h(half2 (&dq)[8], const half* a_ptr,
                                           const half g_result,
                                           const half qs_h) {
  half2 result = {};
  const half2* a2_ptr = (const half2*)a_ptr;
#pragma unroll
  for (int i = 0; i < 8; i++) result = __hfma2(dq[i], *a2_ptr++, result);
  half result_h = __hadd(__low2half(result), __high2half(result));
  return __hfma(result_h, qs_h, g_result);
}

__forceinline__ __device__ half dot22_32_h(half2 (&dq)[16], const half* a_ptr,
                                           const half g_result,
                                           const half qs_h) {
  half2 result = {};
  const half2* a2_ptr = (const half2*)a_ptr;
#pragma unroll
  for (int i = 0; i < 16; i += 1) result = __hfma2(dq[i], *a2_ptr++, result);
  half result_h = __hadd(__low2half(result), __high2half(result));
  return __hfma(result_h, qs_h, g_result);
}

typedef void (*fp_gemm_half_q_half_gptq_kernel)(const half*, const uint32_t*,
                                                const uint32_t*, const half*,
                                                half*, const int, const int,
                                                const int, const int,
                                                const bool, const int*);

template <bool first_block, int m_count>
__global__ void gemm_half_q_half_gptq_4bit_kernel(
    const half* __restrict__ a, const uint32_t* __restrict__ b_q_weight,
    const uint32_t* __restrict__ b_gptq_qzeros,
    const half* __restrict__ b_gptq_scales, half* __restrict__ c,
    const int size_m, const int size_n, const int size_k, const int groups,
    const bool use_v2_format, const int* __restrict__ b_q_perm) {
  MatrixView_half a_(a, size_m, size_k);
  MatrixView_half_rw c_(c, size_m, size_n);
  MatrixView_q4_row b_gptq_qzeros_(b_gptq_qzeros, groups, size_n);
  MatrixView_half b_gptq_scales_(b_gptq_scales, groups, size_n);

  // GPTQv2 and GPTQv1 handles zero points differently
  int zero_offset = use_v2_format ? 0 : 1;

  auto t = threadIdx.x;

  // Block
  auto offset_n = blockIdx.x * BLOCK_KN_SIZE * 4;
  auto offset_m = blockIdx.y * m_count;
  auto offset_k = blockIdx.z * BLOCK_KN_SIZE;

  int end_k = min(offset_k + BLOCK_KN_SIZE, size_k);

  int n = offset_n + t * 4;

  // Preload block_a
  __shared__ half block_a[m_count][BLOCK_KN_SIZE];

  if (offset_k + t < end_k) {
    for (int m = 0; m < m_count; ++m) {
      const half* a_ptr = a_.item_ptr(offset_m + m, 0);
      half* block_a_ptr = block_a[m];

      half a0;
      if (b_q_perm)
        a0 = a_ptr[b_q_perm[offset_k + t]];
      else
        a0 = a_ptr[offset_k + t];
      block_a_ptr[t] = a0;
    }
  }

  // Zero output
  if (n >= size_n) return;

  __syncthreads();

  // Find initial group
  int groupsize = size_k / groups;
  int group = offset_k / groupsize;
  int nextgroup = offset_k + groupsize;

  // a, b offset
  int qk = offset_k / (32 / 4);

  const uint32_t* b_ptr = b_q_weight + qk * size_n + n;
  const half* a_ptr = &block_a[0][0];
  int a_stride = BLOCK_KN_SIZE;

  // Initial group
  int zeros[4];
  float scales[4];
  half2 z1z16[4][2];
  half2 y1y16[4][2];
  b_gptq_qzeros_.item4(zeros, group, n);
  b_gptq_scales_.item4_f(scales, group, n);
  dequant_4bit_8_prep_zero(zeros[0] + zero_offset, z1z16[0], y1y16[0]);
  dequant_4bit_8_prep_zero(zeros[1] + zero_offset, z1z16[1], y1y16[1]);
  dequant_4bit_8_prep_zero(zeros[2] + zero_offset, z1z16[2], y1y16[2]);
  dequant_4bit_8_prep_zero(zeros[3] + zero_offset, z1z16[3], y1y16[3]);

  // Column result
  float block_c[m_count][4] = {};

  // Dequantize and multiply
  int k = offset_k;
  while (k < end_k) {
    if (k == nextgroup) {
      group++;
      nextgroup += groupsize;
      b_gptq_qzeros_.item4(zeros, group, n);
      b_gptq_scales_.item4_f(scales, group, n);
      dequant_4bit_8_prep_zero(zeros[0] + zero_offset, z1z16[0], y1y16[0]);
      dequant_4bit_8_prep_zero(zeros[1] + zero_offset, z1z16[1], y1y16[1]);
      dequant_4bit_8_prep_zero(zeros[2] + zero_offset, z1z16[2], y1y16[2]);
      dequant_4bit_8_prep_zero(zeros[3] + zero_offset, z1z16[3], y1y16[3]);
    }

#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int4* b_ptr4 = (int4*)b_ptr;
      int4 load_int4 = *b_ptr4;

      half2 dq[4][4];
      dequant_4bit_8_gptq(load_int4.x, dq[0], z1z16[0], y1y16[0], size_n,
                          false);
      dequant_4bit_8_gptq(load_int4.y, dq[1], z1z16[1], y1y16[1], size_n,
                          false);
      dequant_4bit_8_gptq(load_int4.z, dq[2], z1z16[2], y1y16[2], size_n,
                          false);
      dequant_4bit_8_gptq(load_int4.w, dq[3], z1z16[3], y1y16[3], size_n,
                          false);

#pragma unroll
      for (int m = 0; m < m_count; m++) {
        block_c[m][0] = fma(dot22_8_f(dq[0], a_ptr + m * a_stride), scales[0],
                            block_c[m][0]);
        block_c[m][1] = fma(dot22_8_f(dq[1], a_ptr + m * a_stride), scales[1],
                            block_c[m][1]);
        block_c[m][2] = fma(dot22_8_f(dq[2], a_ptr + m * a_stride), scales[2],
                            block_c[m][2]);
        block_c[m][3] = fma(dot22_8_f(dq[3], a_ptr + m * a_stride), scales[3],
                            block_c[m][3]);
      }

      b_ptr += size_n;
      a_ptr += 8;
    }

    k += 32;
  }

  for (int m = 0; m < m_count; m++) {
    half2* out = (half2*)c_.item_ptr(offset_m + m, n);
    half2 result01 = __halves2half2(__float2half_rn(block_c[m][0]),
                                    __float2half_rn(block_c[m][1]));
    half2 result23 = __halves2half2(__float2half_rn(block_c[m][2]),
                                    __float2half_rn(block_c[m][3]));
    atomicAdd(out, result01);
    atomicAdd(out + 1, result23);
  }
}

template <bool first_block, int m_count>
__global__ void gemm_half_q_half_gptq_2bit_kernel(
    const half* __restrict__ a, const uint32_t* __restrict__ b_q_weight,
    const uint32_t* __restrict__ b_gptq_qzeros,
    const half* __restrict__ b_gptq_scales, half* __restrict__ c,
    const int size_m, const int size_n, const int size_k, const int groups,
    const bool use_v2_format, const int* __restrict__ b_q_perm) {
  MatrixView_half a_(a, size_m, size_k);
  MatrixView_half_rw c_(c, size_m, size_n);
  MatrixView_q2_row b_gptq_qzeros_(b_gptq_qzeros, groups, size_n);
  MatrixView_half b_gptq_scales_(b_gptq_scales, groups, size_n);

  // GPTQv2 and GPTQv1 handles zero points differently
  int zero_offset = use_v2_format ? 0 : 1;

  auto t = threadIdx.x;

  // Block
  auto offset_n = blockIdx.x * BLOCK_KN_SIZE * 4;
  auto offset_m = blockIdx.y * m_count;
  auto offset_k = blockIdx.z * BLOCK_KN_SIZE;

  int end_k = min(offset_k + BLOCK_KN_SIZE, size_k);

  int n = offset_n + t * 4;

  // Preload block_a
  __shared__ half block_a[m_count][BLOCK_KN_SIZE];

  if (offset_k + t < end_k) {
    for (int m = 0; m < m_count; ++m) {
      const half* a_ptr = a_.item_ptr(offset_m + m, 0);
      half* block_a_ptr = block_a[m];

      half a0;
      if (b_q_perm)
        a0 = a_ptr[b_q_perm[offset_k + t]];
      else
        a0 = a_ptr[offset_k + t];
      block_a_ptr[t] = a0;
    }
  }

  // Zero output
  if (n >= size_n) return;

  __syncthreads();

  // Find initial group
  int groupsize = size_k / groups;
  int group = offset_k / groupsize;
  int nextgroup = offset_k + groupsize;

  // a, b offset
  int qk = offset_k / (32 / 2);

  const uint32_t* b_ptr = b_q_weight + qk * size_n + n;
  const half* a_ptr = &block_a[0][0];
  int a_stride = BLOCK_KN_SIZE;

  // Initial group
  int zeros[4];
  half scales[4];
  b_gptq_qzeros_.item4(zeros, group, n);
  b_gptq_scales_.item4(scales, group, n);
  // Column result
  half block_c[m_count][4] = {};

  // Dequantize and multiply
  int k = offset_k;
  while (k < end_k) {
    if (k == nextgroup) {
      group++;
      nextgroup += groupsize;
      b_gptq_qzeros_.item4(zeros, group, n);
      b_gptq_scales_.item4(scales, group, n);
    }

#pragma unroll
    for (int j = 0; j < 1; j++) {
      const int4* b_ptr4 = (int4*)b_ptr;
      int4 load_int4 = *b_ptr4;

      half2 dq[4][8];
      dequant_2bit_16(load_int4.x, dq[0], size_n, zeros[0] + zero_offset);
      dequant_2bit_16(load_int4.y, dq[1], size_n, zeros[1] + zero_offset);
      dequant_2bit_16(load_int4.z, dq[2], size_n, zeros[2] + zero_offset);
      dequant_2bit_16(load_int4.w, dq[3], size_n, zeros[3] + zero_offset);

#pragma unroll
      for (int m = 0; m < m_count; m++) {
        block_c[m][0] =
            dot22_16_h(dq[0], a_ptr + m * a_stride, block_c[m][0], scales[0]);
        block_c[m][1] =
            dot22_16_h(dq[1], a_ptr + m * a_stride, block_c[m][1], scales[1]);
        block_c[m][2] =
            dot22_16_h(dq[2], a_ptr + m * a_stride, block_c[m][2], scales[2]);
        block_c[m][3] =
            dot22_16_h(dq[3], a_ptr + m * a_stride, block_c[m][3], scales[3]);
      }

      b_ptr += size_n;
      a_ptr += 16;
    }

    k += 16;
  }

  for (int m = 0; m < m_count; m++) {
    half2* out = (half2*)c_.item_ptr(offset_m + m, n);
    half2 result01 = __halves2half2(block_c[m][0], block_c[m][1]);
    half2 result23 = __halves2half2(block_c[m][2], block_c[m][3]);
    atomicAdd(out, result01);
    atomicAdd(out + 1, result23);
  }
}

template <bool first_block, int m_count>
__global__ void gemm_half_q_half_gptq_3bit_kernel(
    const half* __restrict__ a, const uint32_t* __restrict__ b_q_weight,
    const uint32_t* __restrict__ b_gptq_qzeros,
    const half* __restrict__ b_gptq_scales, half* __restrict__ c,
    const int size_m, const int size_n, const int size_k, const int groups,
    const bool use_v2_format, const int* __restrict__ b_q_perm) {
  MatrixView_half a_(a, size_m, size_k);
  MatrixView_half_rw c_(c, size_m, size_n);
  MatrixView_q3_row b_gptq_qzeros_(b_gptq_qzeros, groups, size_n);
  MatrixView_half b_gptq_scales_(b_gptq_scales, groups, size_n);

  // GPTQv2 and GPTQv1 handles zero points differently
  int zero_offset = use_v2_format ? 0 : 1;

  auto t = threadIdx.x;

  // Block
  auto offset_n = blockIdx.x * BLOCK_KN_SIZE * 4;
  auto offset_m = blockIdx.y * m_count;
  auto offset_k = blockIdx.z * BLOCK_KN_SIZE;

  int end_k = min(offset_k + BLOCK_KN_SIZE, size_k);

  int n = offset_n + t * 4;

  // Preload block_a
  __shared__ half block_a[m_count][BLOCK_KN_SIZE];

  if (offset_k + t < end_k) {
    for (int m = 0; m < m_count; ++m) {
      const half* a_ptr = a_.item_ptr(offset_m + m, 0);
      half* block_a_ptr = block_a[m];

      half a0;
      if (b_q_perm)
        a0 = a_ptr[b_q_perm[offset_k + t]];
      else
        a0 = a_ptr[offset_k + t];
      block_a_ptr[t] = a0;
    }
  }

  // Zero output
  if (n >= size_n) return;

  __syncthreads();

  // Find initial group
  int groupsize = size_k / groups;
  int group = offset_k / groupsize;
  int nextgroup = offset_k + groupsize;

  // a, b offset
  int qk = offset_k / 32 * 3;

  const uint32_t* b_ptr = b_q_weight + qk * size_n + n;
  const half* a_ptr = &block_a[0][0];
  int a_stride = BLOCK_KN_SIZE;

  // Initial group
  int zeros[4];
  half scales[4];
  b_gptq_qzeros_.item4(zeros, group, n);
  b_gptq_scales_.item4(scales, group, n);
  // Column result
  half block_c[m_count][4] = {};

  // Dequantize and multiply
  int k = offset_k;
  while (k < end_k) {
    if (k == nextgroup) {
      group++;
      nextgroup += groupsize;
      b_gptq_qzeros_.item4(zeros, group, n);
      b_gptq_scales_.item4(scales, group, n);
    }

#pragma unroll
    for (int j = 0; j < 1; j++) {
      int4 load_int4[3];
      load_int4[0] = *((int4*)b_ptr);
      b_ptr += size_n;
      load_int4[1] = *((int4*)b_ptr);
      b_ptr += size_n;
      load_int4[2] = *((int4*)b_ptr);
      b_ptr += size_n;

      half2 dq[4][16];
      dequant_3bit_32(load_int4[0].x, load_int4[1].x, load_int4[2].x, dq[0],
                      size_n, zeros[0] + zero_offset);
      dequant_3bit_32(load_int4[0].y, load_int4[1].y, load_int4[2].y, dq[1],
                      size_n, zeros[1] + zero_offset);
      dequant_3bit_32(load_int4[0].z, load_int4[1].z, load_int4[2].z, dq[2],
                      size_n, zeros[2] + zero_offset);
      dequant_3bit_32(load_int4[0].w, load_int4[1].w, load_int4[2].w, dq[3],
                      size_n, zeros[3] + zero_offset);

#pragma unroll
      for (int m = 0; m < m_count; m++) {
        block_c[m][0] =
            dot22_32_h(dq[0], a_ptr + m * a_stride, block_c[m][0], scales[0]);
        block_c[m][1] =
            dot22_32_h(dq[1], a_ptr + m * a_stride, block_c[m][1], scales[1]);
        block_c[m][2] =
            dot22_32_h(dq[2], a_ptr + m * a_stride, block_c[m][2], scales[2]);
        block_c[m][3] =
            dot22_32_h(dq[3], a_ptr + m * a_stride, block_c[m][3], scales[3]);
      }
      a_ptr += 32;
    }

    k += 32;
  }

  for (int m = 0; m < m_count; m++) {
    half2* out = (half2*)c_.item_ptr(offset_m + m, n);
    half2 result01 = __halves2half2(block_c[m][0], block_c[m][1]);
    half2 result23 = __halves2half2(block_c[m][2], block_c[m][3]);
    atomicAdd(out, result01);
    atomicAdd(out + 1, result23);
  }
}

template <bool first_block, int m_count>
__global__ void gemm_half_q_half_gptq_8bit_kernel(
    const half* __restrict__ a, const uint32_t* __restrict__ b_q_weight,
    const uint32_t* __restrict__ b_gptq_qzeros,
    const half* __restrict__ b_gptq_scales, half* __restrict__ c,
    const int size_m, const int size_n, const int size_k, const int groups,
    const bool use_v2_format, const int* __restrict__ b_q_perm) {
  MatrixView_half a_(a, size_m, size_k);
  MatrixView_half_rw c_(c, size_m, size_n);
  MatrixView_q8_row b_gptq_qzeros_(b_gptq_qzeros, groups, size_n);
  MatrixView_half b_gptq_scales_(b_gptq_scales, groups, size_n);

  // GPTQv2 and GPTQv1 handles zero points differently
  int zero_offset = use_v2_format ? 0 : 1;

  auto t = threadIdx.x;

  // Block
  auto offset_n = blockIdx.x * BLOCK_KN_SIZE * 4;
  auto offset_m = blockIdx.y * m_count;
  auto offset_k = blockIdx.z * BLOCK_KN_SIZE;

  int end_k = min(offset_k + BLOCK_KN_SIZE, size_k);

  int n = offset_n + t * 4;

  // Preload block_a
  __shared__ half block_a[m_count][BLOCK_KN_SIZE];

  if (offset_k + t < end_k) {
    for (int m = 0; m < m_count; ++m) {
      const half* a_ptr = a_.item_ptr(offset_m + m, 0);
      half* block_a_ptr = block_a[m];

      half a0;
      if (b_q_perm)
        a0 = a_ptr[b_q_perm[offset_k + t]];
      else
        a0 = a_ptr[offset_k + t];
      block_a_ptr[t] = a0;
    }
  }

  // Zero output
  if (n >= size_n) return;

  __syncthreads();

  // Find initial group
  int groupsize = size_k / groups;
  int group = offset_k / groupsize;
  int nextgroup = offset_k + groupsize;

  // a, b offset
  int qk = offset_k / (32 / 8);

  const uint32_t* b_ptr = b_q_weight + qk * size_n + n;
  const half* a_ptr = &block_a[0][0];
  int a_stride = BLOCK_KN_SIZE;

  // Initial group
  int zeros[4];
  half scales[4];
  b_gptq_qzeros_.item4(zeros, group, n);
  b_gptq_scales_.item4(scales, group, n);
  // Column result
  half block_c[m_count][4] = {};

  // Dequantize and multiply
  int k = offset_k;
  while (k < end_k) {
    if (k == nextgroup) {
      group++;
      nextgroup += groupsize;
      b_gptq_qzeros_.item4(zeros, group, n);
      b_gptq_scales_.item4(scales, group, n);
    }

#pragma unroll
    for (int j = 0; j < 4; j++) {
      int4 load_int4[2];
      load_int4[0] = *((int4*)b_ptr);
      b_ptr += size_n;
      load_int4[1] = *((int4*)b_ptr);
      b_ptr += size_n;

      half2 dq[4][4];
      dequant_8bit_8(load_int4[0].x, load_int4[1].x, dq[0], size_n,
                     zeros[0] + zero_offset);
      dequant_8bit_8(load_int4[0].y, load_int4[1].y, dq[1], size_n,
                     zeros[1] + zero_offset);
      dequant_8bit_8(load_int4[0].z, load_int4[1].z, dq[2], size_n,
                     zeros[2] + zero_offset);
      dequant_8bit_8(load_int4[0].w, load_int4[1].w, dq[3], size_n,
                     zeros[3] + zero_offset);

      for (int m = 0; m < m_count; m++) {
        block_c[m][0] =
            dot22_8_h(dq[0], a_ptr + m * a_stride, block_c[m][0], scales[0]);
        block_c[m][1] =
            dot22_8_h(dq[1], a_ptr + m * a_stride, block_c[m][1], scales[1]);
        block_c[m][2] =
            dot22_8_h(dq[2], a_ptr + m * a_stride, block_c[m][2], scales[2]);
        block_c[m][3] =
            dot22_8_h(dq[3], a_ptr + m * a_stride, block_c[m][3], scales[3]);
      }
      a_ptr += 8;
    }
    k += 32;
  }

  for (int m = 0; m < m_count; m++) {
    half2* out = (half2*)c_.item_ptr(offset_m + m, n);
    half2 result01 = __halves2half2(block_c[m][0], block_c[m][1]);
    half2 result23 = __halves2half2(block_c[m][2], block_c[m][3]);
    atomicAdd(out, result01);
    atomicAdd(out + 1, result23);
  }
}

fp_gemm_half_q_half_gptq_kernel pick_gemm_half_q_half_gptq_kernel(
    bool first_block, const int m_count, const int bit) {
#define SELECT_KERNEL(M_COUNT)                                             \
  if (m_count == M_COUNT) {                                                \
    if (bit == 2) return gemm_half_q_half_gptq_2bit_kernel<true, M_COUNT>; \
    if (bit == 3) return gemm_half_q_half_gptq_3bit_kernel<true, M_COUNT>; \
    if (bit == 4) return gemm_half_q_half_gptq_4bit_kernel<true, M_COUNT>; \
    if (bit == 8) return gemm_half_q_half_gptq_8bit_kernel<true, M_COUNT>; \
  }
#if BLOCK_M_SIZE_MAX >= 1
  SELECT_KERNEL(1);
#endif
#if BLOCK_M_SIZE_MAX >= 2
  SELECT_KERNEL(2);
#endif
#if BLOCK_M_SIZE_MAX >= 3
  SELECT_KERNEL(3);
#endif
#if BLOCK_M_SIZE_MAX >= 4
  SELECT_KERNEL(4);
#endif
#if BLOCK_M_SIZE_MAX >= 5
  SELECT_KERNEL(5);
#endif
#if BLOCK_M_SIZE_MAX >= 6
  SELECT_KERNEL(6);
#endif
#if BLOCK_M_SIZE_MAX >= 7
  SELECT_KERNEL(7);
#endif
#if BLOCK_M_SIZE_MAX >= 8
  SELECT_KERNEL(8);
#endif
  return NULL;
}

void gemm_half_q_half_cuda_part(const half* a, const uint32_t* b_q_weight,
                                const uint32_t* b_gptq_qzeros,
                                const half* b_gptq_scales, const int* b_q_perm,
                                half* c, int size_m, int size_n, int size_k,
                                int m_count, int groups, bool use_v2_format,
                                int bit) {
  dim3 blockDim, gridDim;
  blockDim.x = BLOCK_KN_SIZE;
  blockDim.y = 1;
  blockDim.z = 1;
  gridDim.x = DIVIDE(size_n, BLOCK_KN_SIZE * 4);
  gridDim.y = DIVIDE(size_m, m_count);
  gridDim.z = DIVIDE(size_k, BLOCK_KN_SIZE);

  fp_gemm_half_q_half_gptq_kernel kernel =
      pick_gemm_half_q_half_gptq_kernel(true, m_count, bit);

  const cudaStream_t stream = get_current_cuda_stream();
  kernel<<<gridDim, blockDim, 0, stream>>>(
      a, b_q_weight, b_gptq_qzeros, b_gptq_scales, c, size_m, size_n, size_k,
      groups, use_v2_format, b_q_perm);
}

__global__ void reconstruct_exllama_8bit_kernel(
    const uint32_t* __restrict__ b_q_weight, const int* __restrict__ b_q_perm,
    const uint32_t* __restrict__ b_gptq_qzeros,
    const half* __restrict__ b_gptq_scales, const int size_k, const int size_n,
    const int groups, const bool use_v2_format, half* __restrict__ b) {
  MatrixView_half_rw b_(b, size_k, size_n);
  MatrixView_q8_row b_gptq_qzeros_(b_gptq_qzeros, groups, size_n);
  MatrixView_half b_gptq_scales_(b_gptq_scales, groups, size_n);

  // GPTQv2 and GPTQv1 handles zero points differently
  int zero_offset = use_v2_format ? 0 : 1;

  auto offset_k = BLOCK_KN_SIZE * blockIdx.y;
  auto offset_n = BLOCK_KN_SIZE * blockIdx.x * 4;

  int end_k = min(offset_k + BLOCK_KN_SIZE, size_k);

  // Preload remapping table
  __shared__ int perm[BLOCK_KN_SIZE];
  auto t = threadIdx.x;

  if (b_q_perm) {
    if (offset_k + t < size_k) perm[t] = b_q_perm[offset_k + t];
  }

  // Column
  int n = offset_n + t * 4;
  if (n >= size_n) return;

  // Find initial group
  int groupsize = size_k / groups;
  int group = offset_k / groupsize;
  int nextgroup = offset_k + groupsize;

  // b offset
  int qk = offset_k / (32 / 8);

  const uint32_t* b_ptr = b_q_weight + qk * size_n + n;

  // Initial zeros/scale
  int zeros[4];
  half2 scales[4];
  b_gptq_qzeros_.item4(zeros, group, n);
  b_gptq_scales_.item4_h2(scales, group, n);

  __syncthreads();

  int k = offset_k;
  int lk = 0;

  while (k < end_k) {
    if (k == nextgroup) {
      group++;
      nextgroup += groupsize;
      b_gptq_qzeros_.item4(zeros, group, n);
      b_gptq_scales_.item4_h2(scales, group, n);
    }

    for (int p = 0; p < 4; p++) {
      int4 load_int4[2];
      load_int4[0] = *((int4*)b_ptr);
      b_ptr += size_n;
      load_int4[1] = *((int4*)b_ptr);
      b_ptr += size_n;

      half2 dq[4][4];
      dequant_8bit_8(load_int4[0].x, load_int4[1].x, dq[0], size_n,
                     zeros[0] + zero_offset);
      dequant_8bit_8(load_int4[0].y, load_int4[1].y, dq[1], size_n,
                     zeros[1] + zero_offset);
      dequant_8bit_8(load_int4[0].z, load_int4[1].z, dq[2], size_n,
                     zeros[2] + zero_offset);
      dequant_8bit_8(load_int4[0].w, load_int4[1].w, dq[3], size_n,
                     zeros[3] + zero_offset);

      // half* dqh = (half*)dq;
      if (b_q_perm) {
        for (int j = 0; j < 4; j++) {
          for (int v = 0; v < 4; v++) dq[v][j] = __hmul2(scales[v], dq[v][j]);
          b_.set4(perm[lk++], n, __low2half(dq[0][j]), __low2half(dq[1][j]),
                  __low2half(dq[2][j]), __low2half(dq[3][j]));
          b_.set4(perm[lk++], n, __high2half(dq[0][j]), __high2half(dq[1][j]),
                  __high2half(dq[2][j]), __high2half(dq[3][j]));
        }
      } else {
        for (int j = 0; j < 4; j++) {
          for (int v = 0; v < 4; v++) dq[v][j] = __hmul2(scales[v], dq[v][j]);
          b_.set4(offset_k + lk++, n, __low2half(dq[0][j]),
                  __low2half(dq[1][j]), __low2half(dq[2][j]),
                  __low2half(dq[3][j]));
          b_.set4(offset_k + lk++, n, __high2half(dq[0][j]),
                  __high2half(dq[1][j]), __high2half(dq[2][j]),
                  __high2half(dq[3][j]));
        }
      }
    }
    k += 32;
  }
}

__global__ void reconstruct_exllama_4bit_kernel(
    const uint32_t* __restrict__ b_q_weight, const int* __restrict__ b_q_perm,
    const uint32_t* __restrict__ b_gptq_qzeros,
    const half* __restrict__ b_gptq_scales, const int size_k, const int size_n,
    const int groups, const bool use_v2_format, half* __restrict__ b) {
  MatrixView_half_rw b_(b, size_k, size_n);
  MatrixView_q4_row b_gptq_qzeros_(b_gptq_qzeros, groups, size_n);
  MatrixView_half b_gptq_scales_(b_gptq_scales, groups, size_n);

  // GPTQv2 and GPTQv1 handles zero points differently
  int zero_offset = use_v2_format ? 0 : 1;

  auto offset_k = BLOCK_KN_SIZE * blockIdx.y;
  auto offset_n = BLOCK_KN_SIZE * blockIdx.x * 4;

  int end_k = min(offset_k + BLOCK_KN_SIZE, size_k);

  // Preload remapping table
  __shared__ int perm[BLOCK_KN_SIZE];
  auto t = threadIdx.x;

  if (b_q_perm) {
    if (offset_k + t < size_k) perm[t] = b_q_perm[offset_k + t];
  }

  // Column
  int n = offset_n + t * 4;
  if (n >= size_n) return;

  // Find initial group
  int groupsize = size_k / groups;
  int group = offset_k / groupsize;
  int nextgroup = offset_k + groupsize;

  // b offset
  int qk = offset_k / (32 / 4);

  const uint32_t* b_ptr = b_q_weight + qk * size_n + n;

  // Initial zeros/scale
  int zeros[4];
  half2 scales[4];
  half2 z1z16[4][2];
  half2 y1y16[4][2];
  b_gptq_qzeros_.item4(zeros, group, n);
  b_gptq_scales_.item4_h2(scales, group, n);
  dequant_4bit_8_prep_zero(zeros[0] + zero_offset, z1z16[0], y1y16[0]);
  dequant_4bit_8_prep_zero(zeros[1] + zero_offset, z1z16[1], y1y16[1]);
  dequant_4bit_8_prep_zero(zeros[2] + zero_offset, z1z16[2], y1y16[2]);
  dequant_4bit_8_prep_zero(zeros[3] + zero_offset, z1z16[3], y1y16[3]);

  __syncthreads();

  int k = offset_k;
  int lk = 0;

  while (k < end_k) {
    if (k == nextgroup) {
      group++;
      nextgroup += groupsize;
      b_gptq_qzeros_.item4(zeros, group, n);
      b_gptq_scales_.item4_h2(scales, group, n);
      dequant_4bit_8_prep_zero(zeros[0] + zero_offset, z1z16[0], y1y16[0]);
      dequant_4bit_8_prep_zero(zeros[1] + zero_offset, z1z16[1], y1y16[1]);
      dequant_4bit_8_prep_zero(zeros[2] + zero_offset, z1z16[2], y1y16[2]);
      dequant_4bit_8_prep_zero(zeros[3] + zero_offset, z1z16[3], y1y16[3]);
    }

    for (int p = 0; p < 4; p++) {
      half2 dq[4][4];
      const int4* b_ptr4 = (int4*)b_ptr;
      int4 load_int4 = *b_ptr4;

      dequant_4bit_8_gptq(load_int4.x, dq[0], z1z16[0], y1y16[0], size_n,
                          false);
      dequant_4bit_8_gptq(load_int4.y, dq[1], z1z16[1], y1y16[1], size_n,
                          false);
      dequant_4bit_8_gptq(load_int4.z, dq[2], z1z16[2], y1y16[2], size_n,
                          false);
      dequant_4bit_8_gptq(load_int4.w, dq[3], z1z16[3], y1y16[3], size_n,
                          false);

      b_ptr += size_n;
      // half* dqh = (half*)dq;
      if (b_q_perm) {
        for (int j = 0; j < 4; j++) {
          for (int v = 0; v < 4; v++) dq[v][j] = __hmul2(scales[v], dq[v][j]);
          b_.set4(perm[lk++], n, __low2half(dq[0][j]), __low2half(dq[1][j]),
                  __low2half(dq[2][j]), __low2half(dq[3][j]));
          b_.set4(perm[lk++], n, __high2half(dq[0][j]), __high2half(dq[1][j]),
                  __high2half(dq[2][j]), __high2half(dq[3][j]));
        }
      } else {
        for (int j = 0; j < 4; j++) {
          for (int v = 0; v < 4; v++) dq[v][j] = __hmul2(scales[v], dq[v][j]);
          b_.set4(offset_k + lk++, n, __low2half(dq[0][j]),
                  __low2half(dq[1][j]), __low2half(dq[2][j]),
                  __low2half(dq[3][j]));
          b_.set4(offset_k + lk++, n, __high2half(dq[0][j]),
                  __high2half(dq[1][j]), __high2half(dq[2][j]),
                  __high2half(dq[3][j]));
        }
      }
    }
    k += 32;
  }
}

__global__ void reconstruct_exllama_3bit_kernel(
    const uint32_t* __restrict__ b_q_weight, const int* __restrict__ b_q_perm,
    const uint32_t* __restrict__ b_gptq_qzeros,
    const half* __restrict__ b_gptq_scales, const int size_k, const int size_n,
    const int groups, const bool use_v2_format, half* __restrict__ b) {
  MatrixView_half_rw b_(b, size_k, size_n);
  MatrixView_q3_row b_gptq_qzeros_(b_gptq_qzeros, groups, size_n);
  MatrixView_half b_gptq_scales_(b_gptq_scales, groups, size_n);

  // GPTQv2 and GPTQv1 handles zero points differently
  int zero_offset = use_v2_format ? 0 : 1;

  auto offset_k = BLOCK_KN_SIZE * blockIdx.y;
  auto offset_n = BLOCK_KN_SIZE * blockIdx.x * 4;

  int end_k = min(offset_k + BLOCK_KN_SIZE, size_k);

  // Preload remapping table
  __shared__ int perm[BLOCK_KN_SIZE];
  auto t = threadIdx.x;

  if (b_q_perm) {
    if (offset_k + t < size_k) perm[t] = b_q_perm[offset_k + t];
  }

  // Column
  int n = offset_n + t * 4;
  if (n >= size_n) return;

  // Find initial group
  int groupsize = size_k / groups;
  int group = offset_k / groupsize;
  int nextgroup = offset_k + groupsize;

  // b offset
  int qk = offset_k / 32 * 3;

  const uint32_t* b_ptr = b_q_weight + qk * size_n + n;

  // Initial zeros/scale
  int zeros[4];
  half2 scales[4];
  b_gptq_qzeros_.item4(zeros, group, n);
  b_gptq_scales_.item4_h2(scales, group, n);

  __syncthreads();

  int k = offset_k;
  int lk = 0;

  while (k < end_k) {
    if (k == nextgroup) {
      group++;
      nextgroup += groupsize;
      b_gptq_qzeros_.item4(zeros, group, n);
      b_gptq_scales_.item4_h2(scales, group, n);
    }

    for (int p = 0; p < 1; p++) {
      int4 load_int4[3];
      load_int4[0] = *((int4*)b_ptr);
      b_ptr += size_n;
      load_int4[1] = *((int4*)b_ptr);
      b_ptr += size_n;
      load_int4[2] = *((int4*)b_ptr);
      b_ptr += size_n;

      half2 dq[4][16];
      dequant_3bit_32(load_int4[0].x, load_int4[1].x, load_int4[2].x, dq[0],
                      size_n, zeros[0] + zero_offset);
      dequant_3bit_32(load_int4[0].y, load_int4[1].y, load_int4[2].y, dq[1],
                      size_n, zeros[1] + zero_offset);
      dequant_3bit_32(load_int4[0].z, load_int4[1].z, load_int4[2].z, dq[2],
                      size_n, zeros[2] + zero_offset);
      dequant_3bit_32(load_int4[0].w, load_int4[1].w, load_int4[2].w, dq[3],
                      size_n, zeros[3] + zero_offset);

      if (b_q_perm) {
        for (int j = 0; j < 16; j++) {
          for (int v = 0; v < 4; v++) dq[v][j] = __hmul2(scales[v], dq[v][j]);
          b_.set4(perm[lk++], n, __low2half(dq[0][j]), __low2half(dq[1][j]),
                  __low2half(dq[2][j]), __low2half(dq[3][j]));
          b_.set4(perm[lk++], n, __high2half(dq[0][j]), __high2half(dq[1][j]),
                  __high2half(dq[2][j]), __high2half(dq[3][j]));
        }
      } else {
        for (int j = 0; j < 16; j++) {
          for (int v = 0; v < 4; v++) dq[v][j] = __hmul2(scales[v], dq[v][j]);
          b_.set4(offset_k + lk++, n, __low2half(dq[0][j]),
                  __low2half(dq[1][j]), __low2half(dq[2][j]),
                  __low2half(dq[3][j]));
          b_.set4(offset_k + lk++, n, __high2half(dq[0][j]),
                  __high2half(dq[1][j]), __high2half(dq[2][j]),
                  __high2half(dq[3][j]));
        }
      }
    }
    k += 32;
  }
}

__global__ void reconstruct_exllama_2bit_kernel(
    const uint32_t* __restrict__ b_q_weight, const int* __restrict__ b_q_perm,
    const uint32_t* __restrict__ b_gptq_qzeros,
    const half* __restrict__ b_gptq_scales, const int size_k, const int size_n,
    const int groups, const bool use_v2_format, half* __restrict__ b) {
  MatrixView_half_rw b_(b, size_k, size_n);
  MatrixView_q2_row b_gptq_qzeros_(b_gptq_qzeros, groups, size_n);
  MatrixView_half b_gptq_scales_(b_gptq_scales, groups, size_n);

  // GPTQv2 and GPTQv1 handles zero points differently
  int zero_offset = use_v2_format ? 0 : 1;

  auto offset_k = BLOCK_KN_SIZE * blockIdx.y;
  auto offset_n = BLOCK_KN_SIZE * blockIdx.x * 4;

  int end_k = min(offset_k + BLOCK_KN_SIZE, size_k);

  // Preload remapping table
  __shared__ int perm[BLOCK_KN_SIZE];
  auto t = threadIdx.x;

  if (b_q_perm) {
    if (offset_k + t < size_k) perm[t] = b_q_perm[offset_k + t];
  }

  // Column
  int n = offset_n + t * 4;
  if (n >= size_n) return;

  // Find initial group
  int groupsize = size_k / groups;
  int group = offset_k / groupsize;
  int nextgroup = offset_k + groupsize;

  // b offset
  int qk = offset_k / (32 / 2);

  const uint32_t* b_ptr = b_q_weight + qk * size_n + n;

  // Initial zeros/scale
  int zeros[4];
  half2 scales[4];
  b_gptq_qzeros_.item4(zeros, group, n);
  b_gptq_scales_.item4_h2(scales, group, n);

  __syncthreads();

  int k = offset_k;
  int lk = 0;

  while (k < end_k) {
    if (k == nextgroup) {
      group++;
      nextgroup += groupsize;
      b_gptq_qzeros_.item4(zeros, group, n);
      b_gptq_scales_.item4_h2(scales, group, n);
    }

    for (int p = 0; p < 2; p++) {
      const int4* b_ptr4 = (int4*)b_ptr;
      int4 load_int4 = *b_ptr4;

      half2 dq[4][8];
      dequant_2bit_16(load_int4.x, dq[0], size_n, zeros[0] + zero_offset);
      dequant_2bit_16(load_int4.y, dq[1], size_n, zeros[1] + zero_offset);
      dequant_2bit_16(load_int4.z, dq[2], size_n, zeros[2] + zero_offset);
      dequant_2bit_16(load_int4.w, dq[3], size_n, zeros[3] + zero_offset);

      b_ptr += size_n;
      // half* dqh = (half*)dq;
      if (b_q_perm) {
        for (int j = 0; j < 8; j++) {
          for (int v = 0; v < 4; v++) dq[v][j] = __hmul2(scales[v], dq[v][j]);
          b_.set4(perm[lk++], n, __low2half(dq[0][j]), __low2half(dq[1][j]),
                  __low2half(dq[2][j]), __low2half(dq[3][j]));
          b_.set4(perm[lk++], n, __high2half(dq[0][j]), __high2half(dq[1][j]),
                  __high2half(dq[2][j]), __high2half(dq[3][j]));
        }
      } else {
        for (int j = 0; j < 8; j++) {
          for (int v = 0; v < 4; v++) dq[v][j] = __hmul2(scales[v], dq[v][j]);
          b_.set4(offset_k + lk++, n, __low2half(dq[0][j]),
                  __low2half(dq[1][j]), __low2half(dq[2][j]),
                  __low2half(dq[3][j]));
          b_.set4(offset_k + lk++, n, __high2half(dq[0][j]),
                  __high2half(dq[1][j]), __high2half(dq[2][j]),
                  __high2half(dq[3][j]));
        }
      }
    }
    k += 32;
  }
}

void reconstruct_exllama(const uint32_t* b_q_weight,
                         const uint32_t* b_gptq_qzeros,
                         const half* b_gptq_scales, const int* b_q_perm,
                         half* out, int height, int width, int groups,
                         bool use_v2_format, int bit) {
  dim3 blockDim, gridDim;
  blockDim.x = BLOCK_KN_SIZE;
  blockDim.y = 1;
  gridDim.y = DIVIDE(height, BLOCK_KN_SIZE);
  gridDim.x = DIVIDE(width, BLOCK_KN_SIZE);

  auto reconstruct_exllama_kernel = reconstruct_exllama_4bit_kernel;
  if (bit == 2) {
    reconstruct_exllama_kernel = reconstruct_exllama_2bit_kernel;
  } else if (bit == 3) {
    reconstruct_exllama_kernel = reconstruct_exllama_3bit_kernel;
  } else if (bit == 8) {
    reconstruct_exllama_kernel = reconstruct_exllama_8bit_kernel;
  }

  const cudaStream_t stream = get_current_cuda_stream();
  reconstruct_exllama_kernel<<<gridDim, blockDim, 0, stream>>>(
      b_q_weight, b_q_perm, b_gptq_qzeros, b_gptq_scales, height, width, groups,
      use_v2_format, out);
}

__global__ void gemm_half_q_half_alt_4bit_kernel(
    const half2* __restrict__ vec, const uint32_t* __restrict__ mat,
    half* __restrict__ mul, const half* __restrict__ scales,
    const uint32_t* __restrict__ zeros, const int* __restrict__ g_idx,
    int batch, int height, int width, bool use_v2_format) {
  int zero_width = width / 8;
  int vec_height = height * 4;
  const int blockwidth2 = BLOCK_KN_SIZE / 2;
  auto b = blockIdx.y * BLOCK_M_SIZE_MAX;
  int b_end = min(BLOCK_M_SIZE_MAX, batch - b);
  auto h = BLOCK_KN_SIZE * blockIdx.z / 8;
  int h_end = min(BLOCK_KN_SIZE / 8, height - h) * 4;
  auto w = BLOCK_KN_SIZE * blockIdx.x + threadIdx.x;

  // GPTQv2 and GPTQv1 handles zero points differently
  int zero_offset = use_v2_format ? 0 : 1;

  __shared__ half2 blockvec[BLOCK_M_SIZE_MAX][blockwidth2];
  if (threadIdx.x < h_end) {
    for (int m = 0; m < b_end; ++m) {
      blockvec[m][threadIdx.x] =
          vec[(m + b) * vec_height + blockIdx.z * BLOCK_KN_SIZE / 2 +
              threadIdx.x];
    }
  }

  __shared__ half2 deq2[256][8];
  auto val = threadIdx.x / 8;
  auto off = threadIdx.x % 8;
  for (; val < 256; val += BLOCK_KN_SIZE / 8) {
    deq2[val][off] =
        __halves2half2(__int2half_rn(val & 0xF), __int2half_rn(val >> 4));
  }

  __syncthreads();

  int i = width * h + w;
  int g_h = h * 8;
  int k = 0;
  int z_w = w / 8;
  int z_mod = (w % 8) * 4;
  half2 res2;
  half res[BLOCK_M_SIZE_MAX] = {};

  unsigned int tmp;
  while (k < h_end) {
    tmp = mat[i];
    half2 scales_tmp[4];
    half2 zeros_tmp[4];
    for (int tmp_k = 0; tmp_k < 4; tmp_k++) {
      int g = g_idx[g_h + (k + tmp_k) * 2];
      int g2 = g_idx[g_h + (k + tmp_k) * 2 + 1];
      half scale_f = scales[g * width + w];
      half scale_f2 = scales[g2 * width + w];
      half2 scale = __halves2half2(scale_f, scale_f2);
      half2 zero = __halves2half2(
          __hmul(scale_f,
                 __int2half_rn(-((zeros[g * zero_width + z_w] >> z_mod) & 0xF) -
                               zero_offset)),
          __hmul(
              scale_f2,
              __int2half_rn(-((zeros[g2 * zero_width + z_w] >> z_mod) & 0xF) -
                            zero_offset)));
      scales_tmp[tmp_k] = scale;
      zeros_tmp[tmp_k] = zero;
    }
    for (int m = 0; m < b_end; m++) {
#ifndef USE_ROCM
      res2 = {};
#else
      res2.x = __half_as_ushort(__float2half(0));
      res2.y = __half_as_ushort(__float2half(0));
#endif
      res2 = __hfma2(
          __hfma2(deq2[(tmp >> 0) & 0xff][off], scales_tmp[0], zeros_tmp[0]),
          blockvec[m][k + 0], res2);
      res2 = __hfma2(
          __hfma2(deq2[(tmp >> 8) & 0xff][off], scales_tmp[1], zeros_tmp[1]),
          blockvec[m][k + 1], res2);
      res2 = __hfma2(
          __hfma2(deq2[(tmp >> 16) & 0xff][off], scales_tmp[2], zeros_tmp[2]),
          blockvec[m][k + 2], res2);
      res2 = __hfma2(
          __hfma2(deq2[(tmp >> 24) & 0xff][off], scales_tmp[3], zeros_tmp[3]),
          blockvec[m][k + 3], res2);
#ifndef USE_ROCM
      res[m] = __hadd(res[m], __hadd(res2.x, res2.y));
#else
      res[m] = __hadd(
          res[m], __hadd(__ushort_as_half(res2.x), __ushort_as_half(res2.y)));
#endif
    }
    i += width;
    k += 4;
  }
  for (int m = 0; m < b_end; m++) {
    atomicAdd(&mul[(b + m) * width + w], res[m]);
  }
}

__global__ void gemm_half_q_half_alt_8bit_kernel(
    const half2* __restrict__ vec, const uint32_t* __restrict__ mat,
    half* __restrict__ mul, const half* __restrict__ scales,
    const uint32_t* __restrict__ zeros, const int* __restrict__ g_idx,
    int batch, int height, int width, bool use_v2_format) {
  int zero_width = width / 4;
  int vec_height = height * 2;
  const int blockwidth2 = BLOCK_KN_SIZE / 2;
  auto b = blockIdx.y * BLOCK_M_SIZE_MAX;
  int b_end = min(BLOCK_M_SIZE_MAX, batch - b);
  auto h = BLOCK_KN_SIZE * blockIdx.z / 4;
  int h_end = min(BLOCK_KN_SIZE / 4, height - h) * 2;
  auto w = BLOCK_KN_SIZE * blockIdx.x + threadIdx.x;

  // GPTQv2 and GPTQv1 handles zero points differently
  int zero_offset = use_v2_format ? 0 : 1;

  __shared__ half2 blockvec[BLOCK_M_SIZE_MAX][blockwidth2];
  if (threadIdx.x < h_end) {
    for (int m = 0; m < b_end; ++m) {
      blockvec[m][threadIdx.x] =
          vec[(m + b) * vec_height + blockIdx.z * BLOCK_KN_SIZE / 2 +
              threadIdx.x];
    }
  }

  __syncthreads();

  int i = width * h + w;
  int g_h = h * 4;
  int k = 0;
  int z_w = w / 4;
  int z_mod = (w % 4) * 8;
  half2 res2;
  half res[BLOCK_M_SIZE_MAX] = {};

  unsigned int tmp;
  while (k < h_end) {
    tmp = mat[i];
    half2 scales_tmp[2];
    half2 zeros_tmp[2];
    for (int tmp_k = 0; tmp_k < 2; tmp_k++) {
      int g = g_idx[g_h + (k + tmp_k) * 2];
      int g2 = g_idx[g_h + (k + tmp_k) * 2 + 1];
      half scale_f = scales[g * width + w];
      half scale_f2 = scales[g2 * width + w];
      half2 scale = __halves2half2(scale_f, scale_f2);
      half2 zero = __halves2half2(
          __hmul(scale_f, __int2half_rn(
                              -((zeros[g * zero_width + z_w] >> z_mod) & 0xff) -
                              zero_offset)),
          __hmul(
              scale_f2,
              __int2half_rn(-((zeros[g2 * zero_width + z_w] >> z_mod) & 0xff) -
                            zero_offset)));
      scales_tmp[tmp_k] = scale;
      zeros_tmp[tmp_k] = zero;
    }
    for (int m = 0; m < b_end; m++) {
#ifndef USE_ROCM
      res2 = {};
#else
      res2.x = __half_as_ushort(__float2half(0));
      res2.y = __half_as_ushort(__float2half(0));
#endif
      half2 v12 = __halves2half2(__int2half_rn(tmp & 0xFF),
                                 __int2half_rn((tmp >> 8) & 0xFF));
      res2 = __hfma2(__hfma2(v12, scales_tmp[0], zeros_tmp[0]),
                     blockvec[m][k + 0], res2);
      half2 v34 = __halves2half2(__int2half_rn((tmp >> 16) & 0xFF),
                                 __int2half_rn((tmp >> 24) & 0xFF));
      res2 = __hfma2(__hfma2(v34, scales_tmp[1], zeros_tmp[1]),
                     blockvec[m][k + 1], res2);
#ifndef USE_ROCM
      res[m] = __hadd(res[m], __hadd(res2.x, res2.y));
#else
      res[m] = __hadd(
          res[m], __hadd(__ushort_as_half(res2.x), __ushort_as_half(res2.y)));
#endif
    }
    i += width;
    k += 2;
  }
  for (int m = 0; m < b_end; m++) {
    atomicAdd(&mul[(b + m) * width + w], res[m]);
  }
}

void gemm_half_q_half_alt(const half* a, const uint32_t* b_q_weight,
                          const uint32_t* b_gptq_qzeros,
                          const half* b_gptq_scales, const int* b_g_idx,
                          half* c, int size_m, int size_n, int size_k,
                          bool use_v2_format, int bit) {
  dim3 blockDim, gridDim;
  blockDim.x = BLOCK_KN_SIZE;
  blockDim.y = 1;
  blockDim.z = 1;
  gridDim.x = DIVIDE(size_n, BLOCK_KN_SIZE);
  gridDim.y = DIVIDE(size_m, BLOCK_M_SIZE_MAX);
  gridDim.z = DIVIDE(size_k, BLOCK_KN_SIZE);

  auto kernel = gemm_half_q_half_alt_4bit_kernel;
  if (bit == 8) {
    kernel = gemm_half_q_half_alt_8bit_kernel;
  }

  const cudaStream_t stream = get_current_cuda_stream();
  kernel<<<gridDim, blockDim, 0, stream>>>(
      (const half2*)a, b_q_weight, c, b_gptq_scales, b_gptq_qzeros, b_g_idx,
      size_m, size_k / 32 * bit, size_n, use_v2_format);
}

template <class T, int bit>
__global__ void reconstruct_gptq_kernel(
    const uint32_t* __restrict__ w, const half* __restrict__ w_scales,
    const uint32_t* __restrict__ w_zeros, const int* __restrict__ g_idx,
    const int height, const int width, const int group,
    const bool use_v2_format, half* __restrict__ out) {
  // Start of block

  auto column = BLOCK_KN_SIZE * blockIdx.x + threadIdx.x;
  auto row = blockIdx.y * 32 / bit;
  if (column >= width) return;

  // Views

  MatrixView_half_rw out_(out, height, width);
  MatrixView_half w_scales_(w_scales, group, width);
  T w_zeros_(w_zeros, group, width);

  // GPTQv2 and GPTQv1 handles zero points differently
  int zero_offset = use_v2_format ? 0 : 1;

  uint32_t w_read = w[blockIdx.y * width + column];
  half* out_ptr = out_.item_ptr(row, column);

#pragma unroll
  for (int s = 0; s < 32; s += bit) {
    int group = g_idx[row + s / bit];
    half w_scale = w_scales_.item(group, column);
    uint32_t w_zero = w_zeros_.item(group, column) + zero_offset;
    half w_item =
        __hmul(__int2half_rn((int)((w_read >> s) & ((1 << bit) - 1)) - w_zero),
               w_scale);
    *out_ptr = w_item;
    out_ptr += out_.width;
  }
}

__global__ void reconstruct_gptq_3bit_kernel(
    const uint32_t* __restrict__ w, const half* __restrict__ w_scales,
    const uint32_t* __restrict__ w_zeros, const int* __restrict__ g_idx,
    const int height, const int width, const int group,
    const bool use_v2_format, half* __restrict__ out) {
  // Start of block
  auto column = BLOCK_KN_SIZE * blockIdx.x + threadIdx.x;
  auto row = blockIdx.y * 32;
  if (column >= width) return;

  // Views

  MatrixView_half_rw out_(out, height, width);
  MatrixView_half w_scales_(w_scales, group, width);
  MatrixView_q3_row w_zeros_(w_zeros, group, width);

  // GPTQv2 and GPTQv1 handles zero points differently
  int zero_offset = use_v2_format ? 0 : 1;

  uint32_t w1 = w[(blockIdx.y * 3) * width + column];
  uint32_t w2 = w[(blockIdx.y * 3 + 1) * width + column];
  uint32_t w3 = w[(blockIdx.y * 3 + 2) * width + column];
  half* out_ptr = out_.item_ptr(row, column);

#pragma unroll
  for (int i = 0; i < 32; i += 1) {
    int group = g_idx[row + i];
    half w_scale = w_scales_.item(group, column);
    uint32_t w_zero = w_zeros_.item(group, column) + zero_offset;
    int w_item;
    if (i == 10) {
      w_item = (w1 >> 30) | ((w2 << 2) & 0x4);
    } else if (i == 21) {
      w_item = (w2 >> 31) | ((w3 << 1) & 0x6);
    } else if (i < 10) {
      w_item = ((w1 >> (i * 3)) & 0x7);
    } else if (i < 21) {
      w_item = ((w2 >> (i * 3 - 32)) & 0x7);
    } else {
      w_item = ((w3 >> (i * 3 - 64)) & 0x7);
    }
    *out_ptr = __hmul(__int2half_rn(w_item - w_zero), w_scale);
    out_ptr += out_.width;
  }
}

void reconstruct_gptq(const uint32_t* b_q_weight, const uint32_t* b_gptq_qzeros,
                      const half* b_gptq_scales, const int* b_g_idx, half* out,
                      int height, int width, int groups, bool use_v2_format,
                      int bit) {
  dim3 blockDim, gridDim;
  blockDim.x = BLOCK_KN_SIZE;
  blockDim.y = 1;
  gridDim.y = DIVIDE(height, 32 / bit);
  gridDim.x = DIVIDE(width, BLOCK_KN_SIZE);

  auto kernel = reconstruct_gptq_kernel<MatrixView_q4_row, 4>;
  if (bit == 2) {
    kernel = reconstruct_gptq_kernel<MatrixView_q2_row, 2>;
  } else if (bit == 8) {
    kernel = reconstruct_gptq_kernel<MatrixView_q8_row, 8>;
  } else if (bit == 3) {
    kernel = reconstruct_gptq_3bit_kernel;
    gridDim.y = DIVIDE(height, 32);
  }

  const cudaStream_t stream = get_current_cuda_stream();
  kernel<<<gridDim, blockDim, 0, stream>>>(b_q_weight, b_gptq_scales,
                                           b_gptq_qzeros, b_g_idx, height,
                                           width, groups, use_v2_format, out);
}

// [fa2_sm70] THE SAME PREDICATE, ONE PLACE. gptq_gemm() below must know which branch
// gemm_half_q_half_cuda() will take, because only the branch matters for whether C has
// to arrive zeroed:
//   * reconstruct  -> cublasHgemm with beta = 0, which does NOT read C: zeroing is DEAD;
//   * quantized/alt -> the kernels accumulate with atomicAdd over a blockIdx.z split of k
//                     (gridDim.z = ceil(size_k / BLOCK_KN_SIZE)), so C is the accumulator
//                     and MUST start at zero.
// Factored out rather than copied so the two cannot drift apart.
inline bool gptq_gemm_uses_reconstruct(int size_m, bool use_exllama, int bit) {
  if (use_exllama) {
    return ((bit == 8 && size_m > MAX_Q_GEMM_ROWS_8BIT) ||
            (bit != 8 && size_m > MAX_Q_GEMM_ROWS));
  }
  // The 2/3-bit kernels are somehow slower than dequant + gemm baseline, so
  // we disabled them for now.
  return (bit < 4 || size_m > MAX_ALT_GEMM_ROWS);
}

void gemm_half_q_half_cuda(cublasHandle_t cublas_handle, const half* a,
                           const uint32_t* b_q_weight,
                           const uint32_t* b_gptq_qzeros,
                           const half* b_gptq_scales, const int* b_g_idx,
                           half* c, half* temp_dq, int size_m, int size_n,
                           int size_k, int groups, bool use_exllama,
                           bool use_v2_format, int bit) {
  const bool use_reconstruct =
      gptq_gemm_uses_reconstruct(size_m, use_exllama, bit);
  if (use_reconstruct) {
    // Reconstruct FP16 matrix, then cuBLAS
    if (use_exllama) {
      reconstruct_exllama(b_q_weight, b_gptq_qzeros, b_gptq_scales, b_g_idx,
                          temp_dq, size_k, size_n, groups, use_v2_format, bit);
    } else {
      reconstruct_gptq(b_q_weight, b_gptq_qzeros, b_gptq_scales, b_g_idx,
                       temp_dq, size_k, size_n, groups, use_v2_format, bit);
    }

    const half alpha = __float2half(1.0f);
    const half beta = __float2half(0.0f);
    cublasHgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, size_n, size_m, size_k,
                &alpha, temp_dq, size_n, a, size_k, &beta, c, size_n);
  } else if (use_exllama) {
    // Quantized matmul
    int max_chunks = size_m / BLOCK_M_SIZE_MAX;
    int last_chunk = max_chunks * BLOCK_M_SIZE_MAX;
    int last_chunk_size = size_m - last_chunk;

    if (max_chunks) {
      gemm_half_q_half_cuda_part(a, b_q_weight, b_gptq_qzeros, b_gptq_scales,
                                 b_g_idx, c, last_chunk, size_n, size_k,
                                 BLOCK_M_SIZE_MAX, groups, use_v2_format, bit);
    }

    if (last_chunk_size) {
      gemm_half_q_half_cuda_part(
          a + last_chunk * size_k, b_q_weight, b_gptq_qzeros, b_gptq_scales,
          b_g_idx, c + last_chunk * size_n, last_chunk_size, size_n, size_k,
          last_chunk_size, groups, use_v2_format, bit);
    }
  } else {
    gemm_half_q_half_alt(a, b_q_weight, b_gptq_qzeros, b_gptq_scales, b_g_idx,
                         c, size_m, size_n, size_k, use_v2_format, bit);
  }
}

__global__ void shuffle_4bit_kernel(uint32_t* __restrict__ b_q_weight,
                                    const int size_k, const int size_n) {
  auto n = blockIdx.x * THREADS_X + threadIdx.x;
  if (n >= size_n) return;
  int k = 0;
  uint32_t* b_ptr = b_q_weight + n;
  while (k < size_k) {
    shuffle_4bit_8(b_ptr, size_n);
    b_ptr += 1 * size_n;
    k += 8;
  }
}

__global__ void shuffle_8bit_kernel(uint32_t* __restrict__ b_q_weight,
                                    const int size_k, const int size_n) {
  auto n = blockIdx.x * THREADS_X + threadIdx.x;
  if (n >= size_n) return;
  int k = 0;
  uint32_t* b_ptr = b_q_weight + n;
  while (k < size_k) {
    shuffle_8bit_4(b_ptr, size_n);
    b_ptr += 1 * size_n;
    k += 4;
  }
}

__global__ void shuffle_2bit_kernel(uint32_t* __restrict__ b_q_weight,
                                    const int size_k, const int size_n) {
  auto n = blockIdx.x * THREADS_X + threadIdx.x;
  if (n >= size_n) return;
  int k = 0;
  uint32_t* b_ptr = b_q_weight + n;
  while (k < size_k) {
    shuffle_2bit_16(b_ptr, size_n);
    b_ptr += 1 * size_n;
    k += 16;
  }
}

__global__ void shuffle_3bit_kernel(uint32_t* __restrict__ b_q_weight,
                                    const int size_k, const int size_n) {
  auto n = blockIdx.x * THREADS_X + threadIdx.x;
  if (n >= size_n) return;
  int k = 0;
  uint32_t* b_ptr = b_q_weight + n;
  while (k < size_k) {
    shuffle_3bit_32(b_ptr, size_n);
    b_ptr += 3 * size_n;
    k += 32;
  }
}

__global__ void make_sequential_4bit_kernel(const uint32_t* __restrict__ w,
                                            uint32_t* __restrict__ w_new,
                                            const int* __restrict__ q_perm,
                                            const int w_width) {
  const uint64_t* w2 = (uint64_t*)w;
  uint64_t* w_new2 = (uint64_t*)w_new;
  int w2_stride = w_width >> 1;
  auto w2_column = THREADS_X * blockIdx.x + threadIdx.x;
  if (w2_column >= w2_stride) return;
  auto w_new2_row = blockIdx.y;
  int q_perm_idx = w_new2_row << 3;
  uint64_t dst = 0;

#pragma unroll
  for (int i = 0; i < 8; i++) {
    int source_row = q_perm[q_perm_idx++];

    int w2_row = source_row >> 3;
    int w2_subrow = source_row & 0x07;
    int w2_row_shift = w2_subrow << 2;
    int wnew2_row_shift = i << 2;

    uint64_t src = w2[w2_row * w2_stride + w2_column];
    src >>= w2_row_shift;
    src &= 0x0000000f0000000f;
    src <<= wnew2_row_shift;
    dst |= src;
  }
  w_new2[w_new2_row * w2_stride + w2_column] = dst;
}

__global__ void make_sequential_2bit_kernel(const uint32_t* __restrict__ w,
                                            uint32_t* __restrict__ w_new,
                                            const int* __restrict__ q_perm,
                                            const int w_width) {
  const uint64_t* w2 = (uint64_t*)w;
  uint64_t* w_new2 = (uint64_t*)w_new;
  int w2_stride = w_width >> 1;
  auto w2_column = THREADS_X * blockIdx.x + threadIdx.x;
  if (w2_column >= w2_stride) return;
  auto w_new2_row = blockIdx.y;
  int q_perm_idx = w_new2_row << 4;
  uint64_t dst = 0;

#pragma unroll
  for (int i = 0; i < 16; i++) {
    int source_row = q_perm[q_perm_idx++];

    int w2_row = source_row >> 4;
    int w2_subrow = source_row & 0x0f;
    int w2_row_shift = w2_subrow << 1;
    int wnew2_row_shift = i << 1;

    uint64_t src = w2[w2_row * w2_stride + w2_column];
    src >>= w2_row_shift;
    src &= 0x0000000300000003;
    src <<= wnew2_row_shift;
    dst |= src;
  }
  w_new2[w_new2_row * w2_stride + w2_column] = dst;
}

__global__ void make_sequential_3bit_kernel(const uint32_t* __restrict__ w,
                                            uint32_t* __restrict__ w_new,
                                            const int* __restrict__ q_perm,
                                            const int w_width) {
  auto w_column = THREADS_X * blockIdx.x + threadIdx.x;
  if (w_column >= w_width) return;
  auto w_new_row = blockIdx.y * 3;
  auto q_perm_idx = blockIdx.y << 5;
  uint32_t dst[3] = {0, 0, 0};

#pragma unroll
  for (int i = 0; i < 32; i++) {
    int source_row = q_perm[q_perm_idx++];
    int z_w = (source_row / 32) * 3;
    int z_mod = source_row % 32;
    int z_bit;

    if (z_mod != 10) {
      if (z_mod != 21) {
        z_bit = z_mod;
        if (z_bit > 21) {
          z_bit *= 3;
          z_bit -= 64;
          z_w += 2;
        } else if (z_bit > 10) {
          z_bit *= 3;
          z_bit -= 32;
          z_w += 1;
        } else {
          z_bit *= 3;
        }
      } else {
        z_w += 1;
      }
    }

    uint64_t src;
    if (z_mod == 10) {
      src = (w[z_w * w_width + w_column] >> 30) |
            ((w[(z_w + 1) * w_width + w_column] << 2) & 0x4);
    } else if (z_mod == 21) {
      src = (w[z_w * w_width + w_column] >> 31) |
            ((w[(z_w + 1) * w_width + w_column] << 1) & 0x6);
    } else {
      src = w[z_w * w_width + w_column];
      src >>= z_bit;
      src &= 0x07;
    }

    z_w = 0;
    if (i != 10) {
      if (i != 21) {
        z_bit = i;
        if (z_bit > 21) {
          z_bit *= 3;
          z_bit -= 64;
          z_w += 2;
        } else if (z_bit > 10) {
          z_bit *= 3;
          z_bit -= 32;
          z_w += 1;
        } else {
          z_bit *= 3;
        }
      } else {
        z_w += 1;
      }
    }
    if (i == 10) {
      dst[z_w] |= (src & 0x03) << 30;
      dst[z_w + 1] |= ((src & 0x4) >> 2);
    } else if (i == 21) {
      dst[z_w] |= (src & 0x01) << 31;
      dst[z_w + 1] |= ((src & 0x6) >> 1);
    } else {
      dst[z_w] |= (src << z_bit);
    }
  }
  w_new[w_new_row * w_width + w_column] = dst[0];
  w_new[(w_new_row + 1) * w_width + w_column] = dst[1];
  w_new[(w_new_row + 2) * w_width + w_column] = dst[2];
}

__global__ void make_sequential_8bit_kernel(const uint32_t* __restrict__ w,
                                            uint32_t* __restrict__ w_new,
                                            const int* __restrict__ q_perm,
                                            const int w_width) {
  const uint64_t* w2 = (uint64_t*)w;
  uint64_t* w_new2 = (uint64_t*)w_new;
  int w2_stride = w_width >> 1;
  auto w2_column = THREADS_X * blockIdx.x + threadIdx.x;
  if (w2_column >= w2_stride) return;
  auto w_new2_row = blockIdx.y;
  int q_perm_idx = w_new2_row << 2;
  uint64_t dst = 0;

#pragma unroll
  for (int i = 0; i < 4; i++) {
    int source_row = q_perm[q_perm_idx++];

    int w2_row = source_row >> 2;
    int w2_subrow = source_row & 0x03;
    int w2_row_shift = w2_subrow << 3;
    int wnew2_row_shift = i << 3;

    uint64_t src = w2[w2_row * w2_stride + w2_column];
    src >>= w2_row_shift;
    src &= 0x000000ff000000ff;
    src <<= wnew2_row_shift;
    dst |= src;
  }
  w_new2[w_new2_row * w2_stride + w2_column] = dst;
}

void shuffle_exllama_weight(uint32_t* q_weight, int* q_perm, int height,
                            int width, int bit) {
  if (q_perm) {
    uint32_t* new_qweight = NULL;
    cudaMalloc(&new_qweight, height / 32 * bit * width * sizeof(uint32_t));

    dim3 blockDim, gridDim;
    blockDim.x = THREADS_X;
    blockDim.y = 1;
    gridDim.x = DIVIDE(width, THREADS_X);
    gridDim.y = height / 32 * bit;

    auto kernel = make_sequential_4bit_kernel;
    if (bit == 2) {
      kernel = make_sequential_2bit_kernel;
    } else if (bit == 3) {
      kernel = make_sequential_3bit_kernel;
      gridDim.y = height / 32;
    } else if (bit == 8) {
      kernel = make_sequential_8bit_kernel;
    }
    const cudaStream_t stream = get_current_cuda_stream();
    kernel<<<gridDim, blockDim, 0, stream>>>(q_weight, new_qweight, q_perm,
                                             width);
    // Replace qweights
    cudaMemcpyAsync(q_weight, new_qweight,
                    height / 32 * bit * width * sizeof(uint32_t),
                    cudaMemcpyDeviceToDevice);
    // Cleanup
    cudaDeviceSynchronize();
    cudaFree(new_qweight);
  }
  dim3 blockDim, gridDim;
  blockDim.x = THREADS_X;
  blockDim.y = 1;
  gridDim.x = DIVIDE(width, THREADS_X);
  gridDim.y = 1;
  auto shuffle_kernel = shuffle_4bit_kernel;
  if (bit == 2) {
    shuffle_kernel = shuffle_2bit_kernel;
  } else if (bit == 3) {
    shuffle_kernel = shuffle_3bit_kernel;
  } else if (bit == 8) {
    shuffle_kernel = shuffle_8bit_kernel;
  }
  const cudaStream_t stream = get_current_cuda_stream();
  shuffle_kernel<<<gridDim, blockDim, 0, stream>>>(q_weight, height, width);
}

// =================== [fa2_sm70 patch] СВЁРТКА ОБНУЛЕНИЙ GPTQ ==================
// [fa2_sm70 patch] ПАРА-1 + задача #92. Правка целиком наша, в ванильном vLLM её нет.
// Снимает 236 ядер-заливок C на декодный токен, заменяя их ОДНИМ узлом графа, и
// закрывает две мины, каждая из которых даёт неверные числа БЕЗ падения (подробности
// и инварианты -- ниже, разбор "ЕДИНИЦА -- ПРОХОД, А НЕ ГРАФ").
// ============================ [fa2_sm70 / ПАРА-1] ============================
// ПАССАЖИР ОБНУЛЕНИЯ: N ядер-заливок на токен -> ОДИН узел графа.
// (На боевой модели Qwen3.6-27B-INT8 при TP=2 таких умножений ровно 236 на токен:
//  16 полных слоёв x 3 (o_proj, gate_up, down) + 47 линейных x 4 (in_proj_qkvz,
//  out_proj, gate_up, down). Слой 0 не квантован, qkv_proj и in_proj_ba исключены
//  правилом dynamic в quantization_config.)
//
// ЗАЧЕМ. На декодном пути (size_m <= 24 при 8 битах) ядро gemm_half_q_half_gptq_*
// накапливает в C через atomicAdd по расщеплению k (gridDim.z = size_k/128), поэтому C
// ОБЯЗАН приходить нулевым -- убрать нули нельзя. Но по трассе боевого декода каждая
// такая заливка стоит 1.888 мкс ядра + 0.951 мкс зазора между узлами графа при полезной
// работе 2.5-17 КБ: это 99% ЧИСТАЯ ЦЕНА СУЩЕСТВОВАНИЯ ЗАПУСКА.
//
// КАК. Выходы декодных умножений берутся не у аллокатора, а из ПОСТОЯННОЙ плиты
// (cudaMalloc вне графа). Внутри одного прохода смещения детерминированы, поэтому в
// графе они зафиксированы. Вся плита зануляется ОДНИМ cudaMemsetAsync на первом
// умножении токена: к этому моменту выходы ПРЕДЫДУЩЕГО токена уже прочитаны и мертвы.
//
// ================== ЕДИНИЦА -- ПРОХОД, А НЕ ГРАФ (задача #92) ================
// Здесь ровно одна нетривиальная мысль, и она куплена двумя ошибками подряд.
//
// ЕДИНИЦА, на которой обязан стоять один узел обнуления, -- это ПРОХОД (токен), то
// есть весь набор из E умножений одного шага модели. Она НЕ совпадает ни с флагом
// "мы внутри захвата", ни с идентификатором захваченного графа:
//
//  * ОШИБКА 1 (была отгружена, спала). Границу прохода определяли флагом in_capture,
//    снимавшимся только при вызове ВНЕ захвата. Работает лишь потому, что vLLM
//    сегодня успевает сделать между двумя захватами хотя бы одно умножение вне
//    графа. Захвати движок два графа подряд -- второй проход продолжил бы плиту
//    ПРЕЖНЕГО и не получил бы узла обнуления вовсе: тихо неверные числа, без падения.
//
//  * ОШИБКА 2 (первое лечение, было бы регрессом). Границу взяли у CUDA:
//    cudaStreamGetCaptureInfo даёт уникальный id захвата, сменился id -> новый проход.
//    Но при cudagraph_mode=full_and_piecewise ОДИН проход захватывается ДЕСЯТКАМИ
//    графов (замерено на живой сетке: 48 захватов по 4 слота на проход). Тогда узел
//    обнуления ВСЕЙ плиты ставится в КАЖДЫЙ кусок: 48 x 8 МБ = 384 МБ записей на
//    токен вместо 8 МБ. Числа верные, а весь выигрыш механизма (0.51 мс/токен) съеден
//    -- отказ, который не падает и не врёт, а просто отменяет сам себя.
//
// ЛЕЧЕНИЕ. Границу прохода задаёт САМ ПРОХОД: последовательность выходных ширин n
// умножений в проходе детерминирована моделью и одинакова от токена к токену
// (m у всех умножений прохода тоже один -- это число токенов пакета). Поэтому:
//   E        -- длина прохода в умножениях (закрепляется VLLM_GPTQ_FOLD_EXPECT=236
//               либо выучивается на первом проходе);
//   ref[k]   -- эталонная ширина k-го умножения прохода.
// Проход закрывается ровно на E-м умножении, следующее умножение начинает новый ->
// сброс плиты + новый узел обнуления. Разбиение на куски при этом НЕ мешает: куски
// продолжают проход, узел обнуления один и лежит в ПЕРВОМ куске.
//
// ИНВАРИАНТЫ. Нарушение любого -- ОТКАЗ (STD_TORCH_CHECK), а не тихий откат: тихий
// откат здесь и есть тот дефект, из-за которого сервер отвечает неверными числами.
//   (I1) k-е умножение прохода обязано иметь ширину ref[k], а m -- совпадать с m
//        прохода. Это ловит всё сразу: два прохода, слипшихся в один (ошибка 1);
//        проход, потерявший умножения; чужую модель, вклинившуюся в тот же процесс.
//   (I2) узел обнуления ставится ровно один раз за проход, на его первом умножении.
//   (I3) слот обязан влезть в плиту. НЕ нарушение: не влез -- честный откат на
//        new_zeros (численно верно, просто без выигрыша), но откат СЧИТАЕТСЯ и
//        занимает свою позицию k, чтобы последовательность не поехала.
//   (I4) плита живёт на ОДНОМ устройстве. Вызов с чужого устройства -- откат
//        (тоже численно верный), а не свёртка по чужому указателю.
//   (I5) выученная последовательность не имеет периода меньше своей длины. Это
//        единственная защита обучения: если E не закреплён, а первые два прохода
//        слиплись, выучилось бы E=2*236 с периодом 236 -- и слипание стало бы
//        "нормой". Периодичность ловит это на месте.
//        (Закреплённый E защищён иначе: неверное значение ловится на ВТОРОМ проходе
//         сверкой ширин, потому что наименьший период боевой последовательности
//         равен её длине -- 16 слоёв по 3 умножения, затем 47 по 4.)
//
// ПОЧЕМУ НЕ СБРОС С ХОСТА (рассмотрено и отклонено). Честнее всего выглядит явный
// вызов "начался проход" из шима: он даёт правильную единицу и в разбиении, и в целом
// графе. Но: (а) он обязан быть врезан в КАЖДЫЙ путь исполнения (боевой шаг,
// профилировочный прогон, захват, черновая модель спекулятивного декода), и пропуск
// одного пути -- это НОВЫЙ отказ того же класса "молча неверно", только теперь на
// стороне питона, где из ядра его не видно; (б) он ничего не доказывает: ядру всё
// равно нужен инвариант, чтобы поймать пропущенный вызов. Инвариант (I1) ловит
// пропуск границы САМ, без хоста, и работает даже если шим вообще не наш. Поэтому
// граница выводится из данных прохода, а хост не участвует.
namespace {
struct ZeroSlab {
  void* base = nullptr;
  size_t bytes = 0;  // размер плиты И размер обнуления в начале прохода
  size_t bump = 0;   // указатель внутри ТЕКУЩЕГО прохода
  int device = -1;   // (I4) на каком устройстве живёт плита
  bool tried = false;

  // --- состояние ТЕКУЩЕГО прохода ---
  int pos = 0;        // сколько умножений прохода уже прошло (слоты + откаты)
  int slots = 0;      // из них свёрнуто в плиту
  int fb_cap = 0;     // из них откачено по ёмкости (I3)
  int m_cur = 0;      // m прохода (одинаков у всех его умножений)
  bool open = false;  // проход начат
  bool zeroed = false;                // (I2) узел обнуления этого прохода поставлен
  unsigned long long graph_zero = 0;  // id графа, куда лёг узел обнуления
  unsigned long long zeroed_graph_prev = 0;  // то же у ПРЕДЫДУЩЕГО прохода
  unsigned long long graph_last = 0;         // id последнего виденного графа
  int pieces = 0;  // сколько РАЗНЫХ графов в проходе (диагностика разбиения)
  size_t zero_bytes = 0;  // сколько байт зануляет узел обнуления ЭТОГО прохода
  size_t zeroed_now = 0;  // сколько занулено фактически (для печати)
  int zero_m = -1;        // для какого m посчитано zero_bytes (кэш на проход)

  // --- эталон прохода ---
  std::vector<int> ref;  // ширины n по позициям
  bool ref_ready = false;
  int expect = -1;  // E: -1 = ещё не известен
  bool pinned = false;

  // --- накопительное ---
  long long passes = 0;
  long long folded = 0;
  long long fb_device = 0;  // (I4) откаты по чужому устройству
  bool warned_device = false;
  bool warned_same_graph = false;
  bool warned_observe = false;
  bool in_capture = false;  // прежний (ненадёжный) признак -- только для LEGACY
};
ZeroSlab g_zslab;

// Идентификатор текущего захвата (false = поток не захватывает). ОТКАЗ, если CUDA не
// смогла ответить: без него не отличить "внутри графа" от "вне", а это разные правила
// жизни плиты.
bool fold_capture_id(cudaStream_t stream, unsigned long long& id) {
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  unsigned long long raw_id = 0;
  cudaError_t err = cudaStreamGetCaptureInfo(stream, &status, &raw_id);
  STD_TORCH_CHECK(err == cudaSuccess,
                  "[fa2_sm70] fold: cudaStreamGetCaptureInfo failed: ",
                  cudaGetErrorString(err),
                  ". Границу захвата графа определить нечем -> ОТКАЗ. "
                  "Отключите свёртку: VLLM_GPTQ_ZERO_C=1");
  if (status != cudaStreamCaptureStatusActive) return false;
  id = raw_id;
  return true;
}

// ФАЛЬСИФИКАТОР (только для доказательства дефекта, НЕ для боя). VLLM_GPTQ_FOLD_LEGACY:
//   не задано / "0"  -- боевой режим: граница прохода по (I1), проверки включены;
//   "1"              -- ПРЕЖНЯЯ детекция (флаг in_capture) И проверки ВЫКЛЮЧЕНЫ.
//                       Это дефект #92 дословно: два прохода, захваченные подряд,
//                       сливаются в один -> второй граф без узла обнуления -> молча
//                       неверные числа. ОТРИЦАТЕЛЬНЫЙ КОНТРОЛЬ;
//   "check"          -- ПРЕЖНЯЯ детекция, но проверки ВКЛЮЧЕНЫ. То же нарушение, но
//                       теперь оно даёт внятный ОТКАЗ: страж живой, а не декоративный.
// Оба falsifier-режима нужны в ОДНОМ бинарнике: иначе "до" и "после" -- две разные
// сборки, и сравнение ничего не доказывает.
//   "graph"          -- граница по ИДЕНТИФИКАТОРУ ЗАХВАТА (ошибка 2 из шапки),
//                       проверки выключены. Числа верные, но узел обнуления всей
//                       плиты ложится в КАЖДЫЙ кусок разбитого графа. Нужен, чтобы
//                       ЦЕНУ этой ошибки можно было замерить в ОДНОМ бинарнике.
enum LegacyMode { LM_OFF, LM_BUG, LM_BUG_CHECKED, LM_GRAPH };
LegacyMode fold_legacy_mode() {
  static const LegacyMode v = [] {
    const char* e = std::getenv("VLLM_GPTQ_FOLD_LEGACY");
    if (!e || !*e || *e == '0') return LM_OFF;
    if (!std::strcmp(e, "check")) return LM_BUG_CHECKED;
    if (!std::strcmp(e, "graph")) return LM_GRAPH;
    return LM_BUG;
  }();
  return v;
}

// E, закреплённый снаружи. 0/пусто -- выучить на первом проходе (и сказать об этом
// вслух: первый проход при обучении защищён только (I5)).
int fold_expect_env() {
  static const int v = [] {
    const char* e = std::getenv("VLLM_GPTQ_FOLD_EXPECT");
    return (e && *e) ? atoi(e) : 0;
  }();
  return v;
}

// Сколько байт плиты проход ДЕЙСТВИТЕЛЬНО занимает при данном m. Обнулять больше
// нечего: хвост за этой границей этот проход не читает, а проход с бо'льшим m занулит
// свой больший префикс сам. Это снимает связь между ЁМКОСТЬЮ плиты и ЦЕНОЙ обнуления:
// замерено, что узел обнуления стоит ровно (байты / 835 ГБ/с), то есть 1.2 мкс на МБ,
// и при плите 40 МБ он съедал 37 мкс на КАЖДОМ токене, даже при B=1, где нужно 4.06 МБ.
size_t fold_pass_bytes(int m) {
  if (g_zslab.zero_m == m) return g_zslab.zero_bytes;
  size_t total = 0;
  for (int width : g_zslab.ref) {
    total += ((size_t)m * (size_t)width * 2 + 511) & ~(size_t)511;
    if (total >= g_zslab.bytes) {
      total = g_zslab.bytes;
      break;
    }
  }
  g_zslab.zero_m = m;
  g_zslab.zero_bytes = total;
  return total;
}

// ФАЛЬСИФИКАТОР: вернуть обнуление ВСЕЙ плиты (как было), чтобы цена префикса мерилась
// в одном бинарнике.
bool fold_zero_all() {
  static const bool v = [] {
    const char* e = std::getenv("VLLM_GPTQ_FOLD_ZERO_ALL");
    return e && *e && *e != '0';
  }();
  return v;
}

// (I5) наименьший период последовательности. Если он делит длину и меньше её --
// последовательность есть повтор, то есть в один "проход" слиплось несколько.
int fold_min_period(const std::vector<int>& v) {
  const int n = static_cast<int>(v.size());
  for (int p = 1; p < n; ++p) {
    if (n % p) continue;
    bool ok = true;
    for (int i = p; i < n && ok; ++i) ok = (v[i] == v[i - p]);
    if (ok) return p;
  }
  return n;
}

// Закрыть проход: напечатать, проверить длину, приготовить следующий.
void fold_close_pass(bool checks) {
  if (!g_zslab.open) return;
  const int n = g_zslab.pos;
  fprintf(stderr,
          "[fa2_sm70] fold: проход %lld закрыт%s: умножений %d (слотов %d, откатов "
          "по ёмкости %d), графов %d, m=%d, обнулено %zu КБ\n",
          g_zslab.passes,
          (g_zslab.ref_ready || g_zslab.slots) ? "" : " [НАБЛЮДЕНИЕ, без свёртки]", n,
          g_zslab.slots, g_zslab.fb_cap, g_zslab.pieces, g_zslab.m_cur,
          g_zslab.zeroed_now >> 10);
  g_zslab.passes++;
  if (checks && g_zslab.expect > 0) {
    STD_TORCH_CHECK(
        n == g_zslab.expect,
        "[fa2_sm70] fold: ИНВАРИАНТ ПРОХОДА НАРУШЕН. Проход закрылся на ", n,
        " умножениях, а проход модели -- ", g_zslab.expect,
        ". Значит граница прохода определена неверно, и плита либо делится между "
        "двумя токенами, либо не обнуляется вовсе -> МОЛЧА НЕВЕРНЫЕ ЧИСЛА. Отказ "
        "вместо отката. Отключите свёртку: VLLM_GPTQ_ZERO_C=1");
  }
  if (checks && !g_zslab.ref_ready && n > 0) {
    // Первый проход закрылся сам (вызовом вне захвата) -- он и есть эталон.
    const int period = fold_min_period(g_zslab.ref);
    STD_TORCH_CHECK(
        period == n,
        "[fa2_sm70] fold: обучение отравлено. Выученная последовательность длины ", n,
        " имеет период ", period, ", то есть в один проход слиплось ", n / period,
        " прохода: между ними не случилось ни одного умножения вне графа. Закрепите "
        "длину прохода явно (VLLM_GPTQ_FOLD_EXPECT=<число умножений на токен>) или "
        "отключите свёртку: VLLM_GPTQ_ZERO_C=1");
    g_zslab.expect = n;
    g_zslab.ref_ready = true;
    fprintf(stderr,
            "[fa2_sm70] fold: длина прохода ВЫУЧЕНА = %d умножений. Первый проход "
            "прошёл без сверки; чтобы закрыть и его, задайте "
            "VLLM_GPTQ_FOLD_EXPECT=%d\n",
            n, n);
  }
  g_zslab.open = false;
  g_zslab.pos = 0;
  g_zslab.slots = 0;
  g_zslab.fb_cap = 0;
  g_zslab.bump = 0;
  g_zslab.zeroed = false;
  g_zslab.zeroed_graph_prev = g_zslab.graph_zero;
  g_zslab.graph_zero = 0;
  g_zslab.pieces = 0;
  g_zslab.graph_last = 0;
  g_zslab.m_cur = 0;
}

// Проход длиной E закрывается СРАЗУ на своём E-м умножении.
void fold_maybe_close(bool by_count, bool checks) {
  if (!by_count) return;  // при прежней детекции границу ставит только выход из графа
  if (g_zslab.expect > 0 && g_zslab.pos >= g_zslab.expect) fold_close_pass(checks);
}

// НАБЛЮДЕНИЕ: собрать эталон прохода, не выдавая слотов. Первый проход идёт обычным
// путём (new_zeros) -- цена этого одна на подъём, зато обучение не может испортить
// числа: пока эталона нет, свёртки нет.
void fold_observe(int64_t m, int64_t n, bool checks) {
  if (!g_zslab.open) {
    g_zslab.open = true;
    g_zslab.m_cur = static_cast<int>(m);
    g_zslab.ref.clear();
  }
  if (static_cast<int>(m) != g_zslab.m_cur) {
    // Наблюдение НИЧЕМ не рискует (свёртки ещё нет), поэтому помеха его перезапускает,
    // а не роняет сервер. Если эталон так и не соберётся, механизм просто не включится
    // -- и это видно в логе, а числа остаются верными.
    if (!g_zslab.warned_observe) {
      g_zslab.warned_observe = true;
      fprintf(stderr,
              "[fa2_sm70] fold: наблюдение прохода перезапущено (умножение %d пришло "
              "с m=%d, а проход шёл с m=%d). Пока эталон не собран, свёртки нет.\n",
              g_zslab.pos, static_cast<int>(m), g_zslab.m_cur);
    }
    g_zslab.ref.clear();
    g_zslab.pos = 0;
    g_zslab.m_cur = static_cast<int>(m);
  }
  g_zslab.ref.push_back(static_cast<int>(n));
  g_zslab.pos++;
  if (g_zslab.expect > 0 && static_cast<int>(g_zslab.ref.size()) >= g_zslab.expect)
    g_zslab.ref_ready = true;
  fold_maybe_close(true, checks);
}
}  // namespace

// Плиту строим ТОЛЬКО вне захвата (cudaMalloc при захвате запрещён), поэтому зовём это
// на КАЖДОМ вызове gptq_gemm, а не только на квантованной ветке: профилировочный прогон
// vLLM идёт по ветке reconstruct, и если ждать первого квантованного вызова вне графа,
// плиты может не оказаться к моменту захвата -- режим тихо не включится.
void fold_maybe_init() {
  if (g_zslab.tried) return;
  cudaStream_t stream = get_current_cuda_stream();
  cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
  if (cudaStreamIsCapturing(stream, &st) != cudaSuccess) return;
  if (st == cudaStreamCaptureStatusActive) return;
  g_zslab.tried = true;
  const char* mb = std::getenv("VLLM_GPTQ_FOLD_MB");
  size_t want = (size_t)((mb && *mb) ? atoi(mb) : 8) << 20;
  int dev = -1;
  cudaError_t derr = cudaGetDevice(&dev);
  STD_TORCH_CHECK(derr == cudaSuccess, "[fa2_sm70] fold: cudaGetDevice failed: ",
                  cudaGetErrorString(derr));
  const int pin = fold_expect_env();
  if (pin > 0) {
    g_zslab.expect = pin;
    g_zslab.pinned = true;
  }
  if (cudaMalloc(&g_zslab.base, want) == cudaSuccess &&
      cudaMemset(g_zslab.base, 0, want) == cudaSuccess) {
    g_zslab.bytes = want;
    g_zslab.device = dev;  // (I4) плита привязана к устройству
    fprintf(stderr,
            "[fa2_sm70] fold: плита %zu МБ готова (устройство %d, длина прохода %s)\n",
            want >> 20, dev, g_zslab.pinned ? "закреплена" : "будет выучена");
  } else {
    g_zslab.base = nullptr;
    fprintf(stderr, "[fa2_sm70] fold: плита %zu МБ НЕ выделена, откат\n", want >> 20);
  }
}

// Возвращает тензор из плиты (ok=true) либо пустой тензор (ok=false -> обычный путь).
// ok=false бывает вне захвата графа, при нехватке ёмкости (I3), при чужом устройстве
// (I4) и если плиты нет вовсе. Нарушение инварианта прохода -- ОТКАЗ, а не откат.
torch::stable::Tensor fold_take_slot(const torch::stable::Tensor& like, int64_t m,
                                     int64_t n, bool& ok) {
  ok = false;
  const LegacyMode legacy = fold_legacy_mode();
  const bool checks = (legacy == LM_OFF || legacy == LM_BUG_CHECKED);
  cudaStream_t stream = get_current_cuda_stream();

  // --- 0. (I4) чужое устройство -- НЕ трогаем состояние вообще. ------------
  // Плита живёт на той карте, где первым позвали cudaMalloc, а options берутся у
  // входного тензора: на другой карте это был бы указатель в чужую память. Числа при
  // отказе от свёртки остаются верными (обычный путь), поэтому здесь ОТКАТ, а не
  // отказ; но откат считается и один раз кричит в лог.
  if (checks && g_zslab.base) {
    int cur = -1;
    const int want_dev = static_cast<int>(like.get_device_index());
    if (cudaGetDevice(&cur) != cudaSuccess) cur = -1;
    if (want_dev != g_zslab.device || cur != g_zslab.device) {
      g_zslab.fb_device++;
      if (!g_zslab.warned_device) {
        g_zslab.warned_device = true;
        fprintf(stderr,
                "[fa2_sm70] fold: вызов на устройстве %d (текущее %d), а плита на %d "
                "-> НЕ сворачиваем (обычный путь, числа верны). Свёртка рассчитана "
                "на один процесс = одно устройство; при спекулятивном декоде на "
                "второй карте в том же процессе выигрыш там просто не берётся.\n",
                want_dev, cur, g_zslab.device);
      }
      return torch::stable::Tensor();
    }
  }

  // --- 1. Активен ли захват. ----------------------------------------------
  // (наблюдение первого прохода -- ниже: пока эталон не собран, слоты НЕ выдаются)
  cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
  unsigned long long id = 0;
  if (legacy == LM_BUG) {
    // прежняя версия спрашивала только статус, id не спрашивала вовсе
    if (cudaStreamIsCapturing(stream, &st) != cudaSuccess)
      return torch::stable::Tensor();
  } else {
    if (fold_capture_id(stream, id)) st = cudaStreamCaptureStatusActive;
  }

  const bool observing = checks && !g_zslab.ref_ready && g_zslab.base != nullptr;
  if (st != cudaStreamCaptureStatusActive) {
    // Вне графа сворачивать нечего (смещения не зафиксированы), и это же --
    // естественная граница прохода: закрываем открытый.
    // ИСКЛЮЧЕНИЕ: при ЗАКРЕПЛЁННОМ E эталон собирается прямо здесь, на прогонах вне
    // графа (vLLM делает их перед захватом). Тогда к первому же захвату эталон готов
    // и сворачивается ВСЁ. Без закрепления эталон собрать вне графа нечем -- проход
    // там нечем закрыть, -- и он собирается на первом ЗАХВАЧЕННОМ проходе.
    if (observing && g_zslab.pinned && legacy != LM_BUG) {
      fold_observe(m, n, checks);
      return torch::stable::Tensor();
    }
    if (legacy == LM_BUG) {
      if (g_zslab.in_capture)
        fprintf(stderr, "[fa2_sm70] fold: захват закончен, слотов %d, откатов %d\n",
                g_zslab.slots, g_zslab.fb_cap);
      g_zslab.in_capture = false;
      g_zslab.open = false;
      g_zslab.pos = g_zslab.slots = g_zslab.fb_cap = 0;
      g_zslab.bump = 0;
      g_zslab.zeroed = false;
    } else {
      fold_close_pass(checks);
    }
    return torch::stable::Tensor();
  }
  g_zslab.in_capture = true;
  if (!g_zslab.base) return torch::stable::Tensor();

  if (observing) {
    // ПЕРВЫЙ проход только измеряется: ни слотов, ни узла обнуления. Тогда обучение
    // не может дать неверных чисел даже в самом плохом расписании захватов -- худшее,
    // что бывает, это "механизм не включился", и это видно в логе.
    fold_observe(m, n, checks);
    return torch::stable::Tensor();
  }

  // --- 2. Начало прохода. --------------------------------------------------
  // ПРЕЖНЯЯ детекция (LEGACY): проход кончается только вызовом ВНЕ захвата -- два
  //   прохода, захваченные подряд, сливаются (дефект #92).
  // ТЕКУЩАЯ детекция: проход кончается на своём E-м умножении (закрывается ниже, на
  //   месте) ; разбиение графа на куски проход НЕ делит.
  if (legacy == LM_GRAPH && g_zslab.open && id != g_zslab.graph_last) {
    fold_close_pass(false);  // ФАЛЬСИФИКАТОР: граница по графу, а не по проходу
  }
  if (!g_zslab.open) {
    if (legacy == LM_OFF && g_zslab.zeroed_graph_prev == id &&
        !g_zslab.warned_same_graph) {
      g_zslab.warned_same_graph = true;
      fprintf(stderr,
              "[fa2_sm70] fold: ВНИМАНИЕ, в ОДИН граф %llu попало больше одного "
              "прохода. Тогда второй узел обнуления затирает выходы первого прохода "
              "ВНУТРИ одного воспроизведения; это верно, только если те выходы к "
              "тому моменту уже прочитаны. Такая топология не проверялась -- сверьте "
              "числа или отключите свёртку: VLLM_GPTQ_ZERO_C=1\n",
              id);
    }
    g_zslab.open = true;
    g_zslab.m_cur = static_cast<int>(m);
    if (!g_zslab.ref_ready) g_zslab.ref.clear();
  }
  if (id != g_zslab.graph_last) {
    g_zslab.graph_last = id;
    g_zslab.pieces++;
  }

  // --- 3. (I1) k-е умножение прохода обязано быть тем же самым. ------------
  const int k = g_zslab.pos;
  if (checks) {
    STD_TORCH_CHECK(static_cast<int>(m) == g_zslab.m_cur,
                    "[fa2_sm70] fold: ИНВАРИАНТ ПРОХОДА НАРУШЕН. Умножение ", k,
                    " пришло с m=", static_cast<int>(m), ", а проход идёт с m=",
                    g_zslab.m_cur,
                    ". Границу прохода определить нечем -> ОТКАЗ. Отключите "
                    "свёртку: VLLM_GPTQ_ZERO_C=1");
    // Сюда попадают только проходы ПОСЛЕ наблюдения: пока эталона нет, слоты не
    // выдаются вовсе (см. fold_observe выше).
    STD_TORCH_CHECK(g_zslab.ref_ready,
                    "[fa2_sm70] fold: слот выдаётся до того, как собран эталон "
                    "прохода -- внутренняя ошибка порядка проверок");
    {
      STD_TORCH_CHECK(
          k < static_cast<int>(g_zslab.ref.size()),
          "[fa2_sm70] fold: ИНВАРИАНТ ПРОХОДА НАРУШЕН. Умножение ", k,
          " выходит за длину прохода ", static_cast<int>(g_zslab.ref.size()),
          ": в один проход слиплось несколько (два графа захвачены подряд?), плита "
          "делится между токенами -> МОЛЧА НЕВЕРНЫЕ ЧИСЛА. Отказ вместо отката. "
          "Отключите свёртку: VLLM_GPTQ_ZERO_C=1");
      STD_TORCH_CHECK(
          g_zslab.ref[k] == static_cast<int>(n),
          "[fa2_sm70] fold: ИНВАРИАНТ ПРОХОДА НАРУШЕН. Умножение ", k,
          " прохода имеет ширину ", static_cast<int>(n), ", а эталон прохода -- ",
          g_zslab.ref[k],
          ". Последовательность умножений токена изменилась: проход потерял или "
          "добавил умножения, либо в тот же процесс вклинилась другая модель. Плита "
          "тогда делится не так, как при захвате -> МОЛЧА НЕВЕРНЫЕ ЧИСЛА. Отключите "
          "свёртку: VLLM_GPTQ_ZERO_C=1");
    }
  }

  // --- 4. (I3) слот должен влезть. ----------------------------------------
  // ЭТО НЕ НАРУШЕНИЕ ИНВАРИАНТА, а ёмкость: не влез -- уходим на new_zeros, что
  // ЧИСЛЕННО ВЕРНО, просто без выигрыша. Отказывать здесь НЕЛЬЗЯ: при B>1 частичная
  // свёртка -- штатный режим (по боевому логу 232/116/59 слотов из 236 при B=2/4/8 и
  // плите 8 МБ). Но откат перестаёт быть ТИХИМ: он считается, печатается на закрытии
  // прохода и занимает свою позицию k.
  const size_t need = ((size_t)m * (size_t)n * 2 + 511) & ~(size_t)511;
  if (g_zslab.bump + need > g_zslab.bytes) {
    g_zslab.fb_cap++;
    g_zslab.pos++;
    fold_maybe_close(legacy == LM_OFF, checks);
    return torch::stable::Tensor();
  }

  // --- 5. (I2) узел обнуления -- ровно один и ровно в начале прохода. ------
  char* p = static_cast<char*>(g_zslab.base) + g_zslab.bump;
  const bool first = (legacy == LM_BUG) ? (g_zslab.pos == 0) : !g_zslab.zeroed;
  if (first) {
    const size_t zbytes = (g_zslab.ref_ready && !fold_zero_all())
                              ? fold_pass_bytes(static_cast<int>(m))
                              : g_zslab.bytes;
    cudaError_t merr = cudaMemsetAsync(g_zslab.base, 0, zbytes, stream);
    STD_TORCH_CHECK(merr == cudaSuccess,
                    "[fa2_sm70] fold: cudaMemsetAsync не встал в граф: ",
                    cudaGetErrorString(merr), " -> ОТКАЗ");
    g_zslab.zeroed = true;
    g_zslab.graph_zero = id;
    g_zslab.zeroed_now = zbytes;
  }
  if (checks) {
    STD_TORCH_CHECK(
        g_zslab.zeroed, "[fa2_sm70] fold: ИНВАРИАНТ ПРОХОДА НАРУШЕН. Слот ", k,
        " выдаётся в проходе, в который НЕ поставлен узел обнуления плиты. Такой "
        "граф при воспроизведении кладёт atomicAdd поверх выхода предыдущего токена "
        "-> МОЛЧА НЕВЕРНЫЕ ЧИСЛА. Отказ вместо отката. Отключите свёртку: "
        "VLLM_GPTQ_ZERO_C=1");
  }

  g_zslab.bump += need;
  g_zslab.pos++;
  g_zslab.slots++;
  g_zslab.folded++;
  ok = true;
  const int64_t sizes[2] = {m, n};
  const int64_t strides[2] = {n, 1};
  torch::stable::Tensor out = torch::stable::from_blob(
      p, torch::headeronly::IntHeaderOnlyArrayRef(sizes, 2),
      torch::headeronly::IntHeaderOnlyArrayRef(strides, 2), like.device(),
      like.scalar_type());
  // Проход закрываем НА МЕСТЕ, на его E-м умножении, а не по приходу следующего:
  // тогда проверен КАЖДЫЙ проход, включая последний захваченный (после него вызовов
  // может уже не быть -- воспроизведение графа в хост не заходит).
  fold_maybe_close(legacy == LM_OFF, checks);
  return out;
}

}  // namespace gptq
}  // namespace vllm

torch::stable::Tensor gptq_gemm(torch::stable::Tensor a,
                                torch::stable::Tensor b_q_weight,
                                torch::stable::Tensor b_gptq_qzeros,
                                torch::stable::Tensor b_gptq_scales,
                                torch::stable::Tensor b_g_idx, bool use_exllama,
                                bool use_v2_format, int64_t bit) {
  const torch::stable::accelerator::DeviceGuard device_guard(
      a.get_device_index());
  // [fa2_sm70 / ПАРА-1] Режимы обнуления C. Значение VLLM_GPTQ_ZERO_C:
  //   (не задано) / "0"  -- пропустить обнуление ТОЛЬКО на пути reconstruct
  //                         (там cublasHgemm с beta = 0 не читает C);
  //   "1" (любое иное)   -- обнулять всегда (исходное поведение vLLM);
  //   "never"            -- НИКОГДА не обнулять. ЧИСЛЕННО НЕВЕРНО на пути atomicAdd,
  //                         это ФАЛЬСИФИКАТОР: снимает ровно N узлов графа на токен,
  //                         не меняя ни одного байта прочей работы -> ПОТОЛОК выигрыша;
  //   "fold"             -- ПАССАЖИР: C декодных умножений берётся из постоянной плиты,
  //                         и вся плита зануляется ОДНИМ узлом cudaMemsetAsync на токен
  //                         (N ядер -> 1 узел). Работает только внутри захвата графа.
  enum ZeroMode { ZM_DEFAULT, ZM_ALWAYS, ZM_NEVER, ZM_FOLD };
  static const ZeroMode zmode = [] {
    const char* e = std::getenv("VLLM_GPTQ_ZERO_C");
    if (!e || !*e) return ZM_DEFAULT;
    if (!std::strcmp(e, "never")) return ZM_NEVER;
    if (!std::strcmp(e, "fold")) return ZM_FOLD;
    if (*e == '0') return ZM_DEFAULT;
    return ZM_ALWAYS;
  }();

  const bool uses_reconstruct = vllm::gptq::gptq_gemm_uses_reconstruct(
      static_cast<int>(a.size(0)), use_exllama, static_cast<int>(bit));
  bool no_zero_needed = (zmode != ZM_ALWAYS) && uses_reconstruct;
  if (zmode == ZM_NEVER) no_zero_needed = true;

  torch::stable::Tensor c;
  bool c_ready = false;
  if (zmode == ZM_FOLD) vllm::gptq::fold_maybe_init();
  if (zmode == ZM_FOLD && !uses_reconstruct) {
    c = vllm::gptq::fold_take_slot(a, a.size(0), b_q_weight.size(1), c_ready);
  }
  if (!c_ready) {
    c = no_zero_needed
            ? torch::stable::new_empty(a, {a.size(0), b_q_weight.size(1)})
            : torch::stable::new_zeros(a, {a.size(0), b_q_weight.size(1)});
  }
  auto temp_dq =
      torch::stable::empty({b_q_weight.size(0) * 32 / bit, b_q_weight.size(1)},
                           a.scalar_type(), std::nullopt, a.device());

  vllm::gptq::gemm_half_q_half_cuda(
      get_current_cuda_blas_handle(), (const half*)a.data_ptr(),
      (const uint32_t*)b_q_weight.data_ptr(),
      (const uint32_t*)b_gptq_qzeros.data_ptr(),
      (const half*)b_gptq_scales.data_ptr(),
      b_g_idx.device().type() == torch::stable::DeviceType::Meta
          ? NULL
          : (const int*)b_g_idx.data_ptr(),
      (half*)c.data_ptr(), (half*)temp_dq.data_ptr(),
      c.size(0),              // m
      c.size(1),              // n
      a.size(1),              // k
      b_gptq_qzeros.size(0),  // group number
      use_exllama, use_v2_format, bit);
  return c;
}

void gptq_shuffle(torch::stable::Tensor q_weight, torch::stable::Tensor q_perm,
                  int64_t bit) {
  const torch::stable::accelerator::DeviceGuard device_guard(
      q_weight.get_device_index());
  vllm::gptq::shuffle_exllama_weight(
      (uint32_t*)q_weight.data_ptr(),
      q_perm.device().type() == torch::stable::DeviceType::Meta ||
              q_perm.numel() == 0
          ? NULL
          : (int*)q_perm.data_ptr(),
      q_weight.size(0) * 32 / bit, q_weight.size(1), bit);
}
