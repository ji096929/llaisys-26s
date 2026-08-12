#include "argmax_nvidia.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "../../../utils.hpp"

namespace llaisys::ops::nvidia {
namespace {

__device__ float to_float(float v) { return v; }

__device__ float to_float(__half v) { return __half2float(v); }

__device__ float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }

// 单 block 归约（测试规模 numel <= 4096 足够；更大会慢但正确）
template <typename T>
__global__ void argmax_kernel(int64_t *max_idx, T *max_val, const T *vals, size_t n) {
    constexpr unsigned int BLOCK = 256;
    __shared__ float s_val[BLOCK];
    __shared__ size_t s_idx[BLOCK];

    unsigned int tid = threadIdx.x;
    float local_best = -INFINITY;
    size_t local_idx = 0;
    for (size_t i = tid; i < n; i += BLOCK) {
        float v = to_float(vals[i]);
        if (v > local_best) {
            local_best = v;
            local_idx = i;
        }
    }
    s_val[tid] = local_best;
    s_idx[tid] = local_idx;
    __syncthreads();

    for (unsigned int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_val[tid + s] > s_val[tid]) {
                s_val[tid] = s_val[tid + s];
                s_idx[tid] = s_idx[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *max_idx = static_cast<int64_t>(s_idx[0]);
        *max_val = vals[s_idx[0]];
    }
}

} // namespace

void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals,
            llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        argmax_kernel<float><<<1, 256>>>(
            reinterpret_cast<int64_t *>(max_idx),
            reinterpret_cast<float *>(max_val),
            reinterpret_cast<const float *>(vals), numel);
        break;
    case LLAISYS_DTYPE_F16:
        argmax_kernel<__half><<<1, 256>>>(
            reinterpret_cast<int64_t *>(max_idx),
            reinterpret_cast<__half *>(max_val),
            reinterpret_cast<const __half *>(vals), numel);
        break;
    case LLAISYS_DTYPE_BF16:
        argmax_kernel<__nv_bfloat16><<<1, 256>>>(
            reinterpret_cast<int64_t *>(max_idx),
            reinterpret_cast<__nv_bfloat16 *>(max_val),
            reinterpret_cast<const __nv_bfloat16 *>(vals), numel);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
    cudaDeviceSynchronize();
}

} // namespace llaisys::ops::nvidia
