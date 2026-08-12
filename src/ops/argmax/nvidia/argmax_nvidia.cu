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
// 归约路径：每线程局部扫描 → warp shuffle 归约（免共享内存同步）→ warp0 收尾
template <typename T>
__global__ void argmax_kernel(int64_t *max_idx, T *max_val, const T *vals, size_t n) {
    constexpr unsigned int BLOCK = 256;
    constexpr unsigned int WARP = 32;
    __shared__ float s_val[BLOCK / WARP]; // 每个 warp 一个槽位（8 个）
    __shared__ size_t s_idx[BLOCK / WARP];

    unsigned int tid = threadIdx.x;
    unsigned int lane = tid % WARP;
    unsigned int warp = tid / WARP;

    float local_best = -INFINITY;
    size_t local_idx = 0;
    for (size_t i = tid; i < n; i += BLOCK) {
        float v = to_float(__ldg(vals + i)); // 只读提示，走只读缓存路径
        if (v > local_best) {
            local_best = v;
            local_idx = i;
        }
    }

    // warp 内归约：直接读队友的寄存器，不需要共享内存和 __syncthreads
    for (unsigned int offset = WARP / 2; offset > 0; offset >>= 1) {
        float other_val = __shfl_down_sync(0xffffffffu, local_best, offset);
        size_t other_idx = __shfl_down_sync(0xffffffffu, local_idx, offset);
        if (other_val > local_best) {
            local_best = other_val;
            local_idx = other_idx;
        }
    }

    // 每个 warp 的 lane 0 把胜者写进共享内存（只 8 个槽）
    if (lane == 0) {
        s_val[warp] = local_best;
        s_idx[warp] = local_idx;
    }
    __syncthreads();

    // warp 0 收尾：8 个 warp 胜者再做一次 shuffle 归约
    if (warp == 0) {
        float best = -INFINITY;
        size_t best_idx = 0;
        if (lane < BLOCK / WARP) {
            best = s_val[lane];
            best_idx = s_idx[lane];
        }
        for (unsigned int offset = (BLOCK / WARP) / 2; offset > 0; offset >>= 1) {
            float other_val = __shfl_down_sync(0xffffffffu, best, offset);
            size_t other_idx = __shfl_down_sync(0xffffffffu, best_idx, offset);
            if (other_val > best) {
                best = other_val;
                best_idx = other_idx;
            }
        }
        if (lane == 0) {
            *max_idx = static_cast<int64_t>(best_idx);
            *max_val = vals[best_idx];
        }
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
