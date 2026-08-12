#include "rope_nvidia.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "../../../utils.hpp"

namespace llaisys::ops::nvidia {
namespace {

__device__ float to_float(float v) { return v; }
__device__ float to_float(__half v) { return __half2float(v); }
__device__ float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T>
__device__ T from_float(float v);

template <>
__device__ float from_float<float>(float v) { return v; }

template <>
__device__ __half from_float<__half>(float v) { return __float2half(v); }

template <>
__device__ __nv_bfloat16 from_float<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

// 每个 (token, j) 计算一次 sin/cos，然后应用到所有 head
template <typename T>
__global__ void rope_kernel(T *out, const T *in, const int64_t *pos_ids,
                            size_t seq_len, size_t n_heads, size_t half,
                            size_t head_dim, float theta) {
    size_t sj = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = seq_len * half;
    if (sj >= total) {
        return;
    }
    size_t s = sj / half;
    size_t j = sj % half;

    float pos = static_cast<float>(pos_ids[s]);
    float angle = pos / powf(theta, 2.0f * static_cast<float>(j) / static_cast<float>(head_dim));
    float sn = sinf(angle);
    float cs = cosf(angle);

    size_t token_base = s * n_heads * head_dim;
    for (size_t h = 0; h < n_heads; h++) {
        size_t off = token_base + h * head_dim;
        float a = to_float(in[off + j]);
        float b = to_float(in[off + half + j]);
        out[off + j] = from_float<T>(a * cs - b * sn);
        out[off + half + j] = from_float<T>(b * cs + a * sn);
    }
}

} // namespace

void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          llaisysDataType_t type, size_t seq_len, size_t n_heads, size_t head_dim,
          float theta) {
    size_t half = head_dim / 2;
    unsigned int grid = static_cast<unsigned int>((seq_len * half + 255) / 256);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        rope_kernel<float><<<grid, 256>>>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const int64_t *>(pos_ids),
            seq_len, n_heads, half, head_dim, theta);
        break;
    case LLAISYS_DTYPE_F16:
        rope_kernel<__half><<<grid, 256>>>(
            reinterpret_cast<__half *>(out),
            reinterpret_cast<const __half *>(in),
            reinterpret_cast<const int64_t *>(pos_ids),
            seq_len, n_heads, half, head_dim, theta);
        break;
    case LLAISYS_DTYPE_BF16:
        rope_kernel<__nv_bfloat16><<<grid, 256>>>(
            reinterpret_cast<__nv_bfloat16 *>(out),
            reinterpret_cast<const __nv_bfloat16 *>(in),
            reinterpret_cast<const int64_t *>(pos_ids),
            seq_len, n_heads, half, head_dim, theta);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
    cudaDeviceSynchronize();
}

} // namespace llaisys::ops::nvidia
