#include "swiglu_nvidia.hpp"

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

template <typename T>
__global__ void swiglu_kernel(T *out, const T *gate, const T *up, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = to_float(gate[i]);
        float silu = g / (1.0f + expf(-g));
        out[i] = from_float<T>(to_float(up[i]) * silu);
    }
}

} // namespace

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t type, size_t numel) {
    const unsigned int block = 256;
    const unsigned int grid = static_cast<unsigned int>((numel + block - 1) / block);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        swiglu_kernel<float><<<grid, block>>>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(gate),
            reinterpret_cast<const float *>(up), numel);
        break;
    case LLAISYS_DTYPE_F16:
        swiglu_kernel<__half><<<grid, block>>>(
            reinterpret_cast<__half *>(out),
            reinterpret_cast<const __half *>(gate),
            reinterpret_cast<const __half *>(up), numel);
        break;
    case LLAISYS_DTYPE_BF16:
        swiglu_kernel<__nv_bfloat16><<<grid, block>>>(
            reinterpret_cast<__nv_bfloat16 *>(out),
            reinterpret_cast<const __nv_bfloat16 *>(gate),
            reinterpret_cast<const __nv_bfloat16 *>(up), numel);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
    cudaDeviceSynchronize();
}

} // namespace llaisys::ops::nvidia
