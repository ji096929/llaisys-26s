#include "add_maca.hpp"

#include <mc_runtime.h>
#include <mc_common.h>
#include <maca_fp16.h>
#include <maca_bfloat16.h>

#include "../../../utils.hpp"

namespace llaisys::ops::maca {
namespace {

__device__ float to_float(float v) { return v; }
__device__ float to_float(__half v) { return __half2float(v); }
__device__ float to_float(__maca_bfloat16 v) { return __bfloat162float(v); }

template <typename T>
__device__ T from_float(float v);

template <>
__device__ float from_float<float>(float v) { return v; }

template <>
__device__ __half from_float<__half>(float v) { return __float2half(v); }

template <>
__device__ __maca_bfloat16 from_float<__maca_bfloat16>(float v) { return __float2bfloat16(v); }

template <typename T>
__global__ void add_kernel(T *c, const T *a, const T *b, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = from_float<T>(to_float(a[i]) + to_float(b[i]));
    }
}

} // namespace

void add(std::byte *c, const std::byte *a, const std::byte *b,
         llaisysDataType_t type, size_t numel) {
    const unsigned int block = 256;
    const unsigned int grid = static_cast<unsigned int>((numel + block - 1) / block);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        add_kernel<float><<<grid, block>>>(
            reinterpret_cast<float *>(c),
            reinterpret_cast<const float *>(a),
            reinterpret_cast<const float *>(b), numel);
        break;
    case LLAISYS_DTYPE_F16:
        add_kernel<__half><<<grid, block>>>(
            reinterpret_cast<__half *>(c),
            reinterpret_cast<const __half *>(a),
            reinterpret_cast<const __half *>(b), numel);
        break;
    case LLAISYS_DTYPE_BF16:
        add_kernel<__maca_bfloat16><<<grid, block>>>(
            reinterpret_cast<__maca_bfloat16 *>(c),
            reinterpret_cast<const __maca_bfloat16 *>(a),
            reinterpret_cast<const __maca_bfloat16 *>(b), numel);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
    mcDeviceSynchronize();
}

} // namespace llaisys::ops::maca
