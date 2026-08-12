#include "rms_norm_maca.hpp"

#include <mc_runtime.h>
#include <mc_common.h>
#include <maca_fp16.h>
#include <maca_bfloat16.h>

#include "../../../utils.hpp"

#include <cmath>

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
__global__ void rms_norm_kernel(T *out, const T *in, const T *weight,
                                size_t cols, float eps) {
    constexpr unsigned int BLOCK = 256;
    __shared__ float s_sum[BLOCK];

    size_t row = blockIdx.x;
    const T *row_in = in + row * cols;
    T *row_out = out + row * cols;
    unsigned int tid = threadIdx.x;

    float sum = 0.0f;
    for (size_t j = tid; j < cols; j += BLOCK) {
        float v = to_float(row_in[j]);
        sum += v * v;
    }
    s_sum[tid] = sum;
    __syncthreads();

    for (unsigned int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        s_sum[0] = rsqrtf(s_sum[0] / static_cast<float>(cols) + eps);
    }
    __syncthreads();

    float scale = s_sum[0];
    for (size_t j = tid; j < cols; j += BLOCK) {
        row_out[j] = from_float<T>(to_float(row_in[j]) * scale * to_float(weight[j]));
    }
}

} // namespace

void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              llaisysDataType_t type, size_t rows, size_t cols, float eps) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        rms_norm_kernel<float><<<static_cast<unsigned int>(rows), 256>>>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const float *>(weight), cols, eps);
        break;
    case LLAISYS_DTYPE_F16:
        rms_norm_kernel<__half><<<static_cast<unsigned int>(rows), 256>>>(
            reinterpret_cast<__half *>(out),
            reinterpret_cast<const __half *>(in),
            reinterpret_cast<const __half *>(weight), cols, eps);
        break;
    case LLAISYS_DTYPE_BF16:
        rms_norm_kernel<__maca_bfloat16><<<static_cast<unsigned int>(rows), 256>>>(
            reinterpret_cast<__maca_bfloat16 *>(out),
            reinterpret_cast<const __maca_bfloat16 *>(in),
            reinterpret_cast<const __maca_bfloat16 *>(weight), cols, eps);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
    mcDeviceSynchronize();
}

} // namespace llaisys::ops::maca
