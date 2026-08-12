#include "embedding_nvidia.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "../../../utils.hpp"

namespace llaisys::ops::nvidia {
namespace {

template <typename T>
__global__ void embedding_kernel(T *out, const int64_t *index, const T *weight,
                                 size_t rows, size_t cols) {
    size_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const T *src = weight + static_cast<size_t>(index[row]) * cols;
    T *dst = out + row * cols;
    for (size_t j = threadIdx.x; j < cols; j += blockDim.x) {
        dst[j] = src[j];
    }
}

} // namespace

void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t type, size_t rows, size_t cols) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        embedding_kernel<float><<<static_cast<unsigned int>(rows), 256>>>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const int64_t *>(index),
            reinterpret_cast<const float *>(weight), rows, cols);
        break;
    case LLAISYS_DTYPE_F16:
        embedding_kernel<__half><<<static_cast<unsigned int>(rows), 256>>>(
            reinterpret_cast<__half *>(out),
            reinterpret_cast<const int64_t *>(index),
            reinterpret_cast<const __half *>(weight), rows, cols);
        break;
    case LLAISYS_DTYPE_BF16:
        embedding_kernel<__nv_bfloat16><<<static_cast<unsigned int>(rows), 256>>>(
            reinterpret_cast<__nv_bfloat16 *>(out),
            reinterpret_cast<const int64_t *>(index),
            reinterpret_cast<const __nv_bfloat16 *>(weight), rows, cols);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
    cudaDeviceSynchronize();
}

} // namespace llaisys::ops::nvidia
