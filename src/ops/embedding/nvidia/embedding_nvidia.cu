#include "embedding_nvidia.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "../../../utils.hpp"

#include <algorithm>

namespace llaisys::ops::nvidia {
namespace {

// 向量类型映射：f32 → float4（4 元素），f16/bf16 → half2/bfloat162（2 元素）
template <typename T>
struct vec_type;

template <>
struct vec_type<float> {
    using type = float4;
    static constexpr int n = 4;
};

template <>
struct vec_type<__half> {
    using type = __half2;
    static constexpr int n = 2;
};

template <>
struct vec_type<__nv_bfloat16> {
    using type = __nv_bfloat162;
    static constexpr int n = 2;
};

// 标量版本：展平为 total = rows * cols 的 grid-stride，避免 rows/cols 不平衡
template <typename T>
__global__ void embedding_scalar_kernel(T *out, const int64_t *index, const T *weight,
                                        size_t rows, size_t cols) {
    size_t total = rows * cols;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += stride) {
        size_t row = i / cols;
        size_t col = i % cols;
        out[i] = weight[static_cast<size_t>(index[row]) * cols + col];
    }
}

// 向量化版本：要求 cols 是向量元素数的整数倍（且行偏移满足对齐）
template <typename T>
__global__ void embedding_vec_kernel(T *out, const int64_t *index, const T *weight,
                                     size_t rows, size_t vec_cols) {
    using Vec = typename vec_type<T>::type;
    constexpr int N = vec_type<T>::n;

    size_t total = rows * vec_cols;
    size_t stride = blockDim.x * gridDim.x;
    const Vec *w_vec = reinterpret_cast<const Vec *>(weight);
    Vec *o_vec = reinterpret_cast<Vec *>(out);
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += stride) {
        size_t row = i / vec_cols;
        size_t vc = i % vec_cols;
        o_vec[row * vec_cols + vc] = w_vec[static_cast<size_t>(index[row]) * vec_cols + vc];
    }
}

template <typename T>
void embedding_impl(T *out, const int64_t *index, const T *weight,
                    size_t rows, size_t cols) {
    constexpr int N = vec_type<T>::n;
    if (cols % N == 0) {
        // 向量化：每个向量 N 个元素；行偏移 = index*cols 个元素，
        // cols % N == 0 保证字节对齐（f32→16B，f16/bf16→8B）
        size_t vec_cols = cols / N;
        size_t total = rows * vec_cols;
        unsigned int block = 256;
        unsigned int grid = static_cast<unsigned int>((total + block - 1) / block);
        embedding_vec_kernel<T><<<grid, block>>>(out, index, weight, rows, vec_cols);
    } else {
        // 标量回退：block 不超过 cols，避免空转线程
        size_t total = rows * cols;
        unsigned int block = static_cast<unsigned int>(std::min<size_t>(256, cols));
        if (block == 0) {
            block = 1;
        }
        unsigned int grid = static_cast<unsigned int>((total + block - 1) / block);
        embedding_scalar_kernel<T><<<grid, block>>>(out, index, weight, rows, cols);
    }
}

} // namespace

void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t type, size_t rows, size_t cols) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        embedding_impl(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const int64_t *>(index),
            reinterpret_cast<const float *>(weight), rows, cols);
        break;
    case LLAISYS_DTYPE_F16:
        embedding_impl(
            reinterpret_cast<__half *>(out),
            reinterpret_cast<const int64_t *>(index),
            reinterpret_cast<const __half *>(weight), rows, cols);
        break;
    case LLAISYS_DTYPE_BF16:
        embedding_impl(
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
