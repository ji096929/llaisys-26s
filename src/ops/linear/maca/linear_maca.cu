#include "linear_maca.hpp"

#include <mc_runtime.h>
#include <mc_common.h>
#include <maca_fp16.h>
#include <maca_bfloat16.h>
#include <mcblas.h>

#include "../../../utils.hpp"

#include <stdexcept>

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
__global__ void add_bias_kernel(T *y, const T *bias, size_t numel, size_t cols) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        y[i] = from_float<T>(to_float(y[i]) + to_float(bias[i % cols]));
    }
}

mcblasHandle_t get_handle() {
    static mcblasHandle_t handle = [] {
        mcblasHandle_t h = nullptr;
        if (mcblasCreate(&h) != MCBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("mcblasCreate failed");
        }
        return h;
    }();
    return handle;
}

template <typename T>
macaDataType maca_type();

template <>
macaDataType maca_type<float>() { return MACA_R_32F; }

template <>
macaDataType maca_type<__half>() { return MACA_R_16F; }

template <>
macaDataType maca_type<__maca_bfloat16>() { return MACA_R_16BF; }

template <typename T>
void linear_impl(T *y, const T *x, const T *w, const T *bias,
                 size_t rows, size_t out_cols, size_t inner) {
    // 行主序 Y = X·Wᵀ 等价于列主序 Yᵀ = Wᵀ·Xᵀ 的内存形式：
    //   A = W 按列主序解释为 [inner, out_cols]（lda=inner），transa=T 得到 Wᵀ [out_cols, inner]
    //   B = X 按列主序解释为 [inner, rows]（ldb=inner），transb=N
    //   C = Y 按列主序解释为 [out_cols, rows]（ldc=out_cols）
    float alpha = 1.0f;
    float beta = 0.0f;
    macaDataType t = maca_type<T>();
    mcblasStatus_t st = mcblasGemmEx(
        get_handle(),
        MCBLAS_OP_T, MCBLAS_OP_N,
        static_cast<int>(out_cols), static_cast<int>(rows), static_cast<int>(inner),
        &alpha,
        w, t, static_cast<int>(inner),
        x, t, static_cast<int>(inner),
        &beta,
        y, t, static_cast<int>(out_cols),
        MCBLAS_COMPUTE_32F_PEDANTIC,
        MCBLAS_GEMM_DEFAULT);
    if (st != MCBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("mcblasGemmEx failed");
    }

    if (bias != nullptr) {
        size_t numel = rows * out_cols;
        unsigned int grid = static_cast<unsigned int>((numel + 255) / 256);
        add_bias_kernel<T><<<grid, 256>>>(y, bias, numel, out_cols);
    }
    mcDeviceSynchronize();
}

} // namespace

void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t rows, size_t out_cols, size_t inner) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        linear_impl(reinterpret_cast<float *>(out),
                    reinterpret_cast<const float *>(in),
                    reinterpret_cast<const float *>(weight),
                    reinterpret_cast<const float *>(bias),
                    rows, out_cols, inner);
        break;
    case LLAISYS_DTYPE_F16:
        linear_impl(reinterpret_cast<__half *>(out),
                    reinterpret_cast<const __half *>(in),
                    reinterpret_cast<const __half *>(weight),
                    reinterpret_cast<const __half *>(bias),
                    rows, out_cols, inner);
        break;
    case LLAISYS_DTYPE_BF16:
        linear_impl(reinterpret_cast<__maca_bfloat16 *>(out),
                    reinterpret_cast<const __maca_bfloat16 *>(in),
                    reinterpret_cast<const __maca_bfloat16 *>(weight),
                    reinterpret_cast<const __maca_bfloat16 *>(bias),
                    rows, out_cols, inner);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::maca
