#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

namespace llaisys::ops::cpu {
namespace {

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, size_t rows, size_t cols, float eps) {
    for (size_t i = 0; i < rows; i++) {
        const T *row_in = in + i * cols;
        T *row_out = out + i * cols;

        float sum = 0.0f;
        for (size_t j = 0; j < cols; j++) {
            float v = llaisys::utils::cast<float>(row_in[j]);
            sum += v * v;
        }

        float rms = std::sqrt(sum / static_cast<float>(cols) + eps);
        float scale = 1.0f / rms;

        for (size_t j = 0; j < cols; j++) {
            row_out[j] = llaisys::utils::cast<T>(
                llaisys::utils::cast<float>(row_in[j]) * scale
                * llaisys::utils::cast<float>(weight[j]));
        }
    }
}

} // namespace

void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              llaisysDataType_t type, size_t rows, size_t cols, float eps) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out),
                         reinterpret_cast<const float *>(in),
                         reinterpret_cast<const float *>(weight),
                         rows, cols, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out),
                         reinterpret_cast<const llaisys::bf16_t *>(in),
                         reinterpret_cast<const llaisys::bf16_t *>(weight),
                         rows, cols, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out),
                         reinterpret_cast<const llaisys::fp16_t *>(in),
                         reinterpret_cast<const llaisys::fp16_t *>(weight),
                         rows, cols, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
