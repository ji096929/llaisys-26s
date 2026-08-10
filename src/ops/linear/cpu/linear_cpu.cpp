#include "linear_cpu.hpp"

#include "../../../utils.hpp"

namespace llaisys::ops::cpu {
namespace {

template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias,
             size_t rows, size_t out_cols, size_t inner) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < out_cols; j++) {
            float acc = 0.0f;
            for (size_t k = 0; k < inner; k++) {
                acc += llaisys::utils::cast<float>(in[i * inner + k])
                     * llaisys::utils::cast<float>(weight[j * inner + k]);
            }
            if (bias != nullptr) {
                acc += llaisys::utils::cast<float>(bias[j]);
            }
            out[i * out_cols + j] = llaisys::utils::cast<T>(acc);
        }
    }
}

} // namespace

void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t rows, size_t out_cols, size_t inner) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out),
                       reinterpret_cast<const float *>(in),
                       reinterpret_cast<const float *>(weight),
                       reinterpret_cast<const float *>(bias),
                       rows, out_cols, inner);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out),
                       reinterpret_cast<const llaisys::bf16_t *>(in),
                       reinterpret_cast<const llaisys::bf16_t *>(weight),
                       reinterpret_cast<const llaisys::bf16_t *>(bias),
                       rows, out_cols, inner);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out),
                       reinterpret_cast<const llaisys::fp16_t *>(in),
                       reinterpret_cast<const llaisys::fp16_t *>(weight),
                       reinterpret_cast<const llaisys::fp16_t *>(bias),
                       rows, out_cols, inner);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
