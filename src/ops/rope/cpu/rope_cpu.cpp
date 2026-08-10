#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

namespace llaisys::ops::cpu {
namespace {

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids,
           size_t seq_len, size_t n_heads, size_t head_dim, float theta) {
    size_t half = head_dim / 2;
    std::vector<float> sin(half);
    std::vector<float> cos(half);
    for (size_t s = 0; s < seq_len; s++) {
        float pos = static_cast<float>(pos_ids[s]);
        for (size_t j = 0; j < half; j++) {
            float angle = pos / std::pow(theta, 2.0f * static_cast<float>(j) / static_cast<float>(head_dim));
            sin[j] = std::sin(angle);
            cos[j] = std::cos(angle);
        }

        for (size_t h = 0; h < n_heads; h++) {
            const T *vec = in + (s * n_heads + h) * head_dim;
            T *ovec = out + (s * n_heads + h) * head_dim;
            for (size_t j = 0; j < half; j++) {
                float a = llaisys::utils::cast<float>(vec[j]);
                float b = llaisys::utils::cast<float>(vec[j + half]);
                ovec[j] = llaisys::utils::cast<T>(a * cos[j] - b * sin[j]);
                ovec[j + half] = llaisys::utils::cast<T>(b * cos[j] + a * sin[j]);
            }
        }
    }
}

} // namespace

void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          llaisysDataType_t type, size_t seq_len, size_t n_heads, size_t head_dim,
          float theta) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out),
                     reinterpret_cast<const float *>(in),
                     reinterpret_cast<const int64_t *>(pos_ids),
                     seq_len, n_heads, head_dim, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out),
                     reinterpret_cast<const llaisys::bf16_t *>(in),
                     reinterpret_cast<const int64_t *>(pos_ids),
                     seq_len, n_heads, head_dim, theta);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out),
                     reinterpret_cast<const llaisys::fp16_t *>(in),
                     reinterpret_cast<const int64_t *>(pos_ids),
                     seq_len, n_heads, head_dim, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
