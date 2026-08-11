#include "embedding_cpu.hpp"
#include "../../../utils.hpp"

#include <cmath>
#include <cstring>

template <typename T>
void embedding_(T *out, const int64_t *index, const T *embd, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; i++) {
        const T *src = embd + index[i] * cols;
        T *dst = out + i * cols;
        std::memcpy(dst, src, cols * sizeof(T));
    }
}

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t type, size_t rows, size_t cols) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return embedding_(reinterpret_cast<float *>(out), reinterpret_cast<const int64_t *>(index), reinterpret_cast<const float *>(weight), rows, cols);
    case LLAISYS_DTYPE_BF16:
        return embedding_(reinterpret_cast<bf16_t *>(out), reinterpret_cast<const int64_t *>(index), reinterpret_cast<const bf16_t *>(weight), rows, cols);
    case LLAISYS_DTYPE_F16:
        return embedding_(reinterpret_cast<fp16_t *>(out), reinterpret_cast<const int64_t *>(index), reinterpret_cast<const fp16_t *>(weight), rows, cols);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
