#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <limits>
#include <vector>

namespace llaisys::ops::cpu {
namespace {

template <typename T>
void self_attention_(T *out, const T *q, const T *k, const T *v,
                     size_t qlen, size_t kvlen, size_t nh, size_t nkvh, size_t hd,
                     float scale) {
    size_t repeat = nh / nkvh;
    std::vector<float> scores(kvlen);

    for (size_t i = 0; i < qlen; i++) {
        for (size_t h = 0; h < nh; h++) {
            size_t kh = h / repeat;
            size_t q_head_base = (i * nh + h) * hd;

            // 第 1、2 步：打分 + causal 屏蔽
            float max_s = -std::numeric_limits<float>::infinity();
            for (size_t j = 0; j < kvlen; j++) {
                float s = 0.0f;
                for (size_t m = 0; m < hd; m++) {
                    s += llaisys::utils::cast<float>(q[q_head_base + m])
                       * llaisys::utils::cast<float>(k[(j * nkvh + kh) * hd + m]);
                }
                s *= scale;
                if (j > i + (kvlen - qlen)) {
                    s = -std::numeric_limits<float>::infinity();
                }
                scores[j] = s;
                if (s > max_s) {
                    max_s = s;
                }
            }

            // 第 3 步：softmax（减最大值，数值稳定）
            float sum = 0.0f;
            for (size_t j = 0; j < kvlen; j++) {
                scores[j] = std::exp(scores[j] - max_s);
                sum += scores[j];
            }

            // 第 4 步：加权求和
            T *ovec = out + q_head_base;
            for (size_t m = 0; m < hd; m++) {
                float acc = 0.0f;
                for (size_t j = 0; j < kvlen; j++) {
                    acc += scores[j] * llaisys::utils::cast<float>(v[(j * nkvh + kh) * hd + m]);
                }
                ovec[m] = llaisys::utils::cast<T>(acc / sum);
            }
        }
    }
}

} // namespace

void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t type,
                    size_t qlen, size_t kvlen, size_t nh, size_t nkvh, size_t hd,
                    float scale) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val),
                               reinterpret_cast<const float *>(q),
                               reinterpret_cast<const float *>(k),
                               reinterpret_cast<const float *>(v),
                               qlen, kvlen, nh, nkvh, hd, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(attn_val),
                               reinterpret_cast<const llaisys::bf16_t *>(q),
                               reinterpret_cast<const llaisys::bf16_t *>(k),
                               reinterpret_cast<const llaisys::bf16_t *>(v),
                               qlen, kvlen, nh, nkvh, hd, scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(attn_val),
                               reinterpret_cast<const llaisys::fp16_t *>(q),
                               reinterpret_cast<const llaisys::fp16_t *>(k),
                               reinterpret_cast<const llaisys::fp16_t *>(v),
                               qlen, kvlen, nh, nkvh, hd, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
