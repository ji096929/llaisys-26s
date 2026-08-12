#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/self_attention_nvidia.hpp"
#endif
#ifdef ENABLE_MACA_API
#include "maca/self_attention_maca.hpp"
#endif

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    CHECK_ARGUMENT(q->ndim() == 3 && k->ndim() == 3 && v->ndim() == 3,
                   "self_attention: q/k/v must be 3D");
    CHECK_ARGUMENT(attn_val->shape() == q->shape(), "self_attention: attn_val/q shape mismatch");
    CHECK_ARGUMENT(k->shape()[0] == v->shape()[0] && k->shape()[2] == v->shape()[2],
                   "self_attention: k/v shape mismatch");
    CHECK_ARGUMENT(k->shape()[2] == q->shape()[2], "self_attention: head dim mismatch");
    CHECK_ARGUMENT(q->shape()[1] % k->shape()[1] == 0,
                   "self_attention: nh must be a multiple of nkvh");
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "self_attention: all tensors must be contiguous.");

    size_t qlen = q->shape()[0];
    size_t kvlen = k->shape()[0];
    size_t nh = q->shape()[1];
    size_t nkvh = k->shape()[1];
    size_t hd = q->shape()[2];

    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                   attn_val->dtype(), qlen, kvlen, nh, nkvh, hd, scale);
    }

    core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                   attn_val->dtype(), qlen, kvlen, nh, nkvh, hd, scale);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                      attn_val->dtype(), qlen, kvlen, nh, nkvh, hd, scale);
#endif
#ifdef ENABLE_MACA_API
    case LLAISYS_DEVICE_MACA:
        return maca::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                    attn_val->dtype(), qlen, kvlen, nh, nkvh, hd, scale);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
