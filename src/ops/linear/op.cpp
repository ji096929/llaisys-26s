#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    CHECK_ARGUMENT(out->shape()[0] == in->shape()[0]
                       && out->shape()[1] == weight->shape()[0]
                       && in->shape()[1] == weight->shape()[1],
                   "linear: shape mismatch");
    if (bias) {
        CHECK_SAME_DEVICE(out, bias);
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
        CHECK_ARGUMENT(bias->numel() == weight->shape()[0], "linear: bias shape mismatch");
    }
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous()
               && (!bias || bias->isContiguous()),
           "linear: all tensors must be contiguous.");

    size_t rows = in->shape()[0];
    size_t out_cols = weight->shape()[0];
    size_t inner = weight->shape()[1];

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(),
                           bias ? bias->data() : nullptr,
                           out->dtype(), rows, out_cols, inner);
    }

    core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(),
                           bias ? bias->data() : nullptr,
                           out->dtype(), rows, out_cols, inner);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
