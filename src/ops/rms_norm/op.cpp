#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/rms_norm_nvidia.hpp"
#endif

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    CHECK_ARGUMENT(out->shape() == in->shape(), "rms_norm: out/in shape mismatch");
    CHECK_ARGUMENT(weight->numel() == in->shape()[1], "rms_norm: weight must match last dim");
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "rms_norm: all tensors must be contiguous.");

    size_t rows = in->shape()[0];
    size_t cols = in->shape()[1];

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(out->data(), in->data(), weight->data(),
                             out->dtype(), rows, cols, eps);
    }

    core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out->data(), in->data(), weight->data(),
                             out->dtype(), rows, cols, eps);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rms_norm(out->data(), in->data(), weight->data(),
                                out->dtype(), in->shape()[0], in->shape()[1], eps);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
