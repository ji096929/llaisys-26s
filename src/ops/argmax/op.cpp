#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/argmax_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/argmax_nvidia.hpp"
#endif
namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());
    CHECK_ARGUMENT(max_idx->dtype() == LLAISYS_DTYPE_I64, "argmax: max_idx must be i64"); // max_idx 固定是 i64
    CHECK_ARGUMENT(vals->numel() > 0, "argmax: vals must not be empty");
    ASSERT(vals->isContiguous() && max_idx->isContiguous() && max_val->isContiguous(),
           "argmax: all tensors must be contiguous.");
    llaisys::core::context().setDevice(max_idx->deviceType(), max_idx->deviceId());

    switch (max_idx->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(),  vals->dtype(),vals->numel());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::argmax(max_idx->data(), max_val->data(), vals->data(),
                              vals->dtype(), vals->numel());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
