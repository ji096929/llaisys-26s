#include "maca_resource.cuh"

namespace llaisys::device::maca {

Resource::Resource(int device_id) : llaisys::device::DeviceResource(LLAISYS_DEVICE_MACA, device_id) {}

} // namespace llaisys::device::maca
