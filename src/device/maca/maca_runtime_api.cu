#include "../runtime_api.hpp"

#include <mc_runtime.h>
#include <mc_common.h>

#include <stdexcept>
#include <string>

namespace llaisys::device::maca {

namespace runtime_api {
namespace {

void check(mcError_t err, const char *what) {
    if (err != mcSuccess) {
        throw std::runtime_error(std::string(what) + ": " + mcGetErrorString(err));
    }
}

mcMemcpyKind to_maca_kind(llaisysMemcpyKind_t kind) {
    switch (kind) {
    case LLAISYS_MEMCPY_H2H:
        return mcMemcpyHostToHost;
    case LLAISYS_MEMCPY_H2D:
        return mcMemcpyHostToDevice;
    case LLAISYS_MEMCPY_D2H:
        return mcMemcpyDeviceToHost;
    case LLAISYS_MEMCPY_D2D:
        return mcMemcpyDeviceToDevice;
    default:
        throw std::runtime_error("invalid memcpy kind");
    }
}

int getDeviceCount() {
    int count = 0;
    check(mcGetDeviceCount(&count), "mcGetDeviceCount");
    return count;
}

void setDevice(int device) {
    check(mcSetDevice(device), "mcSetDevice");
}

void deviceSynchronize() {
    check(mcDeviceSynchronize(), "mcDeviceSynchronize");
}

llaisysStream_t createStream() {
    mcStream_t stream = nullptr;
    check(mcStreamCreate(&stream), "mcStreamCreate");
    return reinterpret_cast<llaisysStream_t>(stream);
}

void destroyStream(llaisysStream_t stream) {
    check(mcStreamDestroy(reinterpret_cast<mcStream_t>(stream)), "mcStreamDestroy");
}

void streamSynchronize(llaisysStream_t stream) {
    check(mcStreamSynchronize(reinterpret_cast<mcStream_t>(stream)), "mcStreamSynchronize");
}

void *mallocDevice(size_t size) {
    void *ptr = nullptr;
    check(mcMalloc(&ptr, size), "mcMalloc");
    return ptr;
}

void freeDevice(void *ptr) {
    check(mcFree(ptr), "mcFree");
}

void *mallocHost(size_t size) {
    void *ptr = nullptr;
    check(mcMallocHost(&ptr, size, mcMallocHostDefault), "mcMallocHost");
    return ptr;
}

void freeHost(void *ptr) {
    check(mcFreeHost(ptr), "mcFreeHost");
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    check(mcMemcpy(dst, src, size, to_maca_kind(kind)), "mcMemcpy");
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind,
                 llaisysStream_t stream) {
    check(mcMemcpyAsync(dst, src, size, to_maca_kind(kind),
                        reinterpret_cast<mcStream_t>(stream)),
          "mcMemcpyAsync");
}

} // namespace

static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync};

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}
} // namespace llaisys::device::maca
