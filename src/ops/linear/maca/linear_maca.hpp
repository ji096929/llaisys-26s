#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::maca {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t rows, size_t out_cols, size_t inner);
}
