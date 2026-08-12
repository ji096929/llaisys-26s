#include "self_attention_nvidia.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "../../../utils.hpp"

namespace llaisys::ops::nvidia {
namespace {

constexpr unsigned int BLOCK = 256;

__device__ float to_float(float v) { return v; }
__device__ float to_float(__half v) { return __half2float(v); }
__device__ float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T>
__device__ T from_float(float v);

template <>
__device__ float from_float<float>(float v) { return v; }

template <>
__device__ __half from_float<__half>(float v) { return __float2half(v); }

template <>
__device__ __nv_bfloat16 from_float<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

template <typename T>
__global__ void score_kernel(float *scores, const T *q, const T *k,
                             size_t qlen, size_t kvlen, size_t nh, size_t nkvh,
                             size_t hd, float scale) {
    size_t ih = blockIdx.x;
    if (ih >= qlen * nh) {
        return;
    }
    size_t i = ih / nh;
    size_t h = ih % nh;
    size_t kh = h / (nh / nkvh);

    const T *qvec = q + ih * hd;
    const T *kbase = k + kh * hd;
    float *row = scores + ih * kvlen;
    size_t limit = i + (kvlen - qlen); // causal: 只允许 j <= limit

    for (size_t j = threadIdx.x; j < kvlen; j += blockDim.x) {
        float s = -INFINITY;
        if (j <= limit) {
            const T *kvec = kbase + j * nkvh * hd;
            float acc = 0.0f;
            for (size_t m = 0; m < hd; m++) {
                acc += to_float(qvec[m]) * to_float(kvec[m]);
            }
            s = acc * scale;
        }
        row[j] = s;
    }
}

__global__ void softmax_kernel(float *scores, size_t qlen, size_t nh, size_t kvlen) {
    size_t ih = blockIdx.x;
    if (ih >= qlen * nh) {
        return;
    }
    float *row = scores + ih * kvlen;
    unsigned int tid = threadIdx.x;

    __shared__ float s_max[BLOCK];
    __shared__ float s_sum[BLOCK];

    float mx = -INFINITY;
    for (size_t j = tid; j < kvlen; j += BLOCK) {
        mx = fmaxf(mx, row[j]);
    }
    s_max[tid] = mx;
    __syncthreads();
    for (unsigned int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
        }
        __syncthreads();
    }

    float sum = 0.0f;
    for (size_t j = tid; j < kvlen; j += BLOCK) {
        float e = expf(row[j] - s_max[0]);
        row[j] = e;
        sum += e;
    }
    s_sum[tid] = sum;
    __syncthreads();
    for (unsigned int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }

    float inv = 1.0f / s_sum[0];
    for (size_t j = tid; j < kvlen; j += BLOCK) {
        row[j] *= inv;
    }
}

template <typename T>
__global__ void output_kernel(T *out, const float *scores, const T *v,
                              size_t qlen, size_t kvlen, size_t nh, size_t nkvh,
                              size_t hd) {
    size_t ih = blockIdx.x;
    if (ih >= qlen * nh) {
        return;
    }
    size_t h = ih % nh;
    size_t kh = h / (nh / nkvh);

    const float *prob = scores + ih * kvlen;
    const T *vbase = v + kh * hd;
    T *ovec = out + ih * hd;

    for (size_t m = threadIdx.x; m < hd; m += blockDim.x) {
        float acc = 0.0f;
        for (size_t j = 0; j < kvlen; j++) {
            acc += prob[j] * to_float(vbase[j * nkvh * hd + m]);
        }
        ovec[m] = from_float<T>(acc);
    }
}

template <typename T>
void self_attention_impl(T *attn_val, const T *q, const T *k, const T *v,
                         size_t qlen, size_t kvlen, size_t nh, size_t nkvh,
                         size_t hd, float scale) {
    float *scores = nullptr;
    cudaMalloc(&scores, qlen * nh * kvlen * sizeof(float));

    unsigned int grid = static_cast<unsigned int>(qlen * nh);
    score_kernel<T><<<grid, BLOCK>>>(scores, q, k, qlen, kvlen, nh, nkvh, hd, scale);
    softmax_kernel<<<grid, BLOCK>>>(scores, qlen, nh, kvlen);
    output_kernel<T><<<grid, BLOCK>>>(attn_val, scores, v, qlen, kvlen, nh, nkvh, hd);

    cudaFree(scores);
    cudaDeviceSynchronize();
}

} // namespace

void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t type,
                    size_t qlen, size_t kvlen, size_t nh, size_t nkvh, size_t hd,
                    float scale) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        self_attention_impl(reinterpret_cast<float *>(attn_val),
                            reinterpret_cast<const float *>(q),
                            reinterpret_cast<const float *>(k),
                            reinterpret_cast<const float *>(v),
                            qlen, kvlen, nh, nkvh, hd, scale);
        break;
    case LLAISYS_DTYPE_F16:
        self_attention_impl(reinterpret_cast<__half *>(attn_val),
                            reinterpret_cast<const __half *>(q),
                            reinterpret_cast<const __half *>(k),
                            reinterpret_cast<const __half *>(v),
                            qlen, kvlen, nh, nkvh, hd, scale);
        break;
    case LLAISYS_DTYPE_BF16:
        self_attention_impl(reinterpret_cast<__nv_bfloat16 *>(attn_val),
                            reinterpret_cast<const __nv_bfloat16 *>(q),
                            reinterpret_cast<const __nv_bfloat16 *>(k),
                            reinterpret_cast<const __nv_bfloat16 *>(v),
                            qlen, kvlen, nh, nkvh, hd, scale);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia
