#pragma once

#include "llaisys/models/qwen2.h"

#include "../../tensor/tensor.hpp"
#include "../../llaisys/llaisys_tensor.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace llaisys {
namespace model {

class Qwen2Model {
public:
    Qwen2Model(const LlaisysQwen2Meta &meta, llaisysDeviceType_t device, int device_id);
    ~Qwen2Model() = default;

    Qwen2Model(const Qwen2Model &) = delete;
    Qwen2Model &operator=(const Qwen2Model &) = delete;

    const LlaisysQwen2Meta &meta() const { return _meta; }
    LlaisysQwen2Weights *weights() { return &_weights; }

    // 前向：阶段 6 实现，输入 token ids，返回 argmax 的下一个 token
    int64_t infer(const int64_t *token_ids, size_t ntoken);

private:
    tensor_t create_tensor(const std::vector<size_t> &shape);
    llaisysTensor_t make_handle(tensor_t tensor);
    tensor_t buffer(tensor_t &slot, const std::vector<size_t> &shape);
    tensor_t int_buffer(tensor_t &slot, size_t n);
    tensor_t forward_layer(size_t layer, tensor_t hidden, const std::vector<int64_t> &pos_ids);
    tensor_t forward(const int64_t *token_ids, size_t ntoken);
    void append_cache(tensor_t cache, tensor_t new_kv, size_t offset);

    LlaisysQwen2Meta _meta;
    llaisysDeviceType_t _device;
    int _device_id;

    // 权重句柄容器（对外暴露给 Python 填数据）
    LlaisysQwen2Weights _weights;
    std::vector<std::shared_ptr<LlaisysTensor>> _owned;
    std::vector<llaisysTensor_t> _attn_norm_w;
    std::vector<llaisysTensor_t> _attn_q_w;
    std::vector<llaisysTensor_t> _attn_q_b;
    std::vector<llaisysTensor_t> _attn_k_w;
    std::vector<llaisysTensor_t> _attn_k_b;
    std::vector<llaisysTensor_t> _attn_v_w;
    std::vector<llaisysTensor_t> _attn_v_b;
    std::vector<llaisysTensor_t> _attn_o_w;
    std::vector<llaisysTensor_t> _mlp_norm_w;
    std::vector<llaisysTensor_t> _mlp_gate_w;
    std::vector<llaisysTensor_t> _mlp_up_w;
    std::vector<llaisysTensor_t> _mlp_down_w;

    // KV cache：每层一份 k/v，长度记录已缓存 token 数
    std::vector<tensor_t> _k_cache;
    std::vector<tensor_t> _v_cache;
    std::vector<size_t> _cache_len;

    // 预分配缓冲区：跨层复用，按需扩容，避免每步每层分配
    tensor_t _buf_ids;
    tensor_t _buf_pos;
    tensor_t _buf_hidden;
    tensor_t _buf_h_norm;
    tensor_t _buf_q;
    tensor_t _buf_k;
    tensor_t _buf_v;
    tensor_t _buf_attn;
    tensor_t _buf_o;
    tensor_t _buf_h_mlp;
    tensor_t _buf_gate;
    tensor_t _buf_up;
    tensor_t _buf_mlp_out;
    tensor_t _buf_down;
    tensor_t _buf_logits;
};

} // namespace model
} // namespace llaisys
