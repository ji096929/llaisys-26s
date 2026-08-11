#include "qwen2.hpp"

#include "../../utils.hpp"

namespace llaisys {
namespace model {

Qwen2Model::Qwen2Model(const LlaisysQwen2Meta &meta, llaisysDeviceType_t device, int device_id)
    : _meta(meta), _device(device), _device_id(device_id) {
    size_t hs = meta.hs;
    size_t nh = meta.nh;
    size_t nkvh = meta.nkvh;
    size_t dh = meta.dh;
    size_t di = meta.di;
    size_t nlayer = meta.nlayer;
    size_t voc = meta.voc;

    // 单个权重
    _weights.in_embed = make_handle(create_tensor({voc, hs}));
    _weights.out_embed = make_handle(create_tensor({voc, hs}));
    _weights.out_norm_w = make_handle(create_tensor({hs}));

    // 每层权重
    _attn_norm_w.resize(nlayer);
    _attn_q_w.resize(nlayer);
    _attn_q_b.resize(nlayer);
    _attn_k_w.resize(nlayer);
    _attn_k_b.resize(nlayer);
    _attn_v_w.resize(nlayer);
    _attn_v_b.resize(nlayer);
    _attn_o_w.resize(nlayer);
    _mlp_norm_w.resize(nlayer);
    _mlp_gate_w.resize(nlayer);
    _mlp_up_w.resize(nlayer);
    _mlp_down_w.resize(nlayer);

    for (size_t i = 0; i < nlayer; i++) {
        _attn_norm_w[i] = make_handle(create_tensor({hs}));
        _attn_q_w[i] = make_handle(create_tensor({nh * dh, hs}));
        _attn_q_b[i] = make_handle(create_tensor({nh * dh}));
        _attn_k_w[i] = make_handle(create_tensor({nkvh * dh, hs}));
        _attn_k_b[i] = make_handle(create_tensor({nkvh * dh}));
        _attn_v_w[i] = make_handle(create_tensor({nkvh * dh, hs}));
        _attn_v_b[i] = make_handle(create_tensor({nkvh * dh}));
        _attn_o_w[i] = make_handle(create_tensor({hs, nh * dh}));
        _mlp_norm_w[i] = make_handle(create_tensor({hs}));
        _mlp_gate_w[i] = make_handle(create_tensor({di, hs}));
        _mlp_up_w[i] = make_handle(create_tensor({di, hs}));
        _mlp_down_w[i] = make_handle(create_tensor({hs, di}));
    }

    _weights.attn_norm_w = _attn_norm_w.data();
    _weights.attn_q_w = _attn_q_w.data();
    _weights.attn_q_b = _attn_q_b.data();
    _weights.attn_k_w = _attn_k_w.data();
    _weights.attn_k_b = _attn_k_b.data();
    _weights.attn_v_w = _attn_v_w.data();
    _weights.attn_v_b = _attn_v_b.data();
    _weights.attn_o_w = _attn_o_w.data();
    _weights.mlp_norm_w = _mlp_norm_w.data();
    _weights.mlp_gate_w = _mlp_gate_w.data();
    _weights.mlp_up_w = _mlp_up_w.data();
    _weights.mlp_down_w = _mlp_down_w.data();

    // KV cache：每层 [maxseq, nkvh, dh]
    for (size_t i = 0; i < nlayer; i++) {
        _k_cache.push_back(create_tensor({meta.maxseq, nkvh, dh}));
        _v_cache.push_back(create_tensor({meta.maxseq, nkvh, dh}));
        _cache_len.push_back(0);
    }
}

tensor_t Qwen2Model::create_tensor(const std::vector<size_t> &shape) {
    return Tensor::create(shape, _meta.dtype, _device, _device_id);
}

llaisysTensor_t Qwen2Model::make_handle(tensor_t tensor) {
    auto handle = std::make_shared<LlaisysTensor>(LlaisysTensor{tensor});
    _owned.push_back(handle);
    return handle.get();
}

int64_t Qwen2Model::infer(const int64_t *token_ids, size_t ntoken) {
    // TODO: 阶段 6 实现完整前向
    TO_BE_IMPLEMENTED();
    return _meta.end_token;
}

} // namespace model
} // namespace llaisys
