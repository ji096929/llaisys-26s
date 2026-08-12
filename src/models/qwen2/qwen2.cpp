#include "qwen2.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"
#include "../../ops/add/op.hpp"

#include <cmath>
#include <limits>

namespace llaisys {
namespace model {
namespace {

template <typename T>
int64_t argmax_last(const std::byte *data, size_t size) {
    const T *p = reinterpret_cast<const T *>(data);
    int64_t best = 0;
    float best_v = -std::numeric_limits<float>::infinity();
    for (size_t j = 0; j < size; j++) {
        float v = llaisys::utils::cast<float>(p[j]);
        if (v > best_v) {
            best_v = v;
            best = static_cast<int64_t>(j);
        }
    }
    return best;
}

int64_t argmax_logits(const std::byte *data, llaisysDataType_t dtype, size_t size) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return argmax_last<float>(data, size);
    case LLAISYS_DTYPE_BF16:
        return argmax_last<llaisys::bf16_t>(data, size);
    case LLAISYS_DTYPE_F16:
        return argmax_last<llaisys::fp16_t>(data, size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace

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

tensor_t Qwen2Model::buffer(tensor_t &slot, const std::vector<size_t> &shape) {
    if (!slot || slot->shape() != shape) {
        slot = create_tensor(shape);
    }
    return slot;
}

tensor_t Qwen2Model::int_buffer(tensor_t &slot, size_t n) {
    if (!slot || slot->numel() != n) {
        slot = Tensor::create({n}, LLAISYS_DTYPE_I64, _device, _device_id);
    }
    return slot;
}

llaisysTensor_t Qwen2Model::make_handle(tensor_t tensor) {
    auto handle = std::make_shared<LlaisysTensor>(LlaisysTensor{tensor});
    _owned.push_back(handle);
    return handle.get();
}

int64_t Qwen2Model::infer(const int64_t *token_ids, size_t ntoken) {
    auto logits = forward(token_ids, ntoken);
    // 只需要最后一个 token 的 logits
    auto last_logits = logits->slice(0, ntoken - 1, ntoken);
    size_t voc = _meta.voc;
    size_t bytes = voc * llaisys::utils::dsize(_meta.dtype);

    if (logits->deviceType() == LLAISYS_DEVICE_CPU) {
        return argmax_logits(last_logits->data(), _meta.dtype, voc);
    }

    // 设备（GPU）上：logits 在显存里，CPU 不能直接读，拷回 host 再 argmax
    std::vector<std::byte> host_logits(bytes);
    core::context().setDevice(logits->deviceType(), logits->deviceId());
    core::context().runtime().api()->memcpy_sync(
        host_logits.data(), last_logits->data(), bytes, LLAISYS_MEMCPY_D2H);
    return argmax_logits(host_logits.data(), _meta.dtype, voc);
}

tensor_t Qwen2Model::forward(const int64_t *token_ids, size_t ntoken) {
    size_t hs = _meta.hs;

    // token ids 张量
    auto ids = int_buffer(_buf_ids, ntoken);
    ids->load(token_ids);

    // embedding
    auto hidden = buffer(_buf_hidden, {ntoken, hs});
    ops::embedding(hidden, ids, _weights.in_embed->tensor);

    // 位置从当前缓存长度开始（所有层同步）
    size_t base = _cache_len.empty() ? 0 : _cache_len[0];
    std::vector<int64_t> pos_ids(ntoken);
    for (size_t t = 0; t < ntoken; t++) {
        pos_ids[t] = static_cast<int64_t>(base + t);
    }

    for (size_t i = 0; i < _meta.nlayer; i++) {
        hidden = forward_layer(i, hidden, pos_ids);
    }

    // 最终 RMSNorm + lm_head
    auto h_norm = buffer(_buf_h_norm, {ntoken, hs});
    ops::rms_norm(h_norm, hidden, _weights.out_norm_w->tensor, _meta.epsilon);

    auto logits = buffer(_buf_logits, {ntoken, _meta.voc});
    ops::linear(logits, h_norm, _weights.out_embed->tensor, nullptr);

    return logits;
}

tensor_t Qwen2Model::forward_layer(size_t layer, tensor_t hidden,
                                   const std::vector<int64_t> &pos_ids) {
    size_t ntoken = pos_ids.size();
    size_t hs = _meta.hs;
    size_t nh = _meta.nh;
    size_t nkvh = _meta.nkvh;
    size_t dh = _meta.dh;
    size_t di = _meta.di;
    float eps = _meta.epsilon;
    float theta = _meta.theta;

    // 1. 注意力前的 RMSNorm
    auto h_norm = buffer(_buf_h_norm, {ntoken, hs});
    ops::rms_norm(h_norm, hidden, _weights.attn_norm_w[layer]->tensor, eps);

    // 2. q/k/v 投影
    auto q = buffer(_buf_q, {ntoken, nh * dh});
    auto k = buffer(_buf_k, {ntoken, nkvh * dh});
    auto v = buffer(_buf_v, {ntoken, nkvh * dh});
    ops::linear(q, h_norm, _weights.attn_q_w[layer]->tensor, _weights.attn_q_b[layer]->tensor);
    ops::linear(k, h_norm, _weights.attn_k_w[layer]->tensor, _weights.attn_k_b[layer]->tensor);
    ops::linear(v, h_norm, _weights.attn_v_w[layer]->tensor, _weights.attn_v_b[layer]->tensor);

    // 3. reshape 成 3D 并做 RoPE
    auto q3 = q->view({ntoken, nh, dh});
    auto k3 = k->view({ntoken, nkvh, dh});
    auto v3 = v->view({ntoken, nkvh, dh});
    auto pos_t = int_buffer(_buf_pos, ntoken);
    pos_t->load(pos_ids.data());
    ops::rope(q3, q3, pos_t, theta);
    ops::rope(k3, k3, pos_t, theta);

    // 4. 新 k/v 追加进缓存
    size_t cur = _cache_len[layer];
    append_cache(_k_cache[layer], k3, cur);
    append_cache(_v_cache[layer], v3, cur);
    _cache_len[layer] = cur + ntoken;

    // 5. attention（用缓存里的全部历史 k/v）
    size_t kvlen = cur + ntoken;
    auto k_all = _k_cache[layer]->slice(0, 0, kvlen);
    auto v_all = _v_cache[layer]->slice(0, 0, kvlen);
    auto attn = buffer(_buf_attn, {ntoken, nh, dh});
    float scale = 1.0f / std::sqrt(static_cast<float>(dh));
    ops::self_attention(attn, q3, k_all, v_all, scale);

    // 6. o 投影 + 残差
    auto attn2 = attn->view({ntoken, nh * dh});
    auto o = buffer(_buf_o, {ntoken, hs});
    ops::linear(o, attn2, _weights.attn_o_w[layer]->tensor, nullptr);
    ops::add(hidden, hidden, o);

    // 7. MLP
    auto h_mlp = buffer(_buf_h_mlp, {ntoken, hs});
    ops::rms_norm(h_mlp, hidden, _weights.mlp_norm_w[layer]->tensor, eps);

    auto gate = buffer(_buf_gate, {ntoken, di});
    auto up = buffer(_buf_up, {ntoken, di});
    ops::linear(gate, h_mlp, _weights.mlp_gate_w[layer]->tensor, nullptr);
    ops::linear(up, h_mlp, _weights.mlp_up_w[layer]->tensor, nullptr);

    auto mlp_out = buffer(_buf_mlp_out, {ntoken, di});
    ops::swiglu(mlp_out, gate, up);

    auto down = buffer(_buf_down, {ntoken, hs});
    ops::linear(down, mlp_out, _weights.mlp_down_w[layer]->tensor, nullptr);
    ops::add(hidden, hidden, down);

    return hidden;
}

void Qwen2Model::append_cache(tensor_t cache, tensor_t new_kv, size_t offset) {
    size_t ntoken = new_kv->shape()[0];
    auto dst = cache->slice(0, offset, offset + ntoken);
    dst->load(new_kv->data());
}

} // namespace model
} // namespace llaisys
