#ifndef LLAISYS_MODELS_QWEN2_H
#define LLAISYS_MODELS_QWEN2_H

#include "../tensor.h"

__C {
    struct LlaisysQwen2Meta {
        llaisysDataType_t dtype; // 权重数据类型（如 LLAISYS_DTYPE_BF16）
        size_t nlayer;           // 层数：decoder 层数量（如 28）
        size_t hs;               // hidden_size：隐藏维度（如 1536）
        size_t nh;               // num_attention_heads：query 头数（如 12）
        size_t nkvh;             // num_key_value_heads：key/value 头数（GQA，如 2）
        size_t dh;               // head_dim：每个头的维度（hs / nh，如 128）
        size_t di;               // intermediate_size：MLP 中间维度（如 8960）
        size_t maxseq;           // max_position_embeddings：最大序列长度
        size_t voc;              // vocab_size：词表大小
        float epsilon;           // rms_norm_eps：RMSNorm 分母里加的小量
        float theta;             // rope_theta：RoPE 基频（如 10000.0）
        int64_t end_token;       // eos_token_id：生成结束符
    };

    struct LlaisysQwen2Weights {
        llaisysTensor_t in_embed;     // [voc, hs] 词嵌入表：token id → 向量（model.embed_tokens.weight）
        llaisysTensor_t out_embed;    // [voc, hs] lm_head：hidden → 词表分数（lm_head.weight）
        llaisysTensor_t out_norm_w;   // [hs] 最终 RMSNorm 缩放（model.norm.weight）
        llaisysTensor_t *attn_norm_w; // [hs] 每层注意力前 RMSNorm（input_layernorm.weight）
        llaisysTensor_t *attn_q_w;    // [nh*dh, hs] 每层 q 投影权重（self_attn.q_proj.weight）
        llaisysTensor_t *attn_q_b;    // [nh*dh] 每层 q 投影偏置（self_attn.q_proj.bias）
        llaisysTensor_t *attn_k_w;    // [nkvh*dh, hs] 每层 k 投影权重（self_attn.k_proj.weight）
        llaisysTensor_t *attn_k_b;    // [nkvh*dh] 每层 k 投影偏置（self_attn.k_proj.bias）
        llaisysTensor_t *attn_v_w;    // [nkvh*dh, hs] 每层 v 投影权重（self_attn.v_proj.weight）
        llaisysTensor_t *attn_v_b;    // [nkvh*dh] 每层 v 投影偏置（self_attn.v_proj.bias）
        llaisysTensor_t *attn_o_w;    // [hs, nh*dh] 每层注意力输出投影（self_attn.o_proj.weight）
        llaisysTensor_t *mlp_norm_w;  // [hs] 每层 MLP 前 RMSNorm（post_attention_layernorm.weight）
        llaisysTensor_t *mlp_gate_w;  // [di, hs] 每层 gate 投影（mlp.gate_proj.weight）
        llaisysTensor_t *mlp_up_w;    // [di, hs] 每层 up 投影（mlp.up_proj.weight）
        llaisysTensor_t *mlp_down_w;  // [hs, di] 每层 down 投影（mlp.down_proj.weight）
    };

    struct LlaisysQwen2Model;

    __export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice);

    __export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model);

    __export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model);

    __export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken);
}
#endif // LLAISYS_MODELS_QWEN2_H
