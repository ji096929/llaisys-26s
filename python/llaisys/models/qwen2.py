import json
from ctypes import byref, c_int
from pathlib import Path
from typing import Sequence

import safetensors

from ..libllaisys import LIB_LLAISYS, DataType, DeviceType
from ..libllaisys.qwen2 import LlaisysQwen2Meta


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)

        # 1. 读 config.json，填 LlaisysQwen2Meta
        with open(model_path / "config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)

        meta = LlaisysQwen2Meta()
        meta.dtype = int(DataType.BF16)
        meta.nlayer = int(cfg["num_hidden_layers"])
        meta.hs = int(cfg["hidden_size"])
        meta.nh = int(cfg["num_attention_heads"])
        meta.nkvh = int(cfg["num_key_value_heads"])
        meta.dh = int(cfg["hidden_size"]) // int(cfg["num_attention_heads"])
        meta.di = int(cfg["intermediate_size"])
        meta.maxseq = int(cfg["max_position_embeddings"])
        meta.voc = int(cfg["vocab_size"])
        meta.epsilon = float(cfg["rms_norm_eps"])
        meta.theta = float(cfg["rope_theta"])
        meta.end_token = int(cfg.get("eos_token_id", 0))

        # 2. 创建 C++ 模型，拿到权重容器
        device_id = c_int(0)
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            byref(meta), int(device), byref(device_id), 1
        )
        if not self._model:
            raise RuntimeError("Failed to create Qwen2 model")
        self._weights = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model).contents

        # 3. 遍历 safetensors，按名字映射 load 进 C++ 张量
        for file in sorted(model_path.glob("*.safetensors")):
            with safetensors.safe_open(file, framework="torch", device="cpu") as data_:
                for name in data_.keys():
                    tensor = data_.get_tensor(name)
                    handle = self._map_weight(name)
                    if handle is None:
                        raise RuntimeError(f"Unknown weight name: {name}")
                    LIB_LLAISYS.tensorLoad(handle, tensor.data_ptr())

    def __del__(self):
        if getattr(self, "_model", None):
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
            self._model = None

    def _map_weight(self, name: str):
        """transformers 权重名 → LlaisysQwen2Weights 里的句柄。"""
        w = self._weights
        if name == "model.embed_tokens.weight":
            return w.in_embed
        if name == "lm_head.weight":
            return w.out_embed
        if name == "model.norm.weight":
            return w.out_norm_w

        parts = name.split(".")
        if len(parts) >= 4 and parts[0] == "model" and parts[1] == "layers":
            layer = int(parts[2])
            sub = parts[3]
            if sub == "input_layernorm":
                return w.attn_norm_w[layer]
            if sub == "post_attention_layernorm":
                return w.mlp_norm_w[layer]
            if sub == "self_attn" and len(parts) >= 6:
                proj = parts[4]
                is_weight = parts[5] == "weight"
                if proj == "q_proj":
                    return w.attn_q_w[layer] if is_weight else w.attn_q_b[layer]
                if proj == "k_proj":
                    return w.attn_k_w[layer] if is_weight else w.attn_k_b[layer]
                if proj == "v_proj":
                    return w.attn_v_w[layer] if is_weight else w.attn_v_b[layer]
                if proj == "o_proj":
                    return w.attn_o_w[layer]
            if sub == "mlp" and len(parts) >= 5:
                proj = parts[4]
                if proj == "gate_proj":
                    return w.mlp_gate_w[layer]
                if proj == "up_proj":
                    return w.mlp_up_w[layer]
                if proj == "down_proj":
                    return w.mlp_down_w[layer]
        return None

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        # TODO: 阶段 7 实现生成循环
        return []
