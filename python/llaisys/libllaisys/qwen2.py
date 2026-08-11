from ctypes import POINTER, Structure, c_float, c_int, c_int64, c_size_t, c_void_p

from .llaisys_types import llaisysDataType_t, llaisysDeviceType_t
from .tensor import llaisysTensor_t


class LlaisysQwen2Meta(Structure):
    _fields_ = [
        ("dtype", llaisysDataType_t),  # 权重数据类型
        ("nlayer", c_size_t),          # 层数
        ("hs", c_size_t),              # hidden_size
        ("nh", c_size_t),              # query 头数
        ("nkvh", c_size_t),            # key/value 头数
        ("dh", c_size_t),              # head_dim
        ("di", c_size_t),              # intermediate_size
        ("maxseq", c_size_t),          # 最大序列长度
        ("voc", c_size_t),             # 词表大小
        ("epsilon", c_float),          # rms_norm_eps
        ("theta", c_float),            # rope_theta
        ("end_token", c_int64),        # eos_token_id
    ]


class LlaisysQwen2Weights(Structure):
    _fields_ = [
        ("in_embed", llaisysTensor_t),
        ("out_embed", llaisysTensor_t),
        ("out_norm_w", llaisysTensor_t),
        ("attn_norm_w", POINTER(llaisysTensor_t)),
        ("attn_q_w", POINTER(llaisysTensor_t)),
        ("attn_q_b", POINTER(llaisysTensor_t)),
        ("attn_k_w", POINTER(llaisysTensor_t)),
        ("attn_k_b", POINTER(llaisysTensor_t)),
        ("attn_v_w", POINTER(llaisysTensor_t)),
        ("attn_v_b", POINTER(llaisysTensor_t)),
        ("attn_o_w", POINTER(llaisysTensor_t)),
        ("mlp_norm_w", POINTER(llaisysTensor_t)),
        ("mlp_gate_w", POINTER(llaisysTensor_t)),
        ("mlp_up_w", POINTER(llaisysTensor_t)),
        ("mlp_down_w", POINTER(llaisysTensor_t)),
    ]


def load_qwen2(lib):
    lib.llaisysQwen2ModelCreate.argtypes = [
        POINTER(LlaisysQwen2Meta),
        llaisysDeviceType_t,
        POINTER(c_int),  # device_ids
        c_int,           # ndevice
    ]
    lib.llaisysQwen2ModelCreate.restype = c_void_p

    lib.llaisysQwen2ModelDestroy.argtypes = [c_void_p]
    lib.llaisysQwen2ModelDestroy.restype = None

    lib.llaisysQwen2ModelWeights.argtypes = [c_void_p]
    lib.llaisysQwen2ModelWeights.restype = POINTER(LlaisysQwen2Weights)

    lib.llaisysQwen2ModelInfer.argtypes = [
        c_void_p,          # model
        POINTER(c_int64),  # token_ids
        c_size_t,          # ntoken
    ]
    lib.llaisysQwen2ModelInfer.restype = c_int64
