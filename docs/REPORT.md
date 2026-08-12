# LLAISYS-26S 作业报告

本报告记录 LLAISYS-26S 作业的复现流程、测试结果与各平台支持状态。测试模型为 DeepSeek-R1-Distill-Qwen-1.5B（BF16）。

## 1. 测试环境

| 平台 | 环境 |
|---|---|
| CPU | Windows 11，MSVC VS2022，xmake，Python 3.13，PyTorch 2.6.0 |
| NVIDIA | RTX 4050 Laptop（6 GiB，sm_89），驱动 572.83，CUDA Toolkit 12.8（NVCC 12.8.93），Python 3.13，PyTorch 2.6.0+cu126 |
| MetaX | MetaX 曦云 C500（切片卡，25% compute / 16 GiB），MACA 3.5.3，mxcc 1.0.0，Python 3.10，PyTorch 2.6.0+metax |

## 2. 复现流程

### 2.1 CPU

```bash
xmake
xmake install
pip install ./python/

python test/test_runtime.py --device cpu
python test/test_tensor.py
python test/ops/add.py
python test/ops/argmax.py
python test/ops/embedding.py
python test/ops/linear.py
python test/ops/rms_norm.py
python test/ops/rope.py
python test/ops/self_attention.py
python test/ops/swiglu.py
python test/test_infer.py --model [dir_path/to/model] --test
```

### 2.2 NVIDIA

```bash
xmake f --nv-gpu=y -cv
xmake
xmake install
pip install ./python/

python test/test_runtime.py --device nvidia
python test/ops/add.py --device nvidia
python test/ops/argmax.py --device nvidia
python test/ops/embedding.py --device nvidia
python test/ops/linear.py --device nvidia
python test/ops/rms_norm.py --device nvidia
python test/ops/rope.py --device nvidia
python test/ops/self_attention.py --device nvidia
python test/ops/swiglu.py --device nvidia
python test/test_infer.py --model [dir_path/to/model] --test --device nvidia
```

注：6 GiB 显存下运行完整推理时，建议通过 `Qwen2(..., max_seq=4096)` 限制 KV Cache 容量；测试输入长度远小于 4096，不影响正确性。

### 2.3 MetaX C500

```bash
# 准备沐曦构建环境（生成 .maca 驱动与垫片）
source scripts/maca_env.sh

xmake f --mx-gpu=y -cv
xmake
xmake install
pip install ./python/

python test/test_runtime.py --device maca
python test/ops/add.py --device maca
python test/ops/argmax.py --device maca
python test/ops/embedding.py --device maca
python test/ops/linear.py --device maca
python test/ops/rms_norm.py --device maca
python test/ops/rope.py --device maca
python test/ops/self_attention.py --device maca
python test/ops/swiglu.py --device maca
python test/test_infer.py --model <model_dir> --test --device maca
```

注：沐曦 PyTorch 使用 `cuda` 设备别名，参考实现与 LLAISYS 测试统一通过 `--device maca` 入口运行；`test_utils.py` 关闭了参考侧的 TF32，保证与 LLAISYS 严格 FP32 计算公平对比。

## 3. 测试结果

### 3.1 CI

GitHub Actions（Ubuntu Latest + Windows Latest）构建与 Assignment-0 ~ Assignment-3 测试全部通过。最近一次成功运行：run #8，提交 "fix: guard add_dll_directory when CUDA_PATH is unset"。https://github.com/ji096929/llaisys-26s/actions/runs/31565709908

### 3.2 CPU

Runtime、Tensor、八个算子（F32/F16/BF16）及 Qwen2 端到端推理全部通过，`test_infer.py --test` 输出 token 与 Hugging Face 完全一致。

推理耗时（8-token 生成，不含模型加载）：Hugging Face 约 1.9 s；LLAISYS 约 59 s（朴素单线程实现）。

### 3.3 NVIDIA

Runtime、八个算子（F32/F16/BF16）及 Qwen2 端到端推理全部通过，`test_infer.py --test --device nvidia` 输出 token 与 Hugging Face 完全一致。

推理耗时（128-token 生成，不含模型加载）：Hugging Face 约 7.1 s；LLAISYS 约 24.6 s（KV Cache 按 max_position_embeddings=131072 分配，显存压力较大）。限制 `max_seq=4096` 后单 token 生成约 42 ms。

### 3.4 MetaX C500

Runtime、八个算子（F32/F16/BF16）及 Qwen2 端到端推理全部通过，`test_infer.py --test --device maca` 输出 token 与 PyTorch 完全一致。

推理耗时（128-token 生成，不含模型加载，25% compute 切片卡）：PyTorch 约 3.57 s；LLAISYS 约 10.35 s。LLAISYS 较慢是因为算子内核以正确性优先，后续可参考性能优化方向进一步调优。

## 4. 平台支持状态

| 平台 | Runtime / Tensor | 八个算子 | Qwen2 推理 | 状态 |
|---|---|---|---|---|
| CPU | 支持 | F32/F16/BF16 支持 | CPU 推理、KV Cache 支持，token 与 HF 一致 | CI 与本地测试通过 |
| NVIDIA | 支持 | F32/F16/BF16 支持 | 单卡 BF16 推理、KV Cache 支持，token 与 HF 一致 | 本地测试通过 |
| MetaX C500 | 支持 | F32/F16/BF16 支持 | 单卡 BF16 推理、KV Cache 支持，token 与 HF 一致 | 本地测试通过 |

### 4.1 CPU

- 默认随项目构建，无需设备开关；
- 支持 Runtime、Tensor、八个算子和 Qwen2 单设备推理；
- Windows/Ubuntu CI 与本地测试均通过。

### 4.2 NVIDIA

- 支持设备与 Stream 管理、Device/Pinned Host 内存、同步/异步拷贝；
- 八个算子提供独立 `llaisys::ops::nvidia` 实现，支持 F32、F16、BF16；
- Linear 使用 cuBLAS，归约与 Attention 使用 FP32 中间值；
- DeepSeek-R1-Distill-Qwen-1.5B 单卡 BF16 greedy 推理与 KV Cache 可用，token 与 Hugging Face 完全一致；
- 受限于 6 GiB 显存，完整推理建议限制 KV Cache 容量。

### 4.3 MetaX C500

- 独立设备类型 `LLAISYS_DEVICE_MACA`，Python 侧 `DeviceType.MACA`，与 NVIDIA 后端完全隔离；
- `src/device/maca/` 用 MACA 原生 `mc*` API 实现 Runtime（设备/Stream 管理、Device/Pinned Host 内存、同步/异步拷贝）；
- 八个算子提供独立 `llaisys::ops::maca` 实现，支持 F32、F16、BF16，Linear 使用 `mcblasGemmEx`（`MCBLAS_COMPUTE_32F_PEDANTIC` 严格 FP32）；
- `xmake/maca.lua` + `--mx-gpu` 开关复用 xmake 的 CUDA 编译/device-link 流程，编译器驱动为 mxcc；`scripts/maca_env.sh` 生成的包装脚本仅做参数适配（`-m64`/`-rdc=true` 过滤、`-dlink` 翻译），不涉及源码翻译；
- DeepSeek-R1-Distill-Qwen-1.5B 单卡 BF16 greedy 推理与 KV Cache 可用，token 与 PyTorch 完全一致；
- 参考侧关闭 TF32（`allow_tf32 = False`），对比基准公平且更严格。
