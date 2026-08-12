# 作业 4 报告：LLAISYS 多平台 CUDA 适配

## 支持的平台

| 平台 | 后端 | 源码位置 | 验证状态 |
| --- | --- | --- | --- |
| NVIDIA | 原生 CUDA（nvcc + cuBLAS） | `src/device/nvidia/`、`src/ops/*/nvidia/` | 代码随仓库维护，需在 NVIDIA 机器/CI 上运行 |
| 沐曦 MetaX（MACA） | 原生 MACA（mxcc + mcBLAS） | `src/device/maca/`、`src/ops/*/maca/` | 本机 MetaX C500 上完整验证（见下） |

## 沐曦后端设计

LLAISYS 的设备抽象为每个设备类型维护独立的 Runtime API。沐曦后端新增：

- 设备类型 `LLAISYS_DEVICE_MACA`（`include/llaisys.h`），Python 侧 `DeviceType.MACA`；
- `src/device/maca/maca_runtime_api.cu`：用 MACA 原生 `mc*` API 实现
  `mcGetDeviceCount` / `mcSetDevice` / `mcDeviceSynchronize` / `mcStreamCreate` /
  `mcMalloc` / `mcFree` / `mcMallocHost` / `mcMemcpy` 等；
- 8 个算子的 `src/ops/*/maca/` 实现：kernel 语法与 CUDA 同构（类 CUDA 平台的定义），
  数据类型使用 `__half` / `__maca_bfloat16`，线性算子使用 `mcblasGemmEx`
  （`MCBLAS_COMPUTE_32F_PEDANTIC` 保证严格 FP32 精度）；
- `xmake/maca.lua` + `--mx-gpu` 开关：复用 xmake 的 CUDA 编译/device-link 流程，
  编译器驱动为 mxcc（MACA 模式），不经过 cu-bridge/cucc 翻译层。

## 复现步骤

```bash
# 1. 准备沐曦构建环境（生成 .maca 驱动与垫片）
source scripts/maca_env.sh

# 2. 编译并安装共享库
xmake f --mx-gpu=y -cv
xmake
xmake install

# 3. 安装 Python 包
pip install ./python/

# 4. 运行测试（沐曦卡在 PyTorch 侧仍以 cuda 设备名出现）
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

## 本机验证结果

环境：MetaX C500（切片卡，Compute 25% / 16GB），MACA 3.5.3，mxcc 1.0.0。

| 测试 | 结果 |
| --- | --- |
| `test_runtime.py --device maca` | 通过（1 张卡，H2D/D2D/D2H 拷贝） |
| add / argmax / embedding / rms_norm / rope / self_attention / swiglu `--device maca` | 通过 |
| linear `--device maca` | 通过（mcblasGemmEx + PEDANTIC 精度） |
| `test_infer.py --test --device maca` | 通过（token 序列与 PyTorch 完全一致） |

模型推理实测（DeepSeek-R1-Distill-Qwen-1.5B，128 token 上限，贪心解码）：

| 实现 | 耗时 |
| --- | --- |
| PyTorch（MetaX C500） | 3.57s |
| LLAISYS（MACA 后端） | 10.35s |

两者输出 token 序列完全一致；LLAISYS 较慢是因为算子内核以正确性优先，
后续可参考性能优化作业进一步调优。

## 说明

- 沐曦后端是独立实现：Runtime API、算子、构建配置均为 MACA 原生，
  与 NVIDIA 后端的差异只在平台 API 与库（`mc*` vs `cuda*`、`mcBLAS` vs `cuBLAS`）。
- xmake 的 CUDA 规则要求 `nvcc` 名称的驱动，`scripts/maca_env.sh` 生成的包装脚本
  仅做参数适配（`-m64`/`-rdc=true` 过滤、`-dlink` 翻译），不涉及源码翻译。
- 由于沐曦 PyTorch 使用 `cuda` 设备别名，参考实现（torch）与 LLAISYS 测试统一通过
  `--device maca` 入口运行。
- `test/test_utils.py` 中关闭了 PyTorch 的 TF32（`allow_tf32 = False`）：
  沐曦 PyTorch 默认对 f32 矩阵乘使用 TF32，自身误差约 3e-5，而 LLAISYS 的
  `mcblasGemmEx(MCBLAS_COMPUTE_32F_PEDANTIC)` 是严格 FP32（误差约 4.5e-6）。
  关闭参考侧 TF32 后两者误差约 6e-7，测试公平且更严格。
