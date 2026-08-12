#!/usr/bin/env bash
# 沐曦 MetaX (MACA) 构建环境
#
# 生成 xmake CUDA 流程所需的驱动与垫片，并导出环境变量。
#   - .maca/bin/mxcc：参数适配包装，调用 MACA 原生编译器（不经过 cu-bridge）
#   - .maca/bin/nvcc：让 xmake 的 find_cuda 找到 SDK 目录
#   - .maca/lib64：xmake CUDA 规则自动追加的链接库垫片
#
# 用法：
#   source scripts/maca_env.sh
#   xmake f --mx-gpu=y -cv
#   xmake
#   xmake install
set -euo pipefail

MACA_PATH="${MACA_PATH:-/opt/maca}"
MXCC="$MACA_PATH/mxgpu_llvm/bin/mxcc"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOL_DIR="$ROOT/.maca"

if [ ! -x "$MXCC" ]; then
    echo "[maca] error: mxcc not found under $MACA_PATH" >&2
    return 1 2>/dev/null || exit 1
fi

mkdir -p "$TOOL_DIR/bin" "$TOOL_DIR/include" "$TOOL_DIR/lib64"

# mxcc 驱动：把 xmake CUDA 规则的通用参数翻译成 mxcc 能理解的 MACA 参数
cat > "$TOOL_DIR/bin/mxcc" <<EOF
#!/bin/bash
args=()
compile=0
for a in "\$@"; do
    if [ "\$a" = "-c" ]; then
        compile=1
    fi
    case "\$a" in
        -m64|-rdc=true) ;; # xmake CUDA 规则通用参数，MACA 不需要
        -dlink) args+=("-dlink-obj" "-fgpu-rdc" "--maca-link") ;;
        *) args+=("\$a") ;;
    esac
done
if [ "\$compile" = "1" ]; then
    exec "$MXCC" -x maca "\${args[@]}"
else
    exec "$MXCC" "\${args[@]}"
fi
EOF
chmod +x "$TOOL_DIR/bin/mxcc"

# nvcc 驱动：xmake 的 find_cuda 通过它定位 SDK 目录
cat > "$TOOL_DIR/bin/nvcc" <<EOF
#!/bin/bash
exec "$TOOL_DIR/bin/mxcc" "\$@"
EOF
chmod +x "$TOOL_DIR/bin/nvcc"

# 占位头文件：find_cuda 用它确认 SDK；沐曦源码实际使用原生 MACA 头文件
if [ ! -f "$TOOL_DIR/include/cuda_runtime.h" ]; then
    printf '// MetaX backend uses native MACA headers. This file only satisfies xmake CUDA SDK detection.\n' \
        > "$TOOL_DIR/include/cuda_runtime.h"
fi

# xmake CUDA 规则自动追加的链接参数垫片（实际符号来自 MACA 原生库）
ln -sfn "$MACA_PATH/lib/libmcruntime.so" "$TOOL_DIR/lib64/libcudart.so"
ln -sfn "$MACA_PATH/lib/libmcruntime.so" "$TOOL_DIR/lib64/libcudart_static.so"
ln -sfn "$MACA_PATH/lib/libmcblas.so" "$TOOL_DIR/lib64/libcublas.so"
if [ ! -f "$TOOL_DIR/lib64/libcudadevrt.so" ]; then
    printf 'int __llaisys_cudadevrt_stub;\n' \
        | gcc -shared -fPIC -x c - -o "$TOOL_DIR/lib64/libcudadevrt.so"
fi

export MACA_PATH
export CUDA_PATH="$TOOL_DIR"
export PATH="$TOOL_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$MACA_PATH/lib:${LD_LIBRARY_PATH:-}"
if [ "$(id -u)" = "0" ]; then
    export XMAKE_ROOT=y
fi

echo "[maca] MetaX toolchain ready: $("$MXCC" --version | head -1)"
