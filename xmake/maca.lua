-- 沐曦 MetaX (MACA) 后端
--
-- 复用 xmake 的 CUDA 编译/device-link 流程，编译器驱动换成 mxcc 的 MACA 模式：
--   1. 先执行 `source scripts/maca_env.sh`，生成 .maca/bin/{nvcc,mxcc} 驱动；
--   2. xmake 的 CUDA 规则会自动追加 -lcudart/-lcublas/-lcudadevrt 等链接参数，
--      .maca/lib64 下的垫片让链接通过，实际符号全部来自 MACA 原生库。
--
-- 所有源码（src/device/maca、src/ops/*/maca）均使用原生 mc* API 与 mxcc，
-- 不经过 cu-bridge/cucc 翻译层。
toolchain("maca")
    set_kind("compiler")
    set_toolset("cc", "gcc")
    set_toolset("cxx", "g++")
    set_toolset("cu", "gcc@$(env CUDA_PATH)/bin/mxcc")
    set_toolset("culd", "gcc@$(env CUDA_PATH)/bin/mxcc")
    set_toolset("ld", "g++")
    set_toolset("sh", "g++")
toolchain_end()

target("llaisys")
    set_toolchains("maca")
    set_policy("check.auto_ignore_flags", false)

    add_files("../src/device/maca/*.cu")
    add_files("../src/ops/*/maca/*.cu")

    add_cuflags("-fPIC", "-std=c++17", "-offload-arch native",
                "--maca-path=$(env MACA_PATH)", {force = true})
    add_culdflags("-std=c++17", "-offload-arch native",
                  "--maca-path=$(env MACA_PATH)", {force = true})

    add_includedirs("$(env MACA_PATH)/include/mcblas", "$(env MACA_PATH)/include/common")
    add_links("mcruntime", "mcblas")
    add_linkdirs("$(env MACA_PATH)/lib", "$(env CUDA_PATH)/lib64")
target_end()
