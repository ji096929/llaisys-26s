-- .cu 直接编译进最终 DLL：让 nvcc 处理 device link（__cudaRegisterLinkedBinary 等符号）
target("llaisys")
    set_toolchains("cuda")
    add_files("../src/device/nvidia/*.cu")
    add_files("../src/ops/*/nvidia/*.cu")

    add_cuflags("-arch=sm_89")
    if is_plat("windows") then
        add_cuflags("-Xcompiler /MD")
    end

    add_links("cudart", "cublas")
    if is_plat("windows") then
        add_linkdirs("$(env CUDA_PATH)/lib/x64")
    end
target_end()
