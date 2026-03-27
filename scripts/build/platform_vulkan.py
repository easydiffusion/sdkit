"""Vulkan platform build configuration.

IMPORTANT: GGML_NATIVE is disabled to ensure CPU fallback code is portable across
different processors (Intel and AMD). This prevents -march=native optimization which
would cause crashes when the binary runs on a different CPU than it was built on.

This is especially important for Vulkan builds because:
1. CPU code is still used for model loading and some operations
2. The offload_to_cpu feature relies on CPU-optimized kernels
3. Binaries need to work on both Intel and AMD systems

The AVX2/FMA/F16C configuration works on:
- Intel: Haswell and newer (2013+)
- AMD: Excavator/Zen and newer (2015+)
"""

import os
import re
import subprocess


def check_environment():
    """Check if Vulkan SDK is set up."""
    try:
        result = subprocess.run(["vulkaninfo", "--version"], capture_output=True, text=True)
        if not result.stderr:
            print("Vulkan environment detected.")
            return True
    except:
        pass
    print("Vulkan SDK not found. Please install Vulkan SDK.")
    return False


def get_compile_flags(target_any):
    """Get compile flags for Vulkan."""
    # Disable GGML_NATIVE for portable CPU code (works on both Intel and AMD)
    # Enable AVX2/FMA/F16C for good performance while maintaining compatibility
    flags = ["-DSD_VULKAN=ON", "-DGGML_NATIVE=OFF"]

    if "-x64-" in target_any:
        flags.extend(
            [
                "-DGGML_AVX2=ON",
                "-DGGML_FMA=ON",
                "-DGGML_F16C=ON",
                "-DGGML_BMI2=ON",
            ]
        )

    if target_any.startswith("win-arm64"):
        vulkan_sdk_path = os.environ["VULKAN_SDK"].replace("\\", "/")

        flags.append("-G Ninja")
        flags.append("-DCMAKE_TOOLCHAIN_FILE=cmake/arm64-windows-llvm.cmake")
        flags.append(f"-DVulkan_LIBRARY={vulkan_sdk_path}/Lib-ARM64/vulkan-1.lib")

    return flags


def get_platform_name():
    """Get platform name for Vulkan."""
    return "vulkan"


def get_env(target_any):
    """Get environment variables for Vulkan build."""
    if target_any.startswith("win-arm64"):
        env = os.environ.copy()

        # Prevent ggml-vulkan's cmake from picking up MSVC tools
        env["PATH"] = re.sub(r"Tools\\MSVC\\[\d\.]+\\bin\\Hostx64\\x64", r"Tools\\Llvm\\x64\\bin", env["PATH"])

        return env

    return None
