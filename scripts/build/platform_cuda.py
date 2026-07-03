"""CUDA platform build configuration."""

import os
import platform
import subprocess

LINUX_CUDA_LIBS = ["libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12"]
WIN_CUDA_LIBS = ["cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll"]


def check_environment():
    """Check if CUDA environment is set up."""
    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        if result.returncode == 0 and "nvcc" in result.stdout:
            print("CUDA environment detected.")
            return True
    except:
        pass
    print("CUDA environment not found. Please install CUDA toolkit.")
    return False


def get_compile_flags(target_any):
    """Get compile flags for CUDA."""
    # Disable GGML_NATIVE for portable CPU code (works on both Intel and AMD)
    # Disable AVX2/FMA/F16C to maximize compatibility, since the actual operations are performed by the GPU
    return [
        "-DSD_CUDA=ON",
        "-DGGML_NATIVE=OFF",
        "-DGGML_AVX=ON",
        "-DGGML_AVX2=OFF",
        "-DGGML_FMA=OFF",
        "-DGGML_F16C=ON",
        "-DGGML_BMI2=OFF",
    ]


def get_variants():
    """Get list of CUDA compute capability variants to build.

    Returns a list of dicts, each with 'name' and 'compile_flags' keys.
    """
    return [
        {"name": "sm60", "compile_flags": ["-DCMAKE_CUDA_ARCHITECTURES=60"]},
        {"name": "sm75", "compile_flags": ["-DCMAKE_CUDA_ARCHITECTURES=75"]},
        {"name": "sm80", "compile_flags": ["-DCMAKE_CUDA_ARCHITECTURES=80"]},
        {"name": "sm86", "compile_flags": ["-DCMAKE_CUDA_ARCHITECTURES=86"]},
        {"name": "sm89", "compile_flags": ["-DCMAKE_CUDA_ARCHITECTURES=89"]},
        {"name": "sm90", "compile_flags": ["-DCMAKE_CUDA_ARCHITECTURES=90"]},
        {"name": "sm100", "compile_flags": ["-DCMAKE_CUDA_ARCHITECTURES=100"]},
        {"name": "sm120", "compile_flags": ["-DCMAKE_CUDA_ARCHITECTURES=120"]},
    ]


def get_platform_name():
    """Get platform name for CUDA."""
    return "cuda"


def get_manifest_data():
    """Get additional manifest data for CUDA."""
    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            # Parse version from output like "nvcc: NVIDIA (R) Cuda compiler driver\nCopyright (c) 2005-2023 NVIDIA Corporation\nBuilt on Tue_Jun_13_19:16:58_PDT_2023\nCuda compilation tools, release 12.2, V12.2.91\nBuild cuda_12.2.r12.2/compiler.32965470_0"
            for line in result.stdout.split("\n"):
                if "release" in line:
                    version = line.split("release ")[1].split(",")[0]
                    return {"cuda_version": version}
    except:
        pass
    return {}


def get_additional_files():
    "Returns physical paths to the CUDA libraries."

    # Get the list of CUDA libraries based on OS
    os_name = platform.system()
    if os_name == "Linux":
        cuda_libs = LINUX_CUDA_LIBS
    elif os_name == "Windows":
        cuda_libs = WIN_CUDA_LIBS
    else:
        return []

    # Locate the libraries using PATH
    path_dirs = os.environ.get("PATH", "").split(os.pathsep)
    found_libs = []

    for lib in cuda_libs:
        found = False
        for dir_path in path_dirs:
            lib_path = os.path.join(dir_path, lib)
            if os.path.exists(lib_path):
                found = True
                found_libs.append(lib_path)
                break

        if not found:
            raise FileNotFoundError(f"Required CUDA library {lib} not found in PATH.")

    return found_libs
