"""Metal platform build configuration.

IMPORTANT: GGML_NATIVE is disabled to ensure CPU fallback code is portable.
For Apple Silicon, this uses ARM64 architecture flags which are already optimized.
The GGML_ACCELERATE flag enables Apple's Accelerate framework for optimized CPU operations.
"""

import platform


def check_environment():
    """Check if Metal is available (macOS only)."""
    if platform.system() == "Darwin":
        print("Metal environment detected (macOS).")
        return True
    print("Metal is only available on macOS.")
    return False


def get_compile_flags(target_any):
    """Get compile flags for Metal."""
    # Disable GGML_NATIVE; target architecture is set centrally during CMake configure.
    # GGML_ACCELERATE enables Apple's optimized framework
    return ["-DSD_METAL=ON", "-DGGML_NATIVE=OFF", "-DGGML_ACCELERATE=ON"]


def get_platform_name():
    """Get platform name for Metal."""
    return "metal"
