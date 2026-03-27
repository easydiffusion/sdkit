import sys
import argparse
from . import common

from . import platform_cuda  # noqa: F401
from . import platform_vulkan  # noqa: F401
from . import platform_cpu  # noqa: F401
from . import platform_metal  # noqa: F401


def main():
    parser = argparse.ArgumentParser(description="Build the project for a specific platform.")
    parser.add_argument(
        "--platform", required=True, choices=["cuda", "vulkan", "cpu", "metal"], help="The platform to build for"
    )
    parser.add_argument(
        "--arch",
        choices=["x64", "arm64"],
        help="The architecture to build for (default: detected from current platform)",
    )
    parser.add_argument(
        "--variant",
        default=None,
        help="The variant to build for (e.g., 'sm86' for CUDA). If not specified, the default variant for the platform will be used.",
    )
    parser.add_argument(
        "--macos-min-version",
        default=None,
        help="Minimum supported macOS version to pass as CMAKE_OSX_DEPLOYMENT_TARGET.",
    )
    args = parser.parse_args()

    platform = args.platform
    arch = args.arch
    variant = args.variant
    macos_min_version = args.macos_min_version
    try:
        module = globals()[f"platform_{platform}"]
    except KeyError:
        print(f"Module platform_{platform} not found.")
        sys.exit(1)

    check_func = module.check_environment
    compile_flags_func = module.get_compile_flags
    platform_name_func = module.get_platform_name
    variants_func = getattr(module, "get_variants", None)
    manifest_data_func = getattr(module, "get_manifest_data", None)
    additional_files_func = getattr(module, "get_additional_files", None)
    env_func = getattr(module, "get_env", None)

    common.build_project(
        check_func,
        compile_flags_func,
        platform_name_func,
        variants_func,
        manifest_data_func,
        additional_files_func,
        env_func,
        arch,
        variant,
        macos_min_version,
    )


if __name__ == "__main__":
    main()
