import os
import subprocess
import hashlib
import sys
import json
import platform
import tarfile
import re

INTERMEDIATE_LIB_PATTERN = re.compile(r".*\.so\.\d+|.*\.\d+\.dylib$")  # skip foo.so.1, foo.1.dylib etc
MACOS_CMAKE_ARCH = {"x64": "x86_64", "arm64": "arm64"}
DEFAULT_MACOS_DEPLOYMENT_TARGET = {"x64": "10.14", "arm64": "11.0"}


def get_os():
    """Get OS name for target."""
    system = platform.system()
    if system == "Windows":
        return "win"
    elif system == "Darwin":
        return "mac"
    elif system == "Linux":
        return "linux"
    else:
        return system.lower()


def get_arch():
    """Get architecture for target."""
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "x64"
    elif machine in ("arm64", "aarch64"):
        return "arm64"
    else:
        return machine


def get_target_arch(arch=None):
    """Get the requested target architecture or fall back to the host arch."""
    return arch if arch else get_arch()


def get_cmake_configure_args(arch=None, macos_min_version=None):
    """Get target-specific CMake configure arguments."""
    os_name = get_os()
    arch_name = get_target_arch(arch)
    cmake_args = []

    if os_name == "mac":
        cmake_args.append(f"-DCMAKE_OSX_ARCHITECTURES={MACOS_CMAKE_ARCH[arch_name]}")
        deployment_target = macos_min_version or DEFAULT_MACOS_DEPLOYMENT_TARGET[arch_name]
        cmake_args.append(f"-DCMAKE_OSX_DEPLOYMENT_TARGET={deployment_target}")

    return cmake_args


def configure_cmake(build_dir, source_dir, options=None, env=None):
    """Configure CMake with given options."""
    options = options or []
    options = options + ["-DCMAKE_BUILD_TYPE=Release"]
    os.makedirs(build_dir, exist_ok=True)
    cmake_cmd = ["cmake"] + ["-S", source_dir, "-B", build_dir] + options
    print(f"Configuring CMake: {' '.join(cmake_cmd)}")

    result = subprocess.run(cmake_cmd, env=env)
    if result.returncode != 0:
        print("CMake configure failed")
        sys.exit(1)


def build_cmake(build_dir, env=None):
    """Build the project using CMake."""
    cmake_cmd = ["cmake", "--build", build_dir, "--config", "Release"]
    print(f"Building: {' '.join(cmake_cmd)}")
    result = subprocess.run(cmake_cmd, env=env)
    if result.returncode != 0:
        print("Build failed")
        sys.exit(1)


def get_release_files(build_dir):
    """Get the list of all distributable release files (executables and shared libraries)."""
    bin_dir = os.path.join(build_dir, "bin")
    lib_dir = os.path.join(build_dir, "lib")
    files = []
    # Collect all files in bin (executables and dlls)
    if os.path.exists(bin_dir):
        for root, dirs, filenames in os.walk(bin_dir):
            for filename in filenames:
                if INTERMEDIATE_LIB_PATTERN.match(filename) or "vulkan-shaders-gen" in filename:
                    continue
                files.append(os.path.join(root, filename))
    # Collect shared libraries in lib (.so, .dylib, .dll)
    if os.path.exists(lib_dir):
        for root, dirs, filenames in os.walk(lib_dir):
            for filename in filenames:
                if filename.endswith((".so", ".dylib", ".dll")):
                    files.append(os.path.join(root, filename))
    return files


def compute_sha256(file_path):
    """Compute SHA256 hash of a file."""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def prepare_build(build_subdir, check_func):
    """Prepare build directories and check prerequisites."""
    project_root = os.getcwd()
    build_root = os.path.join(project_root, "build")
    build_dir = os.path.join(build_root, build_subdir)

    os.makedirs(build_root, exist_ok=True)

    cmake_file = os.path.join(project_root, "CMakeLists.txt")
    if not os.path.exists(cmake_file):
        print("CMakeLists.txt not found in the current directory.")
        sys.exit(1)

    if not check_func():
        sys.exit(1)

    return project_root, build_dir


def get_target(get_platform_name_func, variant_name, arch=None):
    """Get the target for the build.

    Args:
        get_platform_name_func: Function that returns the platform name
        variant: Optional variant name (e.g., 'sm86'). If None, uses 'any'
        arch: Optional architecture name. If None, uses detected arch
    """
    platform_name = get_platform_name_func()
    os_name = get_os()
    arch_name = get_target_arch(arch)
    target = f"{os_name}-{arch_name}-{platform_name}-{variant_name}"
    return target


def build_project_cmake(build_dir, project_root, options, env):
    """Configure and build the project using CMake."""
    configure_cmake(build_dir, project_root, options, env)
    build_cmake(build_dir, env)


def collect_and_move_artifacts(build_dir, target, additional_files, target_any):
    """Collect release files, compress each to .tar.gz with maximum compression, move to release_artifacts with prefixed names, and create manifest."""
    release_files = get_release_files(build_dir)

    release_artifacts_dir = os.path.join(build_dir, "release_artifacts")
    os.makedirs(release_artifacts_dir, exist_ok=True)

    # read the existing manifest, if present
    existing_manifest = {}
    manifest_path = os.path.join(release_artifacts_dir, f"{target}-manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            existing_manifest = json.load(f)

    print("Collecting and compressing release artifacts...")

    manifest = {"files": {}}

    def add_to_manifest(files, target):
        for file_path in files:
            basename = os.path.basename(file_path)
            new_name = f"{target}-{basename}"
            tar_gz_name = f"{new_name}.tar.gz"
            tar_gz_path = os.path.join(release_artifacts_dir, tar_gz_name)

            sha256 = compute_sha256(file_path)

            # don't compress if the file exists and the hash matches
            curr_sha256 = existing_manifest.get("files", {}).get(basename, {}).get("sha256")
            if os.path.exists(tar_gz_path) and sha256 == curr_sha256:
                print(f"Archive {tar_gz_name} already exists and is up to date. Skipping compression.")
            else:
                if os.path.exists(tar_gz_path):
                    os.remove(tar_gz_path)

                compress(file_path, tar_gz_path)

            uri = tar_gz_name
            manifest["files"][basename] = {"sha256": sha256, "uri": uri}

    add_to_manifest(release_files, target)
    add_to_manifest(additional_files, target_any)

    return release_artifacts_dir, manifest


def write_manifest(release_artifacts_dir, target, manifest):
    """Write the manifest to the release_artifacts directory and print it."""
    manifest_path = os.path.join(release_artifacts_dir, f"{target}-manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)

    print(f"Target: {target}")
    print("Build manifest:")
    print(json.dumps(manifest, indent=4))


def build_project(
    check_func,
    get_compile_flags_func,
    get_platform_name_func,
    get_variants_func=None,
    get_manifest_data_func=None,
    get_additional_files_func=None,
    get_env_func=None,
    arch=None,
    variant=None,
    macos_min_version=None,
):
    """Common build logic for all backends.

    Args:
        check_func: Function to check if environment is ready
        get_compile_flags_func: Function that returns base compile flags
        get_platform_name_func: Function that returns platform name
        get_variants_func: Function that returns list of variants
        get_manifest_data_func: Optional function that returns additional manifest data
        get_additional_files_func: Optional function that returns additional files
        get_env_func: Optional function that returns environment dict for cmake
        arch: Optional architecture to build for
        macos_min_version: Optional macOS deployment target override
    """
    # Check if platform module has get_variants function
    variants = []
    if get_variants_func:
        variants = get_variants_func()

    # If no variants, use a single build with "any" variant
    if not variants:
        variants = [{"name": "any", "compile_flags": []}]

    if variant is not None:
        # Filter variants to only include the specified one
        variants = [v for v in variants if v["name"] == variant]
        if not variants:
            print(f"Variant '{variant}' not found.")
            sys.exit(1)

    # Get target for "any" variant to use for additional files
    target_any = get_target(get_platform_name_func, "any", arch)
    cmake_args = get_cmake_configure_args(arch, macos_min_version)

    # Get environment for cmake
    env = get_env_func(target_any) if get_env_func else None

    # Build for each variant
    for variant in variants:
        variant_name = variant["name"]
        variant_compile_flags = variant.get("compile_flags", [])

        target = get_target(get_platform_name_func, variant_name, arch)
        project_root, build_dir = prepare_build(target, check_func)

        # Combine base compile flags with variant-specific flags
        base_options = get_compile_flags_func(target_any)
        options = cmake_args + base_options + variant_compile_flags

        # Resolve additional files if available
        additional_files = get_additional_files_func() if get_additional_files_func else []
        print(f"Additional files to include: {additional_files}")

        # Build the project
        build_project_cmake(build_dir, project_root, options, env)

        # Collect and move artifacts
        release_artifacts_dir, manifest = collect_and_move_artifacts(build_dir, target, additional_files, target_any)

        # Merge additional manifest data if available
        if get_manifest_data_func:
            additional_data = get_manifest_data_func()
            manifest.update(additional_data)

        write_manifest(release_artifacts_dir, target, manifest)

        print("Release artifacts are located in:", release_artifacts_dir)


def compress(input_file, output_file):
    """Compress a file to .tar.gz with maximum compression."""
    with tarfile.open(output_file, "w:gz", compresslevel=9, dereference=True) as tar:
        tar.add(input_file, arcname=os.path.basename(input_file))
