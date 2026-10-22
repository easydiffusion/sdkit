# Building and platforms

This page explains how to build sdkit from source, which platform/backend
combinations are supported, and how the build automation in `scripts/` works.
Day-to-day development workflow is in [development](development.md).

## Prerequisites

- CMake 3.16 or newer, a C++20 compiler, and Ninja or another generator.
- A **git submodule** must be initialized right after cloning the repo - this is
  easy to forget and the build will fail without it. The submodule is
  `stable-diffusion.cpp` (a fork of leejet's stable-diffusion.cpp, declared in
  `.gitmodules`), and it contains the ggml library inside it:

  ```
  git clone https://github.com/easydiffusion/sdkit.git
  git checkout v3
  git submodule init
  git submodule update --recursive
  ```

- Python 3 with `requests` is needed for the build/release scripts and the
  `generate_image.py` smoke test.

Toolchain setup for the CI runners (Visual Studio Build Tools with MSVC and Clang on
Windows, CUDA toolkit and the Vulkan SDK on Linux, or a Docker-based Linux
environment) is described in the
[release process](release-process.md#runner-setup).

## CMake layout

The top-level `CMakeLists.txt`:

- Sets C++20 and builds shared libraries (`BUILD_SHARED_LIBS`,
  `SD_BUILD_SHARED_LIBS`, `SD_BUILD_SHARED_GGML_LIB`), so the `sdkit` executable
  loads `stable-diffusion` and `ggml` shared libraries at runtime. A post-build step
  copies those DLLs next to the executable; RPATH settings point Linux/macOS builds
  at the binary's own directory.
- Adds the stable-diffusion.cpp submodule as a subdirectory, which brings in the
  `stable-diffusion` and `ggml` targets.
- Globs all of `src/*.cpp` into the `sdkit` executable target, with include paths for
  `include/`, `third_party/`, and the submodule.
- Defines an optional `SDKIT_BUILD_TOOLS` flag that builds the `modelinfo` CLI from
  `tools/` (see [models-and-detection](models-and-detection.md)).
- On Windows it suppresses Unicode warnings from crow headers (`/wd4566`), enables
  `/bigobj` for Debug builds, targets Windows 7+ (`_WIN32_WINNT=0x0601`), and links
  `ws2_32`.

## Backends

The same source builds against four backends, selected by flags passed to CMake:

| Backend | Flag | Notes |
|---|---|---|
| CPU | (default) | Flags tuned for portability: `GGML_NATIVE=OFF` plus AVX2/FMA/F16C/BMI2, which covers Intel Haswell (2013) and later. Turning `GGML_NATIVE=ON` produces faster but machine-specific binaries. |
| CUDA | `SD_CUDA=ON` | CPU-side flags are minimized (AVX only) since the GPU does the work. Builds per-SM variants: sm60, sm75, sm80, sm86, sm89, sm90, sm100, sm120 (via `CMAKE_CUDA_ARCHITECTURES` in the variant list), and records the toolkit version in the build manifest. |
| Vulkan | `SD_VULKAN=ON` | CPU flags minimized like CUDA. On Windows ARM64 it uses the clang toolchain (`cmake/arm64-windows-llvm.cmake`) and the ARM64 Vulkan library from the SDK, with a PATH rewrite so ggml's CMake doesn't pick up MSVC tools. |
| Metal | `SD_METAL=ON` | macOS only; `GGML_ACCELERATE=ON` for Apple Accelerate, with a configurable minimum deployment target (10.14/10.15 for x64, 11.0 for arm64). |

## The build matrix

`scripts/build_all.py` decides what gets built on the current OS:

| OS | Platform/arch combinations |
|---|---|
| Windows | cpu x64, cuda x64, vulkan x64, vulkan arm64 |
| Linux | cpu x64, cuda x64, vulkan x64 |
| macOS | metal arm64 (min 11.0), metal x64 (min 10.15) |

## How scripts/build/ works

Each backend has a `scripts/build/platform_<name>.py` module exposing a common
interface: `check_environment()` (verifies nvcc, vulkaninfo, or the OS),
`get_compile_flags()`, `get_platform_name()`, and optionally `get_variants()`,
`get_manifest_data()`, `get_additional_files()`, and `get_env()`.

`scripts/build/__main__.py` is the dispatcher:

```
python -m scripts.build --platform cuda --arch x64 [--variant sm86] [--macos-min-version 11.0]
```

It looks up the matching platform module and calls `common.build_project()`, which
for each variant:

1. Runs `check_environment()` and aborts if the toolchain is missing.
2. Computes a target name like `linux-x64-cuda-sm86` (or `...-any` when the backend
   has no variants).
3. Configures CMake with `-DCMAKE_BUILD_TYPE=Release` plus the platform flags (and
   `CMAKE_OSX_ARCHITECTURES`/deployment target on macOS).
4. Builds, then collects release files from `build/<target>/bin` and `lib/`,
   skipping versioned intermediates (`foo.so.1`, `foo.1.dylib`) and the
   `vulkan-shaders-gen` helper.
5. Compresses each artifact to `.tar.gz` (maximum compression, skipping recompression
   when the file hash is unchanged from a previous run), writes a
   `<target>-manifest.json` with SHA-256 hashes and archive names, and leaves
   everything in `build/<target>/release_artifacts/`.

These manifests are what the release pipeline consumes; see
[release-process](release-process.md).

## Vendored dependencies

- **stable-diffusion.cpp** - a git submodule pointing at a fork of
  leejet/stable-diffusion.cpp. Updates are pulled in by changing the submodule
  pointer and committing it (`git submodule update --remote` inside the checkout, or
  checking out the desired commit, then committing the new submodule hash). Because
  ggml lives inside the submodule, engine and ggml updates travel together.
- **crow / asio** - the HTTP framework and its networking dependency are vendored as
  plain headers in `third_party/` (`crow.h`, `crow/`, `asio.hpp`, `asio/`), updated
  by copying in new versions.
- **base64.hpp, uuid.h** - small vendored utility headers in `third_party/`.

Everything else (stb image libraries used by sdkit and sd.cpp) comes from inside the
stable-diffusion.cpp submodule.

## macOS CI

`.github/workflows/ci_metal.yaml` builds the macOS Metal artifacts on a hosted
macOS runner (`workflow_dispatch` only): it checks out the code, runs
`git submodule update --init --recursive`, installs CMake and Python, and runs
`python -m scripts.ci` (which builds every macOS platform and uploads the
artifacts; see [release-process](release-process.md)).
