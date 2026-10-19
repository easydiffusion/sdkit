# Building and Platforms

This page covers how to build sdkit from source: the CMake setup, the per-backend
build configurations, which platform/arch combinations are produced on which OS, how
the `scripts/build/` tooling drives the builds, and how `third_party/` dependencies
are vendored. The full, copy-pasteable runner setup steps live in `CI.md` at the
repository root - this page links to it rather than duplicating the commands.

## Prerequisites

- A C++20 toolchain (MSVC or Clang on Windows, GCC/Clang on Linux, AppleClang/Clang
  on macOS).
- CMake 3.16 or newer, plus a build generator (Ninja is recommended).
- Python 3 (for the build scripts in `scripts/build/`).
- The git submodules - **these must be initialized after a fresh checkout**, or the
  build fails with missing headers and sources:

  ```bash
  git submodule init
  git submodule update --recursive
  ```

- Per-backend extras, checked automatically by the build scripts (see
  [Backend configurations](#backend-configurations)):
  - **CUDA**: the CUDA toolkit on `PATH` (`nvcc --version` must succeed).
  - **Vulkan**: the Vulkan SDK (`vulkaninfo --version` must succeed).
  - **Metal**: macOS only (no extra tooling beyond Xcode's toolchain).
  - **CPU**: nothing extra.

## CMake layout

The top-level `CMakeLists.txt` (see [architecture](architecture.md) for the target
table) builds `ggml` (shared), `sam3` (static), `stable-diffusion` (shared), and the
`sdkit` executable, in that dependency order. A post-build step copies the shared
libraries next to the executable, and RPATH is set to `$ORIGIN` (Linux) /
`@executable_path` (macOS) so the runtime artifacts stay together and can be
relocated. `SDKIT_BUILD_TOOLS=ON` additionally builds the `modelinfo` CLI tool
(see [models-and-detection](models-and-detection.md)).

The ggml backend selection is driven entirely by `GGML_*` CMake variables passed at
configure time; there is no separate branch per backend. The same variables flow into
both stable-diffusion.cpp and sam3.cpp, because both link the shared `ggml` target.

A minimal local CPU build is:

```bash
mkdir build && cd build
cmake -S .. -B . -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF
cmake --build . --config Release
```

`GGML_NATIVE=OFF` is used by the build scripts by default (see below) so the binary
doesn't crash on CPUs with a different instruction set than the build machine. For a
personal build you may set `GGML_NATIVE=ON` for a faster binary that only runs on
your machine.

## Backend configurations

Each backend is described by a Python module in `scripts/build/`:

| Module | Backend | CMake flags it sets |
|---|---|---|
| `platform_cpu.py` | CPU | `-DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON` |
| `platform_cuda.py` | CUDA | `-DGGML_CUDA=ON -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_F16C=ON` (AVX2/FMA/BMI2 off for compatibility, since the heavy work runs on the GPU) |
| `platform_metal.py` | Metal | `-DGGML_METAL=ON -DGGML_NATIVE=OFF -DGGML_ACCELERATE=ON` |
| `platform_vulkan.py` | Vulkan | `-DGGML_VULKAN=ON -DGGML_NATIVE=OFF` plus x64-specific AVX flags |

The important convention across all of them is that `GGML_NATIVE` stays off so the
CPU-side fallback code is portable across machines (see the explanatory comment at
the top of `platform_cpu.py` for the full reasoning, including how to get maximum
performance or maximum compatibility).

### Platform and arch matrix

`scripts/build_all.py` defines the combinations that are built for each OS:

| OS | (backend, arch) pairs |
|---|---|
| Windows | `cpu/x64`, `cuda/x64`, `vulkan/x64`, `vulkan/arm64` |
| Linux | `cpu/x64`, `cuda/x64`, `vulkan/x64` |
| macOS | `metal/arm64` (min macOS 11.0), `metal/x64` (min macOS 10.15) |

So the backends per OS are:

- **CPU** - available on Windows, Linux, and macOS (the macOS `metal` entries are the
  primary builds there; a CPU build is still possible locally).
- **CUDA** - Windows x64 and Linux x64. NVIDIA GPUs only.
- **Vulkan** - Windows x64 and arm64, Linux x64. Covers NVIDIA, AMD, and Intel GPUs
  (subject to driver support).
- **Metal** - macOS arm64 (Apple Silicon) and macOS x64 (Intel Macs).

`build_all.py` runs `python -m scripts.build --platform <name> --arch <arch>` (plus a
`--macos-min-version` for the macOS entries) once per pair, sequentially, and aborts
at the first failure.

### How `scripts.build` works

`scripts/build/__main__.py` is the single entry point:

```bash
python -m scripts.build --platform cpu   # or cuda / vulkan / metal
# optional:
#   --arch x64|arm64
#   --variant sm86          (CUDA compute capability variants, see below)
#   --macos-min-version 11.0
```

It resolves the module named `platform_<platform>`, and calls into
`scripts/build/common.py` with that module's hooks. The sequence in
`common.build_project` is:

1. `check_environment()` - verifies the backend's toolchain is available (see the
   prerequisites above), and aborts the build otherwise.
2. `get_compile_flags(target)` - collects the backend's base CMake flags.
3. Optionally `get_variants()` - a list of extra builds to run for this backend,
   each with its own additional flags (only CUDA has variants; see below).
4. For each variant: `prepare_build()` creates `build/<target>/` and runs
   `configure_cmake()` followed by `build_cmake()` (i.e. `cmake -S . -B build/<target>`
   then `cmake --build build/<target> --config Release`).
5. `get_additional_files_func` (CUDA only) locates the runtime CUDA libraries and
   includes them with the release artifacts.
6. `collect_and_move_artifacts` walks `build/<target>/bin` and
   `build/<target>/lib`, compresses each distributable file to
   `<target>-<basename>.tar.gz` under `build/<target>/release_artifacts/` (skipping
   intermediate-versioned `.so.N` / `.N.dylib` files and the `vulkan-shaders-gen`
   helper), and records a `<target>-manifest.json` with the SHA256 of each file.
   Archives whose hash matches the existing manifest are not recompressed, so
   re-running a build is cheap for unchanged artifacts.
7. `get_manifest_data_func` (CUDA only) adds a `cuda_version` key derived from
   `nvcc --version` output, merged into the manifest.

The manifest and the `.tar.gz` artifacts are the input to the upload step
(see [release-process](release-process.md)).

### CUDA variants

`platform_cuda.py` returns one build per compute capability:

`sm60`, `sm75`, `sm80`, `sm86`, `sm89`, `sm90`, `sm100`, `sm120`

Each variant adds `-DCMAKE_CUDA_ARCHITECTURES=<n>` on top of the base CUDA flags, so
a single `build_all` run produces one set of binaries per GPU generation. These
variant names appear in the target string (e.g. `linux-x64-cuda-sm86`) and in the
artifact filenames.

### Windows arm64 (Vulkan) special case

For `win-arm64-vulkan-any` the module adds a Clang-based cross-compilation setup:

- `-G Ninja` with the toolchain file `cmake/arm64-windows-llvm.cmake`.
- `-DVulkan_LIBRARY=<VULKAN_SDK>/Lib-ARM64/vulkan-1.lib` (the ARM64 Vulkan library
  from the SDK).
- The CMake `PATH` is rewritten so the MSVC `Hostx64\x64` tools are not picked up by
  ggml-vulkan's cmake, which would otherwise break the cross-compile.

## Vendored third-party code

`third_party/` is checked out as git submodules (run `git submodule init` and
`git submodule update --recursive` after cloning, as noted above):

| Submodule | Upstream | Role |
|---|---|---|
| `third_party/stable-diffusion.cpp` | [leejet/stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) | diffusion inference engine (see [image-generation](image-generation.md)) |
| `third_party/ggml` | [leejet/ggml](https://github.com/leejet/ggml) | tensor runtime and backend dispatch (shared by sd.cpp and sam3) |
| `third_party/sam3.cpp` | [cmdr2/sam3.cpp](https://github.com/cmdr2/sam3.cpp) | SAM3 segmentation engine (see [segmentation](segmentation.md)) |
| `third_party/stb` | [nothings/stb](https://github.com/nothings/stb) | image loading/encoding/resize |

The stable-diffusion.cpp and ggml entries are forks maintained under the same
user (`leejet`) that sdkit pins to, so updates to the upstream repo are pulled in by
updating the submodule pointer and committing it. To update one:

1. From inside the submodule, `git fetch` and check out the desired commit or tag.
2. From the sdkit repo, `git add third_party/<name>` and commit the new pointer.
3. Rebuild; if the upstream API changed, fix the call sites in `src/` (for example,
   recent commits have had to update VAE tiling parameter names to match a new
   sd.cpp API, or point to sd.cpp's own ggml fork when the upstream one diverged).

Because `stable-diffusion.cpp` and `sam3.cpp` both consume the shared `ggml` target
(see [architecture](architecture.md)), the two engines and ggml must stay in sync
ABI-wise - updating one of the three usually means updating all three together and
verifying that the `GGML_MAX_NAME=128` define is still applied consistently before
ggml is included anywhere.

## Runner setup

`CI.md` at the repository root holds the step-by-step setup for the machines the
official releases are built on:

- **Windows**: MSVC + Clang toolchains, Ninja, and the CUDA toolkit.
- **Linux**: CUDA toolkit 12.8, the Vulkan SDK, and `vulkan-tools` /
  `vulkan-validationlayers`.
- **Docker**: a Linux container recipe is also documented there for repeatable
  CUDA/Vulkan builds.

Follow `CI.md` for the exact commands; this page won't restate them, since those
steps change more often than the build logic does.
