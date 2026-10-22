# Overview

sdkit ("stable diffusion kit") is a C++ program that exposes Stable Diffusion image
generation over a local HTTP API. It is the inference backend used by easy-diffusion:
the front end launches sdkit as a child process, points it at model directories, and
talks to it over HTTP on localhost. In v3 the entire implementation was rewritten
from Python to C++, and it is built on top of
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp), which is
vendored as a git submodule and provides the actual diffusion engine (and the ggml
tensor library underneath it).

This is a work in progress. The `v3` branch is actively developed and the API
surface is still settling. Treat the behavior described in these pages as matching
the current state of the `v3` branch, not as a stable contract.

## What it does

A single `sdkit` executable:

- Loads a Stable Diffusion checkpoint (plus optional VAE, text encoders, LoRAs,
  embeddings, and a ControlNet model) into an sd.cpp context.
- Serves an HTTP API (crow, bound to `127.0.0.1`, default port 8188) with routes for
  txt2img, img2img, progress polling, interrupt, upscaling, ControlNet preprocessing,
  options get/set, and checkpoint refresh. See [server-and-api](server-and-api.md).
- Persists user-facing settings (selected checkpoint, live previews, etc.) in an
  `options.json` file next to the executable.
- Streams progress and live previews back to callers via a task state registry.
- Offers image post-processing: RealESRGAN upscaling and Canny preprocessing for
  ControlNet. See [image-filters-and-upscaling](image-filters-and-upscaling.md).
- Provides a `modelinfo` tool for inspecting GGUF/safetensors model files.

## Main components

Each component lives as a header in `include/` and an implementation in `src/`:

| Component | Header | Responsibility |
|---|---|---|
| `Server` | `server.h` | Owns the crow HTTP app, defines all routes, translates JSON into component calls |
| `ImageGenerator` | `image_generator.h` | Manages the sd.cpp context, builds generation parameters, runs txt2img/img2img, reports progress |
| `ImageFilters` | `image_filters.h` | RealESRGAN upscaling and Canny preprocessing |
| `ModelManager` | `model_manager.h` | Scans model directories, indexes models by type, resolves names to paths |
| `OptionsManager` | `options_manager.h` | Loads/saves/persists `options.json` |
| `TaskStateManager` | `task_state.h` | Tracks per-task progress, live previews, and results |
| model detection | `model_detection.h` | Detects file format and infers model type from tensor keys |

`main.cpp` wires these together: it parses command line arguments, configures the
`ModelManager` directory paths, scans for models, starts a parent-process watchdog,
and runs the server. The overall layering and request lifecycle are described in
[architecture](architecture.md).

## Supported platforms and backends

The same C++ source is built for several platform/backend combinations by the Python
scripts in `scripts/`:

- Windows x64: CPU, CUDA, and Vulkan builds; plus a Vulkan ARM64 build.
- Linux x64: CPU, CUDA, and Vulkan builds.
- macOS: Metal builds for both arm64 (macOS 11.0+) and x64 (macOS 10.15+).

Build and packaging details live in [building-and-platforms](building-and-platforms.md),
and release cuts are described in [release-process](release-process.md).

## Repository map

```
sdkit/
├── CMakeLists.txt        # Top-level build: sdkit executable + submodule targets
├── generate_image.py     # Scripted end-to-end smoke test against a running server
├── include/              # Public component headers
├── src/                  # Implementations (including main.cpp, the entry point)
├── tests/                # Unit tests (model detection)
├── tools/                # CLI tools (modelinfo), built when SDKIT_BUILD_TOOLS=ON
├── scripts/              # Python build/release automation
│   ├── build/            # Per-platform build configuration (cpu, cuda, vulkan, metal)
│   └── upload/           # GitHub release upload logic and hashing
├── cmake/                # Toolchain files for Windows ARM64/LLVM builds
├── third_party/          # Vendored headers: crow, asio, base64.hpp, uuid.h
└── stable-diffusion.cpp/ # Git submodule: the sd.cpp fork (contains ggml inside it)
```

Important: `stable-diffusion.cpp` is a git submodule and is **empty until submodules
are initialized**. After cloning, run:

```
git submodule init
git submodule update --recursive
```

## Where to read next

- [architecture](architecture.md) - how the pieces fit together and how a request flows.
- [server-and-api](server-and-api.md) - every HTTP route with request/response shapes.
- [image-generation](image-generation.md) - txt2img/img2img internals, LoRAs, ControlNet.
- [models-and-detection](models-and-detection.md) - model directories, scanning, file inspection.
- [image-filters-and-upscaling](image-filters-and-upscaling.md) - upscalers and Canny.
- [building-and-platforms](building-and-platforms.md) - building for each platform.
- [development](development.md) - day-to-day workflow and conventions.
- [testing](testing.md) - what tests exist and how to run them.
- [release-process](release-process.md) - cutting a release.
