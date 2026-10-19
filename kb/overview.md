# Overview

`sdkit` (**s**table **d**iffusion **kit**) is a C++ library that runs Stable Diffusion
inference behind a small HTTP API. A typical client (for example, the EasyDiffusion UI)
sends a JSON request describing a text-to-image or image-to-image job; the server loads
the required models, runs the diffusion pipeline, applies any filters or segmentation
steps, and returns PNG images over the same HTTP connection.

## What v3 changes

Version 3 is a ground-up rewrite of the older Python-based sdkit. The inference core is
now [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp), which itself
runs on [ggml](https://github.com/leejet/ggml). This moves the whole stack to C++ and
changes what sdkit is:

- The library is a set of C++ component classes (generator, filters, segmenter, model
  manager) wrapped by an HTTP server, rather than Python modules.
- Model weights are consumed as GGUF or safetensors files via stable-diffusion.cpp and
  a small built-in detection layer (see [models-and-detection](models-and-detection.md)).
- Segmentation is handled by a SAM3 wrapper (`third_party/sam3.cpp`) instead of the
  older rembg/RVM approach.
- The server and API surface are new (see [server-and-api](server-and-api.md)).

## Main components

The code is organized into a handful of component classes, each in `src/` with a public
header in `include/`:

- **Server** (`src/server.cpp`, `include/server.h`) - the crow-based HTTP layer that
  exposes routes for options, model listing, generation, progress, interruption,
  ControlNet preprocessing, upscaling, and segmentation.
- **ImageGenerator** (`src/image_generator.cpp`, `include/image_generator.h`) - wraps
  stable-diffusion.cpp and performs txt2img and img2img, including LoRA, ControlNet,
  VAE, and vision-based reference-image handling (Qwen and similar models).
- **ImageFilters** (`src/image_filters.cpp`, `include/image_filters.h`) - RealESRGAN
  upscaling and ControlNet preprocessing (Canny) via
  [image-filters-and-upscaling](image-filters-and-upscaling.md).
- **ImageSegmenter** (`src/image_segmenter.cpp`, `include/image_segmenter.h`) - SAM3
  based segmentation with text, point, and box prompting (see
  [segmentation](segmentation.md)).
- **ModelManager** (`src/model_manager.cpp`, `include/model_manager.h`) - registry of
  model paths and types, plus lazy loading and unloading.
- **Model detection** (`src/model_detection.cpp`, `include/model_detection.h`) -
  inspects model files (magic bytes, tensor keys) to classify them.
- **OptionsManager** (`src/options_manager.cpp`, `include/options_manager.h`) -
  persists server options to `options.json`.
- **TaskState** (`src/task_state.cpp`, `include/task_state.h`) - per-request generation
  progress and interruption flags.
- **main.cpp** (`src/main.cpp`) - entry point that parses startup flags, starts the
  server, and acts as a parent-process watchdog for children it spawns.

## Supported platforms and backends

The build system produces several backend flavors driven by stable-diffusion.cpp's
ggml backends:

- **CPU** - supported everywhere; the default fallback.
- **CUDA** - Linux x86_64 (and used for Windows NVIDIA builds as well).
- **Vulkan** - Windows and Linux; covers AMD and Intel GPUs where drivers allow.
- **Metal** - macOS (Apple Silicon).

Exact build matrices and per-OS setup steps live in
[building-and-platforms](building-and-platforms.md) and in the repo's `CI.md`.

## Current status

The v3 branch is a work in progress. What exists and works today:

- HTTP server with generation, upscaling, segmentation, and ControlNet preprocessing.
- Model loading for checkpoints, VAEs, LoRAs, embeddings, ControlNets, upscalers,
  text encoders, and segmentation models.
- C++ unit tests for model detection only. There are no automated tests yet for
  generation, filters, or segmentation; those are exercised through demo pages and
  manual checks (see [testing](testing.md)).

Some features are stubbed but not yet wired up - face restoration models (GFPGAN,
CodeFormer) are listed as model types but not yet reachable from a route. The API and
internal interfaces are still evolving, so expect churn between commits.

## Repository map

```
sdkit/
├── src/             component implementations (server, generator, filters, etc.)
│   └── filters/     header-only filter helpers (controlnet.hpp)
├── include/         public headers matching the src/ components
├── demo/            self-contained browser demos for each feature
├── scripts/         build, CI, and upload tooling (Python)
├── tests/           C++ unit tests
├── tools/           small CLI utilities (modelinfo)
├── third_party/     vendored engines: stable-diffusion.cpp, ggml, sam3.cpp, plus
│                    single-header libs (crow, asio, stb, uuid, base64)
├── cmake/           CMake helper modules
├── CI.md            runner setup and release commands
├── CMakeLists.txt   top-level build definition
└── generate_image.py  end-to-end smoke test against a running server
```

Note: `third_party/` entries are git submodules. After cloning, run
`git submodule init` and `git submodule update --recursive`, or the build will fail
with missing headers and sources.

## Where to go next

- [architecture](architecture.md) - how the layers fit together and what happens on a
  request.
- [server-and-api](server-and-api.md) - every HTTP route and its request/response shape.
- [image-generation](image-generation.md) - the txt2img and img2img pipelines in detail.
- [models-and-detection](models-and-detection.md) - model types, directories, and file
  inspection.
- [building-and-platforms](building-and-platforms.md) - how to compile everything.
