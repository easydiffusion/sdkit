# sdkit v3 Knowledge Base

This folder is the internal documentation for sdkit v3 - a C++ HTTP server that runs
Stable Diffusion inference via stable-diffusion.cpp. It's written for a new developer
joining the project: each page explains one area of the codebase in plain prose, with
pointers into the source. If you're looking for a quick start instead, see
[development.md](development.md).

Pages in reading order:

1. [overview.md](overview.md) - what sdkit is, what v3 changes, the main components,
   and a map of the repository directories.
2. [architecture.md](architecture.md) - the layering (crow HTTP server → component
   classes → third-party engines), request lifecycle, threading/locking, and the
   CMake target layout.
3. [server-and-api.md](server-and-api.md) - the HTTP routes, request/response shapes,
   options.json persistence, and server startup/shutdown.
4. [image-generation.md](image-generation.md) - txt2img and img2img internals, params,
   LoRA/ControlNet/reference-image handling, and progress/interrupt.
5. [models-and-detection.md](models-and-detection.md) - model types and directories,
   scanning, and detecting GGUF vs safetensors via magic bytes and tensor keys.
6. [image-filters-and-upscaling.md](image-filters-and-upscaling.md) - RealESRGAN
   upscaling and ControlNet preprocessing, including the custom Canny implementation.
7. [segmentation.md](segmentation.md) - SAM3 segmentation with text, point, and box
   prompting; masks returned as base64 grayscale PNGs.
8. [demo-pages.md](demo-pages.md) - the interactive browser demos in `demo/` and how
   they call the API.
9. [building-and-platforms.md](building-and-platforms.md) - building from source, the
   CPU/CUDA/Vulkan/Metal backends, and how `third_party/` is vendored.
10. [development.md](development.md) - local dev setup, coding conventions, and an
    end-to-end walkthrough for adding a new feature.
11. [testing.md](testing.md) - what automated tests exist, how to run them, and the
    manual/scripted checks for the inference features.
12. [release-process.md](release-process.md) - cutting a release: the build matrix,
    artifact hashing, and the upload flow.
13. [utilities.md](utilities.md) - the small shared plumbing: logging, string/path
    helpers, image encode/decode, and task state.
