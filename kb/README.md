# sdkit v3 knowledge base

This knowledge base documents the sdkit v3 codebase for developers joining the
project. It explains what the system is, how it is built and structured, how each
component works, and how releases are cut. It is written against the current state
of the `v3` branch (a C++ rewrite built on stable-diffusion.cpp, still in progress).

## Reading order

### Orientation

- [overview](overview.md) - what sdkit is, what v3 changes, the main components,
  supported backends, and a map of the repository.
- [architecture](architecture.md) - the layering, the request lifecycle for a
  txt2img call, threading and locking, and the CMake target layout.

### Feature pages

- [server-and-api](server-and-api.md) - every HTTP route with method, path, and
  request/response shape; startup, shutdown, the parent-process watchdog, and
  `options.json` persistence.
- [image-generation](image-generation.md) - the txt2img and img2img flows, all
  generation parameters, LoRA loading, ControlNet, reference images, VAE handling,
  and progress/interrupt via TaskState.
- [models-and-detection](models-and-detection.md) - `ModelManager`'s model types and
  directory conventions, plus `model_detection`: detecting file formats, extracting
  tensor keys, inferring model types, and the `modelinfo` tool.
- [image-filters-and-upscaling](image-filters-and-upscaling.md) - RealESRGAN
  upscaling and ControlNet preprocessing (Canny) in `ImageFilters`.

### Process pages

- [building-and-platforms](building-and-platforms.md) - building from source,
  submodule initialization, per-platform backends, and how the build scripts and
  vendored dependencies work.
- [development](development.md) - day-to-day workflow: local setup, running the
  server, coding conventions, and a walkthrough for adding a new feature.
- [testing](testing.md) - the model-detection unit tests, the scripted end-to-end
  smoke test, manual API testing, and an honest account of the coverage gaps.
- [release-process](release-process.md) - cutting a release: the release command,
  the platform matrix, artifact hashing, and uploading to GitHub releases.

## Notes

- Submodules must be initialized after cloning (`git submodule init` and
  `git submodule update --recursive`) - this is called out in
  [building-and-platforms](building-and-platforms.md) and
  [development](development.md).
