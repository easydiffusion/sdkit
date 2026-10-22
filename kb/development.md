# Development

This page describes the day-to-day workflow: setting up a local environment, building
and running the server, the coding conventions used in the repo, and a walkthrough
for adding a new feature. Build details and platform specifics live in
[building-and-platforms](building-and-platforms.md); testing is covered in
[testing](testing.md).

## Setting up a local environment

1. Clone the repo and initialize submodules - this step is mandatory, the build fails
   without it because the diffusion engine and ggml live inside the
   `stable-diffusion.cpp` submodule:

   ```
   git submodule init
   git submodule update --recursive
   ```

2. Install the tools: CMake 3.16+, a C++20 compiler, Ninja, and Python 3 with
   `requests` (used by the scripts and the smoke test).
3. Configure and build (CPU build by default):

   ```
   cmake -S . -B build
   cmake --build build --config Release
   ```

   The server executable lands in `build/bin/`, next to the stable-diffusion and
   ggml shared libraries that the post-build step copies there. Add
   `-DSDKIT_BUILD_TOOLS=ON` if you also want the `modelinfo` tool.

## Running the server locally

Start the server with the directories you want it to scan:

```
./build/bin/sdkit --port 8188 \
  --ckpt-dir /path/to/models/StableDiffusion \
  --vae-dir /path/to/models/VAE \
  --lora-dir /path/to/models/Lora \
  --embeddings-dir /path/to/models/embeddings \
  --controlnet-dir /path/to/models/ControlNet \
  --realesrgan-models-path /path/to/models/RealESRGAN
```

Engine flags such as `--vae-on-cpu`, `--vae-tiling`, `--offload-to-cpu`,
`--diffusion-fa`, `--clip-on-cpu`, and `--control-net-cpu` are startup-only; see
[server-and-api](server-and-api.md) for the full list.

Two conveniences worth knowing:

- A local (gitignored) `run_server.bat` is a handy place to keep your usual
  command line; it is not committed, so each developer maintains their own.
- `generate_image.py` is a scripted smoke test: it points the server at a checkpoint
  via the options API, generates one image, and saves it under `generated_images/`.
  Edit the `model_name` in the script, start the server, then run
  `python generate_image.py`. See [testing](testing.md).

## Coding conventions

The conventions are consistent across the codebase and worth following:

- **Header/source pairs.** Each component is a class with its header in `include/`
  and implementation in `src/` (`image_generator.h` / `image_generator.cpp`, etc.).
  New `src/*.cpp` files are picked up automatically by the CMake glob; new headers go
  in `include/` so they can be included from anywhere. Small helpers shared by
  nothing else can be `static` functions inside a single `.cpp`.
- **Naming.** Classes and methods use PascalCase (`ImageGenerator::ensureModelLoaded`),
  member variables use lower_snake_case with a trailing underscore
  (`current_model_path_`), and free functions/enums follow the sd.cpp style they
  interoperate with (`sd_image_t`, `EULER_A_SAMPLE_METHOD`).
- **Logging.** Use the `LOG_VERBOSE/LOG_DEBUG/LOG_INFO/LOG_WARNING/LOG_ERROR` macros
  from `include/logging.h`; they printf-style format with a timestamp and level, and
  the level is controlled by `--log-level`. Warnings and errors go to stderr,
  everything else to stdout. stable-diffusion.cpp's own log output is bridged into
  this system via the `sd_log_cb` callback.
- **HTTP layer.** All routes live in `Server::setupRoutes()` in `src/server.cpp`,
  use crow's JSON for parsing and building responses, and return JSON `{"message":
  ...}` bodies for errors. Webui-compatible naming (sampler/scheduler/upscaler names,
  ControlNet scripts, hash suffixes) is translated in `server.cpp`, keeping component
  classes free of webui quirks.
- **Engine API.** Interaction with stable-diffusion.cpp happens only through its C
  API (`new_sd_ctx`, `generate_image`, `upscale`, `preprocess_canny`, callbacks).
  Buffers returned by the engine or stb loaders must be freed via `freeImage`/`free`;
  the codebase is careful to free them on all paths, including errors.
- **Python.** The `scripts/` package is plain Python 3 with `flake8` configured
  (max line length 120; see `.flake8`).

## Adding a new feature: end-to-end walkthrough

Suppose you're adding a new capability (for example, a new kind of image
processing). The steps below mirror how existing features are structured:

1. **Component class.** Create `include/my_feature.h` declaring the class and its
   params/results, and `src/my_feature.cpp` implementing it. Follow the pattern of
   `ImageFilters`: take the dependencies you need as `std::shared_ptr` constructor
   arguments, keep engine calls in the implementation file, and free engine-owned
   memory on every path. If the feature needs model files, add a directory setter to
   `ModelManager` and a matching CLI argument in `main.cpp`'s `CommandLineArgs`/
   `print_usage`, plus a `server_params` field if the engine must be configured at
   startup (see [models-and-detection](models-and-detection.md)).
2. **Route.** Add a handler method to `Server` (declared in `include/server.h`),
   register it in `setupRoutes()` with the right method/path, parse the JSON body
   with crow, and return JSON - including the `{"message": ...}` shape for errors.
   If a webui-compatible alias is involved, put the name translation in a static
   helper in `server.cpp`.
3. **Progress/tasks (if long-running).** Create a task via `TaskStateManager`, hand
   the task id to your component, and update progress from callbacks, mirroring how
   `ImageGenerator` does it (see [image-generation](image-generation.md)). Expose a
   polling route if clients need one.
4. **Model plumbing (if the feature uses models).** Extend `ModelType` and add a
   directory setter if there's a new model category, or classify new file kinds in
   `model_detection` and add a test case in `tests/test_model_detection.cpp` (see
   [testing](testing.md)).
5. **Tests.** Add unit tests to `tests/` where the logic is testable without a GPU
   or model files (like the model-detection tests), and verify the feature end to
   end with a small script or `generate_image.py` against a running server.
6. **Docs.** Update the relevant page in `kb/` so the next reader finds accurate
   documentation.

## Useful references

- [architecture](architecture.md) - how the layers fit together and who owns which
  mutex.
- [building-and-platforms](building-and-platforms.md) - all the build-time flags.
- [testing](testing.md) - what exists, how to run it, and the known gaps.
