# Architecture

This page describes the big-picture design of sdkit v3: the layering, how a request
travels through the system, threading and locking, and how the CMake targets are
organized. Component-level details are covered in the linked pages:
[server-and-api](server-and-api.md), [image-generation](image-generation.md), and
[models-and-detection](models-and-detection.md).

## Layering

From top to bottom, the system is layered as follows:

1. **crow HTTP server.** The `Server` class (`src/server.cpp`, header
   `include/server.h`) owns a `crow::SimpleApp`. crow itself is vendored in
   `third_party/` (crow headers plus the asio networking library it uses). The app is
   bound to `127.0.0.1`, runs multithreaded, and every route handler is a lambda
   registered in `Server::setupRoutes()` that delegates to component classes.
2. **Component classes in `src/`.** `OptionsManager`, `TaskStateManager`,
   `ModelManager`, `ImageFilters`, and `ImageGenerator` each encapsulate one concern
   and are shared via `std::shared_ptr` (the `Server` owns them and passes references
   down). These classes never touch sockets; they only talk to each other and to the
   engine below.
3. **stable-diffusion.cpp (the engine).** Vendored as a git submodule. sdkit calls its
   C API: `new_sd_ctx` / `free_sd_ctx` for the model context, `generate_image` for
   actual diffusion, `new_upscaler_ctx` / `upscale` for RealESRGAN, and
   `preprocess_canny` for ControlNet preprocessing. Progress and preview events come
   back through `sd_set_progress_callback` / `sd_set_preview_callback`.
4. **ggml.** The tensor library used by stable-diffusion.cpp. It is built as a shared
   library from within the stable-diffusion.cpp submodule's own CMake
   (`stable-diffusion.cpp/ggml`), and sdkit links against it directly for utilities
   like GGUF parsing (`gguf.h` is included by `src/model_detection.cpp`). The backend
   (CPU, CUDA, Vulkan, Metal) is selected at build time via sd.cpp/ggml CMake flags.

## Component map: include/ headers and src/ implementations

| Header (include/) | Implementation (src/) | Responsibility |
|---|---|---|
| `server.h` | `server.cpp` | Routes, request parsing, response shaping; defines `ServerParams` |
| `image_generator.h` | `image_generator.cpp` | SD context lifecycle, generation params, progress/preview callbacks |
| `image_filters.h` | `image_filters.cpp` | Upscaling and ControlNet preprocessing |
| `model_manager.h` | `model_manager.cpp` | Directory scanning and the model index |
| `model_detection.h` | `model_detection.cpp` | Format detection and model-type inference |
| `options_manager.h` | `options_manager.cpp` | `options.json` load/save |
| `task_state.h` | `task_state.cpp` | Task registry for progress/results |
| `image_utils.h` | `image_utils.cpp` | base64 <-> `sd_image_t`, resizing, freeing |
| `logging.h` | `logging.cpp` | Log levels, colored logging, sd.cpp log bridge |
| `string_utils.h` | `string_utils.cpp` | UTF-8/UTF-16 path conversions for Windows |

`main.cpp` (in `src/`, no header) is the entry point: it parses CLI arguments,
configures the `ModelManager`, and starts the `Server`. Helper code with no public
surface (e.g. the webui-name conversion tables) is `static` inside `server.cpp`.

## Request lifecycle: a txt2img call end to end

1. The client `POST`s a JSON body to `/v1/sdapi/v1/txt2img`. crow dispatches to the
   handler lambda, which calls `Server::handleTxt2Img()`.
2. `handleTxt2Img` parses the JSON body and calls `generateImage(json_body, false)`.
3. `generateImage` extracts a task id from the optional `force_task_id` field
   (default `default_task`) and creates a task in the `TaskStateManager`.
4. It maps the JSON into an `ImageGenerationParams` struct: prompt, sizes, steps, CFG,
   seed, sampler/scheduler (converting webui-style names), LoRA path/alpha lists,
   ControlNet parameters extracted from `alwayson_scripts.controlnet`, and, for img2img,
   the init image, mask, and denoising strength. Dimensions are rounded to multiples
   of 64 when a ControlNet model is requested.
5. It calls `ImageGenerator::generateTxt2Img(params, task_id)`.
6. `generateInternal` first calls `ensureModelLoaded()`, which reads the selected
   checkpoint (and optional VAE/text encoders/ControlNet) from the options and
   `ModelManager`, creates or recreates the `sd_ctx_t` if anything changed.
7. It then locks the generator mutex, registers progress/preview callbacks, fills an
   `sd_img_gen_params_t`, and calls the engine's `generate_image()`. During sampling,
   the engine callbacks write progress (and JPEG-encoded live previews) into the
   `TaskStateManager`, which the client reads by polling `/v1/internal/progress`.
8. The resulting `sd_image_t` array is converted to base64 PNGs, the task is marked
   completed with an info string, and the HTTP response returns `{"images": [...],
   "info": "..."}`. Errors become HTTP 500 with a `message` field.

## Threading and locking

The crow server is multithreaded, so shared state is guarded by mutexes:

- `ModelManager::mutex_` - guards the directory map and the model index. Scanning can
  be triggered from HTTP refresh routes while requests read the index.
- `OptionsManager::mutex_` - guards the in-memory options and the `options.json` file.
  Options can be updated over HTTP while a generation is being prepared.
- `TaskStateManager::mutex_` - guards the task map. Progress callbacks from the
  engine thread and progress polls from HTTP threads run concurrently.
- `ImageGenerator::mutex_` - the big one. It serializes access to the `sd_ctx_t` and
  the actual generation, so only one generation runs at a time; a second request
  blocks on this mutex until the first finishes. `interrupt()` also takes this mutex.
- `g_callback_data` + `g_callback_mutex` (in `image_generator.cpp`) - a process-global
  structure that captures the current task id for the engine's progress/preview
  callbacks, because the C callbacks carry no per-context user data. It is cleared
  after each generation.
- A log mutex in `logging.cpp` serializes log lines.

## CMake target layout

The top-level `CMakeLists.txt` defines:

- `sdkit` - the server executable. All `src/*.cpp` files are globbed into it. It links
  `stable-diffusion`, `ggml`, and `Threads`, plus `ws2_32` on Windows. A post-build
  step copies the `stable-diffusion` and `ggml` shared libraries next to the
  executable, and RPATH settings make Linux/macOS binaries resolve them in place.
- `stable-diffusion` and `ggml` - shared libraries built from the stable-diffusion.cpp
  submodule (`add_subdirectory(stable-diffusion.cpp)`), with
  `SD_BUILD_SHARED_LIBS`/`SD_BUILD_SHARED_GGML_LIB` enabled.
- `modelinfo` - a CLI inspection tool in `tools/`, only built when
  `-DSDKIT_BUILD_TOOLS=ON`. It reuses `src/model_detection.cpp` and `src/logging.cpp`.

Tests in `tests/` are not wired into the CMake build; see [testing](testing.md).

Build-time backend selection (CPU/CUDA/Vulkan/Metal) happens through flags passed to
CMake by the scripts in `scripts/build/`; see
[building-and-platforms](building-and-platforms.md).
