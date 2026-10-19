# Architecture

This page describes how sdkit v3 is put together: the layers, the life of a request,
and where the concurrency control lives. For route-by-route API details see
[server-and-api](server-and-api.md), for the generation pipeline see
[image-generation](image-generation.md), and for model handling see
[models-and-detection](models-and-detection.md).

## Layering

The stack has four layers, from top to bottom:

1. **HTTP layer** - `src/server.cpp` embeds [crow](https://github.com/CrowCpp/Crow)
   (header-only, in `third_party/crow.h` with `asio` underneath). It owns routing,
   JSON parsing, and response shaping. It never touches model weights directly.
2. **Component classes** - one class per feature, each a header/source pair in
   `include/` + `src/`:
   - `ImageGenerator` - txt2img / img2img
   - `ImageFilters` - upscaling and ControlNet preprocessing
   - `ImageSegmenter` - SAM3 segmentation
   - `ModelManager` - model registry, loading, unloading
   - `ModelDetection` - file inspection for classifying model files
   - `OptionsManager` - `options.json` persistence
   - `TaskStateManager` - progress and interruption per task
3. **Third-party engines** - the actual ML:
   - [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) -
     diffusion inference, LoRA application, VAE decode, upscalers, samplers.
   - [ggml](https://github.com/leejet/ggml) - tensor runtime and backend dispatch
     (CPU / CUDA / Vulkan / Metal). Built once as a shared library and shared by both
     stable-diffusion.cpp and sam3.cpp.
   - [sam3.cpp](https://github.com/cmdr2/sam3.cpp) - segmentation, built as a static
     library against the shared ggml.
4. **OS** - std::thread / std::mutex, sockets, file I/O.

A single `sdkit` executable links all of this together; stable-diffusion and ggml are
built as shared libraries and copied next to the binary by a CMake post-build step.

## include/ headers vs src/ implementations

Each public header in `include/` declares one component's interface:

| Header | Responsibility |
|---|---|
| `server.h` | `ServerParams` and the `Server` class (owns all other components) |
| `image_generator.h` | `ImageGenerationParams`, `ImageGenerator` |
| `image_filters.h` | `ImageFilters` |
| `image_segmenter.h` | `SegmentationParams`, `SegmentationResult`, `ImageSegmenter` |
| `model_manager.h` | `ModelManager` (paths per `ModelType`, load/unload helpers) |
| `model_detection.h` | file-inspection helpers used by the manager and the CLI tool |
| `options_manager.h` | `OptionsManager` |
| `task_state.h` | `TaskState`, `TaskStateManager` |
| `image_utils.h` | base64 encode/decode, image load/save via stb |
| `logging.h` | `LogLevel` and `log_message` |
| `string_utils.h` | small string helpers |

The corresponding `src/*.cpp` files hold the implementations. `src/filters/controlnet.hpp`
is a header-only helper that holds the custom Canny implementation used by
`ImageFilters` (see [image-filters-and-upscaling](image-filters-and-upscaling.md)).

`src/main.cpp` is the only entry point: it parses command-line flags into
`ServerParams`, instantiates `Server`, and runs it.

## Request lifecycle: a txt2img call

A POST to `/v1/sdapi/v1/txt2img` flows through these stages:

1. **Route entry** - crow dispatches to `Server::handleTxt2Img`, which parses the JSON
   body and calls `Server::generateImage(json_body, false)`.
2. **Task registration** - the handler reads `force_task_id` (defaulting to
   `"default_task"` if absent) and creates a `TaskState` via the shared
   `TaskStateManager`. This is what `/v1/internal/progress` and
   `/v1/sdapi/v1/interrupt` later key on.
3. **Parameter parsing** - the JSON body is mapped onto an `ImageGenerationParams`
   struct: prompt, negative prompt, LoRA paths/alphas, dimensions, steps, CFG, seed,
   batch size, sampler, scheduler, and so on. Webui-style sampler and scheduler names
   ("Euler a", "karras", ...) are translated to stable-diffusion.cpp's names here.
4. **Model resolution** - `ImageGenerator` asks `ModelManager` for the checkpoint,
   VAE, text encoders, LoRAs, and ControlNet models referenced by the params, loading
   them lazily if not already resident.
5. **Inference** - the generator builds the embedding and latents, then hands the
   sampling to stable-diffusion.cpp. Between steps it updates `TaskState` (progress
   step count and a preview image) and checks the interrupt flag so a canceled job
   stops early.
6. **Decode and filter** - latents are decoded to RGB via the VAE. If the request
   asked for upscaling, `ImageFilters::upscaleBatch` runs the configured RealESRGAN
   model over the results.
7. **Response** - images are base64-encoded and returned as JSON along with the
   final task state.

An img2img request follows the same path with two differences: the client provides an
input image (decoded from base64) and a `strength` value that controls how much noise
is added before sampling starts.

## Threading and locking

The server runs crow in multithreaded mode (`app_.multithreaded()`), so more than one
request can be in flight. The shared state is therefore protected by mutexes:

- **`ModelManager`** - has a mutex guarding model loading/unloading and the registry.
  Two concurrent generation requests could otherwise race each other into loading the
  same checkpoint twice, or evict a model another request is still using. Model
  operations are relatively rare (once per job or when the user refreshes the list),
  so the lock is not on the hot path of individual sampling steps.
- **`TaskStateManager`** - guards creation, updates, and lookup of per-task state.
  The progress endpoint, the generation handler, and the interrupt handler all touch
  the same map concurrently, so every read/write of a task's state goes through the
  manager's lock.
- **`OptionsManager`** - guards get/set of the options and their serialization to
  `options.json`, since the options endpoint can be hit while a generation is running.
- **`ImageSegmenter`** - owns its own mutex around the SAM3 model. sam3.cpp has no
  internal thread-safety, so the segmenter serializes all calls through one model
  instance (see [segmentation](segmentation.md)).

`ImageGenerator` itself serializes work through the stable-diffusion context; only one
generation runs on the sd.cpp context at a time, with its progress updates funneled
through `TaskStateManager` (which is thread-safe). If a second generation request
arrives while one is running, it is queued behind the first by the generator's own
lock rather than being refused.

## CMake target layout

The top-level `CMakeLists.txt` (C++20) defines these targets:

| Target | Type | Source | Notes |
|---|---|---|---|
| `ggml` | shared | `third_party/ggml` | built first, so both sd.cpp and sam3.cpp link the *same* ggml. sd.cpp skips its vendored copy when a `ggml` target already exists. |
| `sam3` | static | `third_party/sam3.cpp/sam3.cpp` | single-file library; compiles its own stb symbols as `static` to avoid clashes with the sdkit binary's own stb usage. |
| `stable-diffusion` | shared | `third_party/stable-diffusion.cpp` | needs `GGML_MAX_NAME=128` defined before ggml is included so all consumers share one ABI. |
| `sdkit` | executable | `src/*.cpp` (recursive glob) | links ggml, stable-diffusion, sam3, and Threads. |
| `modelinfo` (optional) | executable | `tools/modelinfo.cpp` | only built when `SDKIT_BUILD_TOOLS=ON`; see [models-and-detection](models-and-detection.md). |

Notes on the build:

- Backends (CUDA / Metal / Vulkan / CPU) are chosen by passing `GGML_*` CMake
  variables, which both sd.cpp and sam3 respect because they use the shared ggml.
- `BUILD_SHARED_LIBS`, `SD_BUILD_SHARED_LIBS`, and `SD_BUILD_SHARED_GGML_LIB` are set
  to `ON` so the runtime artifacts are small executables next to their shared libs;
  a post-build step copies the DLLs/SOs next to `sdkit`.
- On Windows the build sets `_WIN32_WINNT=0x0601` (Windows 7+) and adds `/bigobj`.
- On Linux/macOS, RPATH is set to `$ORIGIN` / `@executable_path` so the shared ggml
  and stable-diffusion libraries are found next to the binary.

See [building-and-platforms](building-and-platforms.md) for how this is driven from
the Python build scripts and per-platform specifics.
