# Development

This page is the day-to-day guide for working on sdkit: setting up a local
environment, building and running the server, the conventions the code follows, and
a walkthrough for landing a new feature. For build backends and per-OS details see
[building-and-platforms](building-and-platforms.md); for how the demos work see
[demo-pages](demo-pages.md); for the test story see [testing](testing.md).

## Local dev environment

1. **Clone with submodules** - the third-party engines live in `third_party/` as git
   submodules, so a plain clone is not enough:

   ```bash
   git clone https://github.com/easydiffusion/sdkit.git
   cd sdkit
   git submodule init
   git submodule update --recursive
   ```

2. **Install a toolchain** - a C++20 compiler, CMake 3.16+, Ninja, and Python 3.
   Follow `CI.md` for the exact versions and setup commands used on the release
   runners (Windows: MSVC + Clang + Ninja; Linux: CUDA + Vulkan SDK).
3. **Configure and build** - for a first local build, a plain CPU build is the
   fastest path (see [building-and-platforms](building-and-platforms.md) for the
   backend-specific flags):

   ```bash
   mkdir build && cd build
   cmake -S .. -B . -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF
   cmake --build . --config Release
   ```

   This produces `build/bin/sdkit` (or `build/bin/sdkit.exe`) next to the shared
   `ggml` and `stable-diffusion` libraries, which are copied there automatically by a
   post-build step.

4. **Add a model** - point sdkit at a checkpoint directory at startup:

   ```bash
   cd build
   ./bin/sdkit --ckpt-dir <path-to-models> --log-level info
   ```

   See [server-and-api](server-and-api.md) for the full flag list. The server logs
   its scanned models, and `/v1/sdapi/v1/models` lists them at runtime.

## Quick check: generate_image.py

`generate_image.py` at the repository root is a scripted smoke test that drives a
running server over HTTP. Its flow:

1. `set_model(model_name, port=8188)` - POSTs `{"sd_model_checkpoint": <name>}` to
   `/v1/sdapi/v1/options` to select the model.
2. `generate_image(prompt, ...)` - POSTs the generation request to
   `/v1/sdapi/v1/txt2img` and writes the returned base64 images to timestamped
   `.png` files in the current directory.

It's useful as a fast end-to-end check after a change: start the server in one
terminal, run `python generate_image.py` in another, and verify that an image comes
out. It only exercises text-to-image (there's no scripted img2img / segmentation /
upscaler equivalent), so for the other features use the demo pages - see
[testing](testing.md).

There is no `run_server.bat` script in the repo; start the server by running the
binary directly from `build/bin/` with the flags you need.

## Coding conventions

The repo is small enough that the conventions are mostly implicit, but they're
consistent:

- **Header/source pairs** - every component is a class in `include/<name>.h` with its
  implementation in `src/<name>.cpp`, named after the feature (e.g. `image_generator`
  ↔ `ImageGenerator`, `model_manager` ↔ `ModelManager`). The header is the public
  surface; other components include only headers, never other components' `.cpp`
  internals.
- **Namespaces** - the code doesn't use a project-wide namespace; classes are global
  but prefixed by their domain (`ModelManager`, `ImageFilters`, ...). Small internal
  helpers live in `namespace detail` or `namespace controlnet` (see
  [image-filters-and-upscaling](image-filters-and-upscaling.md)).
- **Logging** - use the macros from `logging.h` (`LOG_INFO`, `LOG_WARNING`,
  `LOG_DEBUG`, ...) rather than `printf` or `std::cout`, so everything goes through
  the levelled logger and can be filtered with `--log-level`. Messages are
  printf-style with `%s` / `%d` / `%zu` etc. Use `LOG_WARNING` for recoverable
  problems (e.g. a skipped file during a model scan) and `LOG_ERROR` for failures
  that will be reported to the client.
- **Errors** - throw `std::runtime_error` with a human-readable message from
  component methods; the server's route handlers catch and convert that into a `500`
  response with a `message` field (see [server-and-api](server-and-api.md)). Don't
  return error codes from these methods - the handler layer is written around
  exceptions.
- **Threading** - a component that can be called from multiple HTTP handlers holds
  its own `std::mutex` and locks it for the whole method (see
  [architecture](architecture.md) for the per-component list and rationale). Locks
  are taken at the start of the public method and released at the end - no nested
  locking between components.
- **Resources** - images flow through `sd_image_t` buffers that must be freed once
  by whoever created them; use RAII guards or the `freeImage` helper from
  `image_utils.h` when a method can throw on the way out (see
  [utilities](utilities.md)).
- **JSON** - the server parses requests with crow's JSON reader and builds responses
  with `crow::json::wvalue`. Field names follow webui naming (`sampler_name`,
  `steps`, `cfg_scale`, ...) so existing clients keep working.
- **Indentation** - 4 spaces in `src/`, `include/`, `tools/`, and `tests/`; the demo
  pages use 4-space indentation too. Files use LF line endings.

## Adding a new feature: end-to-end walkthrough

Landing a feature means touching a predictable set of files. Here's the checklist,
with pointers to the layer each one belongs to:

1. **Component class** - create `include/<feature>.h` with the class's params/result
   structs and public methods, and `src/<feature>.cpp` with the implementation.
   Follow the pattern of an existing component (for example, `ImageSegmenter` is a
   good template for "lazy model + mutex + params/result struct"; see
   [segmentation](segmentation.md)). If the feature needs a third-party engine,
   vendor it under `third_party/` as a git submodule and add it as a CMake target in
   `CMakeLists.txt` (see [architecture](architecture.md)).
2. **Wire it into the server** - in `include/server.h`, add the component as a member
   of `Server` (shared pointer if other components need it, unique pointer if only
   the server owns it) and construct it in the `Server` constructor in
   `src/server.cpp`. Add the HTTP route in `Server::setupRoutes()`:
   `CROW_ROUTE(app_, "/v1/...")` with the HTTP method, a `try/catch` body that
   parses the request, calls the component, and returns a JSON response (or a
   `400`/`500` error object). Follow the request/response conventions in
   [server-and-api](server-and-api.md).
3. **Model support** - if the feature loads model files, add a `ModelType` entry,
   a directory setter (`set<...>Dir`) and CLI flag in `main.cpp` +
   `ServerParams`, and make sure the model scan handles the file extensions used by
   that model type (see [models-and-detection](models-and-detection.md)). If the
   feature needs format detection, extend `model_detection.h`.
4. **Tests** - if the feature has pure logic (e.g. file inspection, parameter
   validation, an algorithm) add a unit test in `tests/` following the pattern in
   `test_model_detection.cpp` (see [testing](testing.md)). Features that need real
   model weights or a GPU aren't covered by automated tests today - they're verified
   manually via a demo page or `generate_image.py`.
5. **Demo page** - add `demo/<feature>-demo.html` and register it in the `DEMOS`
   array in `demo/index.html`; no route change is needed because `handleDemoPage`
   serves any `.html` in `demo/`. Follow the walkthrough in
   [demo-pages](demo-pages.md).
6. **Verify manually** - start the server from `build/`, open
   `http://127.0.0.1:8188/`, exercise the new demo page, and run
   `python generate_image.py` as a smoke test.

For a worked example of the result, compare the route table in
[server-and-api](server-and-api.md) with the segmentation page in
[segmentation](segmentation.md) - that's what the above steps look like when they're
done.

## Documentation

New user-facing behavior (routes, CLI flags, model types, options keys) should be
reflected in the KB pages here - update the page the feature belongs to, and this
index in `kb/README.md` if a new page is needed. The KB is the primary documentation
for the v3 branch; the top-level `README.md` is kept minimal.
