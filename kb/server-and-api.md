# Server and API

This page documents the HTTP layer: how the server starts, what routes exist, and how
options are persisted. The big-picture layering is in
[architecture](architecture.md); route handler internals that belong to a specific
feature are described on that feature's page
([image-generation](image-generation.md),
[image-filters-and-upscaling](image-filters-and-upscaling.md),
[segmentation](segmentation.md)).

## Entry point and startup

`src/main.cpp` is the process entry point. Startup proceeds as follows:

1. Signal handlers for `SIGINT`/`SIGTERM` are installed so the server shuts down
   cleanly.
2. Command-line arguments are parsed (see below).
3. A `ModelManager` is created, configured with the model directories from the
   command line, and `scanAllDirectories()` builds the model index.
4. If `--parent-pid` was given, a watchdog thread starts (see below).
5. A `ServerParams` struct is filled with the port, model manager, and the CPU
   offload / tiling flags, and `Server::run()` starts the crow HTTP loop.

### Command-line parameters

| Flag | Effect |
|---|---|
| `--port <n>` | Listen port (default 8188) |
| `--log-level <level>` | `verbose`, `debug`, `info`, `warning`, or `error` (default `info`) |
| `--parent-pid <pid>` | Enable the parent-process watchdog |
| `--ckpt-dir`, `--vae-dir`, `--hypernetwork-dir`, `--gfpgan-models-path`, `--realesrgan-models-path`, `--lora-dir`, `--codeformer-models-path`, `--embeddings-dir`, `--controlnet-dir`, `--text-encoder-dir`, `--segmentation-models-path` | Model directories, one per model type |
| `--vae-on-cpu`, `--vae-tiling`, `--vae-tile-size <WxH>` | VAE placement / tiling |
| `--offload-to-cpu`, `--diffusion-fa`, `--control-net-cpu`, `--clip-on-cpu`, `--chroma-disable-dit-mask` | Inference placement and quality knobs passed through to stable-diffusion.cpp |

These placement flags live in `ServerParams` (`include/server.h`) and are set once at
startup; they are not changeable per request. They flow into `ImageGenerator`, which
uses them when constructing the stable-diffusion context.

### Parent-process watchdog

When sdkit is launched as a child of another process (for example, from a UI), the
parent may be killed without any chance to stop the child. The watchdog thread
(`parent_watchdog` in `src/main.cpp`) polls the parent PID every second:

- On Windows it uses `OpenProcess` + `WaitForSingleObject` with a 1-second timeout.
- On Unix-like systems it calls `kill(parent_pid, 0)` and checks the result.

If the parent is gone, the watchdog logs a warning, calls `g_server->stop()`, and the
process exits. The loop also honors the global `g_should_exit` flag, so a normal
signal-driven shutdown does not leave the thread hanging.

## Server construction and shutdown

`Server` (`include/server.h`) owns every other component:

- `OptionsManager` (shared pointer) - created in the constructor, then `load()` is
  called to read `options.json`.
- `TaskStateManager` (shared pointer) - created in the constructor.
- `ImageFilters` (shared pointer) - created with the model manager.
- `ImageSegmenter` (unique pointer) - created with the model manager.
- `ImageGenerator` (unique pointer) - created with the task state manager, options
  manager, model manager, and image filters, plus the `ServerParams`.

The destructor calls `stop()`, which sets `should_stop_` and calls `app_.stop()` to
end the crow loop. `run()` binds to `127.0.0.1` only (the server is not exposed
directly to the network; clients reach it via a local port), enables multithreaded
request handling, and blocks until stopped.

Crow's default log handler is replaced with `FilteredLogHandler`, which routes crow's
messages into sdkit's own `log_message` logger and drops the noisy `/ping` and
`/internal/progress` request logs.

## Routes

All routes are defined in `Server::setupRoutes()` in `src/server.cpp`. Routes follow
the API conventions of the Forge/Automatic1111 webui (`/v1/sdapi/v1/...`) so that
existing clients work, plus a few `/v1/internal/...` routes for sdkit-specific needs.

| Method | Path | Purpose |
|---|---|---|
| GET | `/v1/internal/ping` | Liveness check; returns `200 OK` |
| GET | `/` | Serves `demo/index.html` |
| GET | `/demo/<name>.html` | Serves a demo page (see [demo-pages](demo-pages.md)) |
| GET | `/v1/sdapi/v1/models` | List all models grouped by type |
| GET | `/v1/sdapi/v1/options` | Return current options |
| POST | `/v1/sdapi/v1/options` | Update options and save them |
| POST | `/v1/sdapi/v1/txt2img` | Text-to-image generation |
| POST | `/v1/sdapi/v1/img2img` | Image-to-image generation |
| POST | `/v1/internal/progress` | Poll task progress |
| POST | `/v1/sdapi/v1/interrupt` | Interrupt the current generation |
| POST | `/v1/sdapi/v1/extra-batch-images` | Upscale (or pass through) a list of images |
| POST | `/v1/controlnet/detect` | Run ControlNet preprocessing (Canny) on images |
| POST | `/v1/sdapi/v1/segment` | Run segmentation on an image |
| POST | `/v1/sdapi/v1/refresh-checkpoints` | Rescan checkpoint directory |
| POST | `/v1/sdapi/v1/refresh-vae-and-text-encoders` | Rescan VAE and text encoder directories |

### Request/response shapes

**`/v1/sdapi/v1/models`** - No request body. Responds with a JSON object mapping each
model type name (for example `CHECKPOINT`, `LORA`, `VAE`, `CONTROLNET`,
`SEGMENTATION`) to a list of model names, as returned by
`ModelManager::getAllModelsGrouped()`.

**`/v1/sdapi/v1/options`** - GET returns the options object as stored by
`OptionsManager`. POST takes the same shape and applies each key via
`OptionsManager::setOptions`, which also writes the file. Responds with `200 OK` or a
`{"message": ...}` error object.

**`/v1/sdapi/v1/txt2img` and `/v1/sdapi/v1/img2img`** - The full parameter set is
described in [image-generation](image-generation.md). Both share one handler path:
`handleTxt2Img` / `handleImg2Img` parse the body and call the shared
`generateImage(json_body, is_img2img)`. The request carries an optional
`force_task_id` (defaults to `"default_task"`); the response is:

```json
{
  "images": ["<base64 png>", ...],
  "info": "{\"prompt\": ..., \"steps\": ..., ...}"
}
```

The `info` value is itself a JSON string (the same payload is also written into the
task state as `infotexts`). On failure the server responds with `500` and a
`{"message": ...}` body, and the task state is completed with the error.

**`/v1/internal/progress`** - POST body `{"id_task": "..."}`. Returns the task's
`completed`, `progress`, `live_preview` (base64), and `id_live_preview` fields from
`TaskState`, or `404` if the task ID is unknown.

**`/v1/sdapi/v1/interrupt`** - No body. Calls `ImageGenerator::interrupt()`, which
sets the generator's interrupt flag; the running generation checks it between steps.

**`/v1/sdapi/v1/extra-batch-images`** - POST body `{"imageList": [{"data":
"<base64>"}...], "upscaler_1": "...", "upscaling_resize": <n>}`. If
`upscaling_resize` is greater than zero, the images are upscaled via
`ImageFilters::upscaleBatch` (upscaler names are translated from webui-style names
like `"R-ESRGAN 4x+"`). Otherwise the images are returned unchanged. Response is
`{"images": [...]}`. If an upscaler was requested but none is available, the server
returns `500` with a message asking to configure an upscaler model in options.

**`/v1/controlnet/detect`** - POST body with `controlnet_input_images` (list of
base64 images), optional `controlnet_module` (defaults to `"canny"`). Runs
`ImageFilters::applyControlNetFilterBatch` and returns `{"images": [...]}`. Only
Canny is implemented today; see
[image-filters-and-upscaling](image-filters-and-upscaling.md).

**`/v1/sdapi/v1/segment`** - POST body per [segmentation](segmentation.md): required
`image` and `model`, plus optional `text_prompt`, `positive_points` /
`negative_points` (lists of `[x, y]`), `boxes` (lists of `[x0, y0, x1, y1]`), and
`score_threshold`. Returns `{"masks": [...], "scores": [...]}`.

**Refresh endpoints** - No body; they call `ModelManager::refreshCheckpoints()` or
`refreshVaeAndTextEncoders()` to rescan the corresponding directories after files are
added or removed on disk.

### Error handling convention

Every handler wraps its work in `try/catch`. Malformed JSON returns `400` with
`{"message": "Invalid JSON"}` (or a field-specific message). Exceptions are logged
via sdkit's logger and returned as `500` with `{"message": "..."}`. Demo-page lookup
returns `404` with a hint that the server should be started from the repository root,
or the `demo/` directory placed next to the executable.

## Demo page serving

`handleDemoPage` serves plain `.html` files from the `demo/` directory. It rejects
anything that is not a simple filename (no path separators, no `..`, must end in
`.html`) and looks for the file relative to the current working directory first, then
relative to the executable's directory (on Windows via `GetModuleFileNameA`). Serving
the demos same-origin is what allows their JavaScript to call the API from the
browser; see [demo-pages](demo-pages.md) for the pages themselves.

## Options persistence

`OptionsManager` (`include/options_manager.h`, `src/options_manager.cpp`) keeps a
`crow::json::wvalue` map of options and persists it to `options.json` (name
configurable via the constructor; the server uses the default). Behavior:

- On first load, if the file is missing or empty, it seeds these defaults:
  - `sd_model_checkpoint`: `""` (empty; the generator then falls back to the first
    checkpoint found in the checkpoint directory)
  - `live_previews_enable`: `true`
  - `CLIP_stop_at_last_layers`: `1`
  - `sdxl_clip_l_skip`: `false`
  - `samples_format`: `"png"`
  - `forge_additional_modules`: empty list
- `setOptions` merges the incoming JSON key-by-key into the in-memory map and
  immediately writes the file. Types are copied as strings, numbers, booleans, or
  lists of strings.
- Every access (load, save, get, set) is guarded by a mutex, so the options endpoint
  can be hit while a generation is in flight.
- The destructor calls `save()`, so options are also flushed at shutdown.

The option keys mirror the webui naming (they are consumed by `ImageGenerator`, e.g.
`sd_model_checkpoint` selects the model and `CLIP_stop_at_last_layers` maps to clip
skip; see [image-generation](image-generation.md)).
