# Server and API

The HTTP layer is owned by the `Server` class (`include/server.h`,
`src/server.cpp`). It runs a vendored crow HTTP app bound to `127.0.0.1` on the port
given at startup (default 8188), in multithreaded mode. It is an API-only process,
with no static file serving.

## Startup and shutdown

`main.cpp` is the entry point:

1. It installs `SIGINT`/`SIGTERM` handlers so the server shuts down cleanly.
2. It parses command line arguments (`parse_args`):
   - `--port <port>` (default 8188) and `--log-level <level>` (default `info`).
   - `--parent-pid <pid>` - enables the parent-process watchdog (see below).
   - Model directory arguments: `--ckpt-dir`, `--vae-dir`, `--hypernetwork-dir`,
     `--gfpgan-models-path`, `--realesrgan-models-path`, `--lora-dir`,
     `--codeformer-models-path`, `--embeddings-dir`, `--controlnet-dir`,
     `--text-encoder-dir`.
   - Engine flags that can only be set at startup (they affect how the sd.cpp context
     is created): `--vae-on-cpu`, `--vae-tiling`, `--vae-tile-size <size>` (in
     pixels, default 256x256), `--offload-to-cpu`, `--diffusion-fa`,
     `--control-net-cpu`, `--clip-on-cpu`, `--chroma-disable-dit-mask`.
3. It creates a `ModelManager`, applies the directory arguments, and calls
   `scanAllDirectories()`.
4. If `--parent-pid` was given, it starts a watchdog thread.
5. It builds a `ServerParams` (port, model manager, and the engine flags above) and
   constructs the `Server`. The `Server` constructor creates the `OptionsManager`,
   `TaskStateManager`, `ImageFilters`, and `ImageGenerator`, loads `options.json`,
   and registers all routes.

`Server::run()` blocks on `app_.run()`. `Server::stop()` sets a stop flag and calls
`app_.stop()`. Both the signal handlers and the watchdog call `stop()`.

### Parent-process watchdog

easy-diffusion launches sdkit as a child process. If the parent dies, the watchdog
shuts the server down so no orphaned process keeps the GPU busy. On Windows it opens
the parent process handle and waits with a 1-second timeout (so it can also honor the
exit flag); on POSIX it polls `kill(pid, 0)` every second. When the parent
disappears, it stops the server.

## Routes

All routes are registered in `Server::setupRoutes()`. Error responses are JSON with a
`message` field; success codes are noted per route.

| Method | Path | Purpose |
|---|---|---|
| GET | `/v1/internal/ping` | Liveness check |
| GET | `/v1/sdapi/v1/options` | Read options |
| POST | `/v1/sdapi/v1/options` | Update options |
| POST | `/v1/sdapi/v1/txt2img` | Text-to-image generation |
| POST | `/v1/sdapi/v1/img2img` | Image-to-image generation |
| POST | `/v1/internal/progress` | Poll progress for a task |
| POST | `/v1/sdapi/v1/interrupt` | Request interruption |
| POST | `/v1/sdapi/v1/extra-batch-images` | Upscale (or pass through) a batch of images |
| POST | `/v1/controlnet/detect` | Preprocess images for ControlNet |
| POST | `/v1/sdapi/v1/refresh-checkpoints` | Rescan the checkpoint directory |
| POST | `/v1/sdapi/v1/refresh-vae-and-text-encoders` | Rescan VAE and text-encoder directories |

### ping

`GET /v1/internal/ping` returns `200` with the body `OK`.

### options

`GET /v1/sdapi/v1/options` returns the full options object (JSON).

`POST /v1/sdapi/v1/options` takes a JSON body whose top-level keys are option names;
each key replaces the corresponding option, and the merged set is saved to
`options.json` immediately. Returns `200 OK`, or `400` for invalid JSON, `500` if the
save failed. See [Options persistence](#options-persistence) below.

### txt2img / img2img

Both routes share one handler path: the JSON body is mapped into an
`ImageGenerationParams` struct and passed to the `ImageGenerator`. The full parameter
list and generation flow are documented in [image-generation](image-generation.md).
Request fields relevant to the HTTP layer:

- Generation parameters: `prompt`, `negative_prompt`, `width`, `height`, `steps`,
  `cfg_scale`, `seed`, `batch_size` (maps to `batch_count`), `sampler_name`,
  `scheduler`, `lora_paths`, `lora_alphas`, `ref_images`.
- `force_task_id` (optional string) - task id used for progress polling. Defaults to
  `default_task`.
- img2img only: `init_images` (list; only the first image is used), `mask`,
  `denoising_strength` (default 0.75).
- ControlNet (optional): `alwayson_scripts.controlnet.args[0]` with `image` (base64),
  `weight`, and `model`. Webui-style model names with a hash suffix
  (`model [a3cd7cd6]`) have the suffix stripped. When a ControlNet model is present,
  width/height are rounded to the nearest multiple of 64.

Sampler and scheduler names are translated from webui (Forge/Automatic1111) style to
sd.cpp style via static mapping tables in `server.cpp`, e.g. `"Euler a"` →
`euler_a`, `"DDIM"` → `ddim_trailing`, `"automatic"`/`"uniform"` → `discrete`,
`"karras"` → `karras`, `"align_your_steps"` → `ays`.

Successful response (`200`):

```json
{
  "images": ["<base64 PNG>", "..."],
  "info": "{\"infotexts\": \"<JSON string of generation params>\"}"
}
```

Generation runs synchronously in the request thread; a second generation blocks on
the generator's mutex until the first finishes (see
[architecture](architecture.md)). On failure the route returns `500` with a
`message`, and the task is marked completed with the error as its info.

### progress

`POST /v1/internal/progress` with `{"id_task": "<task id>"}`. Returns `200` with:

```json
{
  "completed": false,
  "progress": 0.37,
  "live_preview": "<base64 JPEG or empty>",
  "id_live_preview": 3
}
```

`400` if `id_task` is missing or the JSON is invalid; `404` if the task id is
unknown. `progress` is the fraction of sampling steps completed; `live_preview` is
the latest preview frame (empty when previews are disabled);
`id_live_preview` increments each time a new preview arrives.

### interrupt

`POST /v1/sdapi/v1/interrupt` calls `ImageGenerator::interrupt()` and returns
`200 OK`. Note that in the current code the interrupt flag is set and later reset at
the start of the next generation; the flag is not consulted by the sampling loop
itself, so callers should treat interruption as best-effort right now.

### extra-batch-images

`POST /v1/sdapi/v1/extra-batch-images` with:

```json
{
  "imageList": [{"data": "<base64>"}, ...],
  "upscaler_1": "R-ESRGAN 4x+",
  "upscaling_resize": 4
}
```

If `upscaling_resize` is greater than 0, each image is upscaled with the named
RealESRGAN model via `ImageFilters::upscaleBatch` (webui upscaler names are mapped,
e.g. `R-ESRGAN 4x+` → `RealESRGAN_x4plus`), and the response is `{"images": [...]}`.
If `upscaling_resize` is 0/absent, the input base64 strings are returned unchanged.
A missing or unloaded upscaler yields `500`.

### controlnet-detect

`POST /v1/controlnet/detect` with `{"controlnet_input_images": ["<base64>", ...],
"controlnet_module": "canny"}`. Returns `{"images": [...]}` where each entry is the
preprocessed image (currently always a Canny edge map, regardless of the module
name - see [image-filters-and-upscaling](image-filters-and-upscaling.md)).

## Options persistence

`OptionsManager` (`include/options_manager.h`, `src/options_manager.cpp`) keeps an
in-memory JSON document backed by an `options.json` file in the working directory.
Defaults are created when the file is missing or empty:

- `sd_model_checkpoint` - empty until the user selects a model.
- `live_previews_enable` - true.
- `CLIP_stop_at_last_layers` - 1.
- `sdxl_clip_l_skip` - false.
- `samples_format` - `"png"`.
- `forge_additional_modules` - empty list.

Unknown keys from the file are preserved. `setOptions` (invoked by the POST options
route) merges keys and saves; the destructor also saves. The `ImageGenerator` reads
the options on every generation to decide which models to load, so changing
`sd_model_checkpoint` or `forge_additional_modules` over HTTP triggers a model
reload on the next request (see [image-generation](image-generation.md)).

`options.json` is in `.gitignore`; it is runtime state, not configuration checked
into the repo.

## Logging

crow's own log output is piped through a `FilteredLogHandler` that drops `/ping` and
`/internal/progress` lines (to keep polling from flooding the log) and forwards the
rest to sdkit's logging system with a `[CROW]` prefix. Log verbosity is controlled by
`--log-level` (`verbose`, `debug`, `info`, `warning`, `error`); see
[development](development.md) for the logging conventions.
