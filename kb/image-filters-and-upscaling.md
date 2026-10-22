# Image filters and upscaling

`ImageFilters` (`include/image_filters.h`, `src/image_filters.cpp`) implements the
non-diffusion image operations: upscaling with RealESRGAN models and preprocessing
for ControlNet. It is created by the `Server` at startup and shared with the
`ImageGenerator` (which uses it for ControlNet preprocessing during generation). The
HTTP entry points are documented in [server-and-api](server-and-api.md).

## Upscaling

### upscaleBatch

`upscaleBatch(base64_images, upscaler_name, upscale_factor)` is the batch entry point
used by the `/v1/sdapi/v1/extra-batch-images` route:

1. `ensureUpscalerLoaded(upscaler_name)` makes sure the requested RealESRGAN model is
   loaded (see below). If it fails, the method logs an error and returns an empty
   list; the server turns that into an HTTP 500 with an "upscaler not available"
   message.
2. Each image is decoded from base64 (`base64ToImage`), upscaled
   (`upscaleImage`), converted back to base64 PNG (`imageToBase64`), and freed.
3. Images that fail to decode or upscale produce an empty string at that position in
   the result list, so positions are preserved.

### ensureUpscalerLoaded and upscaler tracking

The class tracks its current upscaler in two members:

- `upscaler_ctx_` - the engine's upscaler context (`upscaler_ctx_t*`), freed in the
  destructor.
- `current_upscaler_path_` - the name of the currently loaded upscaler.

`ensureUpscalerLoaded` behaves as a lazy loader:

- It requires a non-empty upscaler name.
- If the same name is already loaded, it returns immediately (no reload).
- Otherwise it resolves the RealESRGAN directory from the `ModelManager`
  (`getRealesrganModelsPath()`, set via `--realesrgan-models-path`) and builds
  `<dir>/<name>.pth`. An unset directory throws `std::runtime_error`.
- It creates the context with `new_upscaler_ctx(path, false, -1, 128, nullptr,
  nullptr)` and records the name. Switching upscalers frees the old context first.

So one upscaler is loaded at a time, and switching models costs one model load. The
upscaler name arrives from the HTTP route, where webui-style names are translated
(e.g. `R-ESRGAN 4x+` → `RealESRGAN_x4plus`, `R-ESRGAN 4x+ Anime6B` →
`RealESRGAN_x4plus_anime_6B`); see [server-and-api](server-and-api.md).

### upscaleImage

`upscaleImage(input_image, factor)` calls the engine's `upscale()` function, takes
the first result image, frees the rest of the result array, and returns the upscaled
`sd_image_t`. An invalid input image or a missing upscaler context returns an empty
image.

## ControlNet preprocessing

### applyControlNetFilterBatch

`applyControlNetFilterBatch(base64_images, module)` is the batch entry point used by
the `/v1/controlnet/detect` route. It decodes each image, applies
`applyControlNetFilter`, and re-encodes the results to base64, preserving positions
(with empty strings for failures).

### The Canny implementation

`applyControlNetFilter(input_image, module)` copies the input image data into a new
buffer and calls `preprocess_canny(result_image, 0.08f, 0.08f, 0.8f, 1.0f, false)` -
the Canny edge detector provided by stable-diffusion.cpp itself.

Important detail: the `module` argument is currently **not** dispatched. Whatever
module name the client sends (`canny`, or anything else), the filter always runs the
Canny preprocessing. If Canny fails, the original (unprocessed) image is returned
with a warning, so downstream behavior degrades gracefully rather than erroring.

There is no separate Canny implementation inside sdkit (for example, there is no
`src/filters/` directory); the preprocessing lives in the engine submodule. The same
code path is used internally by `ImageGenerator::createControlImage` during
generation, so ControlNet conditioning and the `/v1/controlnet/detect` route produce
the same kind of control image (see [image-generation](image-generation.md)).

## Concurrency notes

`ImageFilters` has no mutex of its own. It is called from the server's multithreaded
request handlers, so two concurrent upscaling requests would race on
`upscaler_ctx_`/`current_upscaler_path_`. In practice this is shielded by the
`ImageGenerator`'s generation mutex for generation-time preprocessing, but the
standalone routes (`extra-batch-images`, `controlnet/detect`) rely on callers not
hammering them concurrently - worth keeping in mind when changing this code.
