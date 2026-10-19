# Image Generation

This page covers the text-to-image and image-to-image pipelines end to end: how a
generation request turns into a stable-diffusion.cpp call, how models are selected
and (re)loaded, and how progress and previews reach the client. For the routes and
their JSON shapes see [server-and-api](server-and-api.md); for the model files
themselves see [models-and-detection](models-and-detection.md).

## ImageGenerationParams

`include/image_generator.h` defines the parameters struct that the server fills from
the request JSON:

```cpp
struct ImageGenerationParams {
    std::string prompt;
    std::string negative_prompt;
    std::vector<std::string> lora_paths;
    std::vector<float> lora_alphas;
    int width = 512, height = 512;
    int steps = 20;
    float cfg_scale = 7.0f;
    int64_t seed = -1;
    int batch_count = 1;
    sample_method_t sampler = EULER_A_SAMPLE_METHOD;
    scheduler_t scheduler = DISCRETE_SCHEDULER;

    // img2img
    std::string init_image_base64, mask_base64;
    float strength = 0.75f;

    // ControlNet
    std::string control_image_base64;
    float control_strength = 1.0f;
    std::string controlnet_model;

    // Vision-based models (Qwen, etc.)
    std::vector<std::string> ref_images_base64;
    bool auto_resize_ref_image = true;

    int clip_skip = -1;
};
```

A `batch_size` field is also declared but the HTTP layer maps the request's
`batch_size` into `batch_count` (the number of images generated in one sd.cpp call);
the generator itself always runs one call per request. Sampler and scheduler arrive
as webui-style names and are translated by `Server::generateImage` (see
[server-and-api](server-and-api.md)) before they reach the generator.

## ImageGenerator overview

`ImageGenerator` (`src/image_generator.cpp`) is the component that owns the
stable-diffusion context (`sd_ctx_t*`) and drives every generation. It is constructed
once per server with the task-state manager, options manager, model manager, and
image filters, plus the `ServerParams` placement flags (VAE on CPU, tiling, offload
to CPU, flash attention, ControlNet/CLIP on CPU, disable DiT mask).

All of its work is serialized behind its own mutex: a generation request first loads
the model if needed, then locks the generator so only one inference runs at a time on
the shared context. Progress and live previews are reported out-of-band via global
callback state, so the HTTP progress endpoint never has to touch the generator's
lock.

## The generation call path

`generateTxt2Img` and `generateImg2Img` are thin wrappers around one internal
function, `generateInternal(params, is_img2img, task_id)`:

1. **Model load** - `ensureModelLoaded(params.controlnet_model)` is called *before*
   taking the generator's lock, so a model switch happens without blocking other
   requests longer than necessary (see [Model loading](#model-loading) below).
2. **Lock and task setup** - the generator's mutex is taken, `interrupted_` is
   cleared, and the global `CallbackData` struct is pointed at this task so the
   sd.cpp callbacks can report progress to the right `TaskStateManager` entry.
3. **Callback registration** - `sd_set_progress_callback` is always installed. The
   preview callback (`sd_set_preview_callback` with `PREVIEW_PROJ`) is only
   installed when the `live_previews_enable` option is `true` (the default); it is
   cleared otherwise.
4. **LoRA list** - `lora_paths` and `lora_alphas` are zipped into `sd_lora_t`
   entries (`is_high_noise = false`). If the alpha list is shorter than the path
   list, a warning is logged and the extra paths are ignored.
5. **Parameter mapping** - a `sd_img_gen_params_t` is filled:
   - prompt, negative prompt (empty string if none), `clip_skip`
   - width/height/batch_count from params
   - seed: if the client passed `-1` (or omitted it), the wall-clock time is used;
     otherwise the client's seed is honored, so results are reproducible.
   - `sample_params.sample_method` / `sample_params.scheduler` /
     `sample_params.sample_steps` / `sample_params.guidance.txt_cfg` (CFG)
6. **img2img images** - `createInitImage` and `createMaskImage` decode, resize, and
   fill `gen_params.init_image`, `gen_params.mask_image`, and `strength`.
7. **ControlNet** - if both `control_image_base64` and `controlnet_model` are set,
   `createControlImage` decodes, resizes, and preprocesses (Canny) the image, then
   fills `gen_params.control_image` and `control_strength`.
8. **Reference images** - `createRefImages` decodes any `ref_images_base64` entries
   (used by vision-capable models such as Qwen); optionally resizes them to the
   generation size when `auto_resize_ref_image` is set.
9. **VAE tiling** - if the `--vae-tiling` CLI flag was given, the parsed
   `vae_tile_size` string (either `"WxH"` or a single number for both) is applied to
   `gen_params.vae_tiling_params` (default 256x256, `target_overlap` 0.5).
10. **Inference** - `generate_image(sd_ctx_, &gen_params, &result, &num_results)`
    runs the actual sampling. A false return is treated as "no images".
11. **Cleanup and result encoding** - all temporary images (init, mask, control,
    refs) are freed, each result is encoded to base64 PNG via `imageToBase64`, and
    the result array is freed. The callback's task ID is cleared, and the vector of
    base64 strings is returned to the server handler, which builds the JSON response
    and completes the task state.

`createInitImage` decodes the base64 init image and resizes it to the generation
size (with the aspect-preserving resize helper from `image_utils`). `createMaskImage`
produces either a fully white (255) single-channel mask when no mask was supplied, or
the decoded and resized user mask. `createControlImage` decodes and resizes, then
delegates preprocessing to `image_filters_->applyControlNetFilter(control_image,
"canny")` - if preprocessing fails it logs a warning and falls back to the unfiltered
control image.

## Model loading

`ensureModelLoaded(controlnet_model)` decides whether the currently loaded
stable-diffusion context is still valid, and (re)creates it when needed. It is the
piece that connects the generator to [models-and-detection](models-and-detection.md) and to
`OptionsManager` persistence.

### How the checkpoint is picked

1. The current options are read. `sd_model_checkpoint` (see
   [server-and-api](server-and-api.md)) holds the *lookup name* of the checkpoint -
   the relative path with extension, as defined in
   [models-and-detection](models-and-detection.md) for `CHECKPOINT` models.
2. `ModelManager::getModelByName(name, ModelType::CHECKPOINT)` resolves it to an
   absolute path. An empty option or a name that does not resolve is an error: the
   generator logs `No model selected` / `Model not found` and fails the request.
3. If `controlnet_model` was passed, the same lookup runs against `CONTROLNET` models;
   a miss is a warning (generation proceeds without ControlNet) rather than an error.

### Additional modules

The `forge_additional_modules` option (a list of file paths) provides standalone
VAE and text encoders. Rather than trusting filenames, the generator runs
`inspectModelType` (see [models-and-detection](models-and-detection.md)) on each file and
routes it into the matching parameter slot based on what the file actually is:

- `"vae"` -> `params.vae_path`
- `"clip_l"` -> `params.clip_l_path`
- `"clip_g"` -> `params.clip_g_path`
- `"t5xxl"` -> `params.t5xxl_path`
- `"llm"` -> `params.llm_path`

Anything unrecognized is logged as a warning and skipped. If at least one of these
modules is present, the generator loads the checkpoint's diffusion weights separately
(`params.diffusion_model_path = model_path; params.model_path = nullptr`) instead of
treating the file as a complete checkpoint - this is how models whose text encoders
live in separate files are supported.

### Reload logic

The generator tracks the currently loaded paths in member fields
(`current_model_path_`, `current_vae_path_`, `current_clip_l_path_`,
`current_clip_g_path_`, `current_t5xxl_path_`, `current_llm_path_`,
`current_controlnet_path_`, `current_embeddings_dir_`, ...). A reload is needed when
the context is missing or when *any* of these differs from what the current options
and request ask for. If nothing changed, the existing context is reused - this is
what makes repeat generations on the same model fast.

On a reload, the old context is freed, and a fresh `sd_ctx_params_t` is built:

- embeddings are discovered by scanning the embeddings directory recursively for
  `.gguf` / `.safetensors` / `.pt` / `.sft` files and mapping name -> path, then
  passed as `params.embeddings` / `params.embedding_count` so that prompts can
  reference them by name.
- the LoRA directory is noted (actual LoRA files are attached per-request in the
  `gen_params.loras` list, not at context load).
- the RNG type is fixed to `CUDA_RNG`.
- the placement flags from `ServerParams` are turned into backend assignment strings
  (`te=cpu` for CLIP, `vae=cpu` for VAE, `controlnet=cpu` for ControlNet, `*=cpu` for
  full offload) that stable-diffusion.cpp interprets to pin those submodules to the
  CPU; `diffusion_flash_attn` is set from the `--diffusion-fa` flag.
- `new_sd_ctx(&params)` creates the context; on success `initialized_` is set and
  the current paths are recorded.

Note that `ensureModelLoaded` takes the generator's mutex itself, which is why
`generateInternal` calls it *before* taking that mutex (a nested lock would
deadlock).

## Progress and live previews

Two global callbacks in `src/image_generator.cpp` bridge sd.cpp's C callbacks into
`TaskStateManager` (see [architecture](architecture.md)):

- **`progress_callback(step, steps, time, data)`** - fires after each sampling step
  and calls `updateTaskProgress(task_id, step/steps)`. The progress endpoint reads
  this to show a percentage.
- **`preview_callback(step, frame_count, frames, is_noisy, data)`** - fires when the
  model produces a preview frame. The first frame is JPEG-encoded in-memory
  (`stbi_write_jpg_to_func`, quality 75) and base64-encoded, then passed to
  `updateTaskProgress(task_id, progress, preview_base64)`. The client renders this
  `live_preview` string while generation is running.

Both are only active while a task ID is registered in the global callback data, which
is set at the start of `generateInternal` and cleared when generation ends (success
or failure). This is why an interrupt (which only sets a flag on the generator) stops
the current run between steps, while progress polling continues to work from a
separate HTTP request.

## Interrupts

`ImageGenerator::interrupt()` sets `interrupted_` under the mutex. The sd.cpp
generation loop checks this flag via the progress callback mechanism; a run stops at
the next step boundary and returns whatever was produced so far. The task state is
then completed normally with the (partial) result set.

## Seed behavior

- `seed == -1` or absent: the current wall-clock time is used, so every generation is
  different unless the client sends a concrete seed.
- Any other value: passed through unchanged, so the same seed + prompt + steps
  reproduces the same image.

## What a failure looks like

`generateInternal` throws `std::runtime_error` in these cases, which the server
handler converts into a `500` response with a message:

- `ensureModelLoaded` failed (no checkpoint configured, checkpoint not found, or
  context creation failed).
- Decoding or resizing of an init image, mask, control image, or reference image
  failed.
- `generate_image` returned no results.

When generation throws, the task state is completed with the error message so the
client sees the failure through the progress endpoint too.
