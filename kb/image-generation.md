# Image generation

`ImageGenerator` (`include/image_generator.h`, `src/image_generator.cpp`) owns the
stable-diffusion.cpp context and implements both generation flows. This page walks
through the flows, the parameters, and how progress and interruption work. The HTTP
side is covered in [server-and-api](server-and-api.md).

## ImageGenerationParams

All generation inputs are collected in `ImageGenerationParams` before being handed
to the generator:

| Field | Default | Notes |
|---|---|---|
| `prompt`, `negative_prompt` | empty | Raw text passed to the engine |
| `width`, `height` | 512 | Rounded to multiples of 64 by the server when ControlNet is used |
| `steps` | 20 | Sampling steps |
| `cfg_scale` | 7.0 | Text guidance |
| `seed` | -1 | Negative seeds are replaced with the current Unix time at generation time |
| `batch_count`, `batch_size` | 1 | `batch_size` from the HTTP request maps to `batch_count` |
| `sampler`, `scheduler` | EULER_A / DISCRETE | `sample_method_t`/`scheduler_t` enums after name conversion |
| `lora_paths`, `lora_alphas` | empty | Parallel lists; see LoRA section |
| `init_image_base64`, `mask_base64`, `strength` | empty / 0.75 | img2img only |
| `control_image_base64`, `control_strength`, `controlnet_model` | empty / 1.0 | ControlNet only |
| `ref_images_base64`, `auto_resize_ref_image` | empty / true | Reference images for vision-based models (Qwen etc.) |
| `clip_skip` | -1 | Passed through to the engine |

## txt2img flow

`generateTxt2Img` calls `generateInternal(params, false, task_id)`:

1. **Ensure the model is loaded.** `ensureModelLoaded(controlnet_model)` runs before
   the generation lock is taken. It reads `sd_model_checkpoint` from the options,
   resolves it to a full path via `ModelManager`, resolves the ControlNet model if
   one was requested, and inspects every entry in `forge_additional_modules` to
   classify it (VAE, CLIP-L, CLIP-G, T5-XXL, or LLM) using
   [model detection](models-and-detection.md). If any resolved path differs from
   what is currently loaded, the old `sd_ctx_t` is freed and a new one is created
   with `new_sd_ctx`. VAE/CLIP/T5/LLM/ControlNet paths are passed separately to the
   context; embeddings are collected recursively from the embeddings directory
   (`.gguf`, `.safetensors`, `.pt`, `.sft`) and passed as a name→path list. Startup
   flags (`--vae-on-cpu`, `--offload-to-cpu`, etc.) are translated into backend
   assignments and context parameters here (e.g. `te=cpu`, `vae=cpu`,
   `controlnet=cpu`, `*=cpu` for parameter offload), along with
   `diffusion_fa` (flash attention) and `chroma_disable_dit_mask`.
2. **Take the generation mutex.** Only one generation runs at a time;
   `interrupted_` is reset and `current_task_id_` is recorded.
3. **Register callbacks.** A global callback structure (guarded by its own mutex)
   records the task id and total steps. `sd_set_progress_callback` reports step
   progress; `sd_set_preview_callback(preview_callback, PREVIEW_PROJ, 3, true,
   false, nullptr)` is installed only when the `live_previews_enable` option is on.
4. **Build LoRA list.** `lora_paths` and `lora_alphas` are zipped into
   `sd_lora_t` entries (`is_high_noise` is always false; a warning is logged and the
   extra paths dropped if the alpha list is shorter).
5. **Fill `sd_img_gen_params_t`.** Initialized with `sd_img_gen_params_init`, then:
   prompt/negative prompt, clip skip, sizes, batch count, seed, sampler/scheduler
   from the params, and `guidance.txt_cfg = cfg_scale`.
6. **Run the engine.** `generate_image(sd_ctx_, &gen_params, &result, &num_images)`
   blocks until all images are done. The engine's progress callback updates the task
   state each step, and the preview callback encodes the first frame as a
   quality-75 JPEG, base64-encodes it, and stores it in the task state with an
   incremented preview id.
7. **Convert and clean up.** Init/mask/control/ref images are freed, results are
   converted to base64 PNGs via `imageToBase64`, the result array is freed, and the
   global callback data is cleared. Failures throw `std::runtime_error`, which the
   server turns into HTTP 500.

## img2img additions

When `init_image_base64` is present:

- **Init image**: decoded from base64 (data URI prefixes are stripped by
  `base64ToImage`), resized to the request dimensions (clamped to multiples of 8),
  and set as `gen_params.init_image` with `gen_params.strength` as the denoising
  strength.
- **Mask**: if `mask_base64` is empty, a fully white (255) 1-channel mask of the
  target size is allocated, meaning no masking. Otherwise the mask is decoded with
  1 channel and resized to match.

## ControlNet

ControlNet is used when both `control_image_base64` and `controlnet_model` are set
(the server extracts these from the `alwayson_scripts.controlnet` block). The
control image is decoded, resized to the request dimensions, and preprocessed by
`ImageFilters::applyControlNetFilter` (currently always Canny; if preprocessing
fails, the unprocessed image is used with a warning). The model itself is attached
at context creation time by `ensureModelLoaded`, and `control_strength` is set on
the generation params. The server also rounds dimensions to multiples of 64 when a
ControlNet model is requested.

## Reference images

For vision-capable models (Qwen-style), `ref_images_base64` holds one or more base64
images. Each is decoded; with `auto_resize_ref_image` (default true) they are
resized to the request dimensions, and the engine receives
`ref_image_args = "resize_before_vae=true"`. Empty entries are skipped with a
warning.

## VAE handling

There are two paths:

- A separate VAE listed in `forge_additional_modules` (classified as `vae` by model
  detection) is loaded as the context's `vae_path` in `ensureModelLoaded`.
- VAE runtime behavior (tiling and CPU placement) is fixed at startup: `--vae-tiling`
  enables tiled decoding with tile sizes parsed from `--vae-tile-size` (default
  256x256 pixels, overlap 0.5), and `--vae-on-cpu` adds a `vae=cpu` backend
  assignment. These are startup-only options; see [server-and-api](server-and-api.md).

## Progress and interruption via TaskState

`TaskStateManager` (`include/task_state.h`, `src/task_state.cpp`) keeps a mutex-
guarded map of `TaskState` records:

- `createTask` resets a task (progress 0, not completed, not interrupted).
- `updateTaskProgress(progress, live_preview)` is called by the engine callbacks;
  a non-empty preview bumps `id_live_preview`.
- `completeTask(images, info)` marks the task done and stores the result images and
  an info JSON string.
- `interruptTask` marks the interrupted flag; `clearTask` removes a task.
  In the current tree neither is called by the server or the generator, and the
  generator's own `interrupted_` flag is set by `interrupt()` but not yet consulted
  by the sampling loop - interruption is best-effort (see
  [server-and-api](server-and-api.md)).

The server's `/v1/internal/progress` route reads a task by id and returns progress,
the preview, and the preview counter.

One caveat: because the engine callbacks capture the current task id in a
process-global structure, only one generation's progress is meaningful at a time -
which is consistent with the generator mutex serializing generations.

## Memory ownership

`sd_image_t` buffers come from the engine or stb loaders and must be freed with
`freeImage` (or `free`/`stbi_image_free` for result arrays). The generator carefully
frees init, mask, control, ref, and result images on every path, including error
paths. `image_utils.h` documents the conversion helpers used throughout.
