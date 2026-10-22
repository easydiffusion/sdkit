# Notes & troubleshooting

## Common errors

| Error | Cause | Fix |
|---|---|---|
| `"No model selected"` | `sd_model_checkpoint` option is empty | Set it via `POST /v1/sdapi/v1/options` (see [Models & options](api/models.md)) |
| `"Model not found"` | Checkpoint name doesn't resolve to a file in `--ckpt-dir` | Check the name (path relative to `--ckpt-dir`, **with** extension, e.g. `mymodel.safetensors`). If you added the file after startup, call `POST /v1/sdapi/v1/refresh-checkpoints` |
| `"Upscaler not available. Please configure an upscaler model in options."` | No RealESRGAN model loaded for [upscaling](api/filters.md#upscaling-images) | Start the server with `--realesrgan-models-path <dir>` and make sure the model file exists (e.g. `RealESRGAN_x4plus.pth`), matching the upscaler name |
| `"Invalid JSON"` (HTTP 400) | Request body isn't valid JSON | Check the request body and `Content-Type: application/json` header |
| `"Task not found"` (HTTP 404) | Unknown task ID passed to `/v1/internal/progress` | Use the same `force_task_id` you sent with the generation request |

## Debugging

- Increase log verbosity with `--log-level debug` (or `verbose`) when
  [starting the server](server.md#ports-and-logging). Model scanning, ControlNet
  parameter logging, and generation errors all get logged.
- The [modelinfo tool](https://github.com/easydiffusion/sdkit) (built with
  `SDKIT_BUILD_TOOLS=ON`) can tell you how a model file is classified - useful when
  a text encoder or VAE is being routed to the wrong slot.

## Out of memory

If generation crashes or fails with allocation errors, reduce VRAM pressure with the
flags from [Running the server](server.md#performance-and-vram-flags):

- Start with the [low-VRAM preset](server.md#recommended-flag-presets):
  `--offload-to-cpu --vae-tiling`.
- Enable `--vae-on-cpu`, `--clip-on-cpu`, or `--control-net-cpu` to keep individual
  components off the GPU.
- Lower `width`/`height` and `batch_size` in generation requests.
- On CUDA, use the right build variant for your GPU (e.g. `sm120` for newer cards).

## Where to get help

- **Bugs and feature requests** -
  [GitHub issues](https://github.com/easydiffusion/sdkit/issues)
- **Questions and community help** - the
  [Easy Diffusion Discord](https://discord.com/invite/u9yhsFmEkB)
- **Commercial integration help** - contact **cmdr2** (project maintainer) via
  GitHub or Discord.