# Running the server

The sdkit v3 release contains a single server executable - `sdkit.exe` on Windows,
`sdkit` everywhere else. Start it from the directory where you extracted the
binaries:

```bash
./sdkit --ckpt-dir ./models/stable-diffusion --vae-dir ./models/vae
```

The server binds to `127.0.0.1` only (it is not exposed to the network) and listens
on port `8188` by default. It shuts down cleanly on `SIGINT`/`SIGTERM` (Ctrl+C).

!!! tip

    The executable looks for model files relative to the paths you pass on the
    command line, so run it from a directory where those paths make sense - or pass
    absolute paths.

## Command-line flags

All flags are parsed by `src/main.cpp` at startup. Configuration set here cannot be
changed per request - model directories and placement flags apply for the lifetime of
the process.

### Ports and logging

| Flag | Type | Default | Description |
|---|---|---|---|
| `--port <port>` | int | `8188` | Server port |
| `--log-level <level>` | string | `info` | Log verbosity: `verbose`, `debug`, `info`, `warning`, or `error` |
| `--parent-pid <pid>` | int | `0` | Parent-process PID. Enables a watchdog that shuts the server down when the parent exits - useful when sdkit is spawned as a child of a UI |

### Model directories

Each directory is scanned recursively at startup (and on the
[refresh endpoints](api/models.md#post-v1sdapiv1refresh-checkpoints)). Recognized model
extensions: `.gguf`, `.ggml`, `.safetensors`, `.ckpt`, `.pth`, `.pt`, `.sft`.
Directories that don't exist are skipped silently.

| Flag | Model type |
|---|---|
| `--ckpt-dir <path>` | Stable Diffusion checkpoints (SD 1.x/2.x, SDXL, Flux, Qwen, …) |
| `--vae-dir <path>` | VAE decoders |
| `--text-encoder-dir <path>` | Standalone text encoders (CLIP-L, CLIP-G, T5-XXL, LLM) |
| `--lora-dir <path>` | LoRA adapters |
| `--controlnet-dir <path>` | ControlNet adapters |
| `--embeddings-dir <path>` | Textual-inversion embeddings |
| `--hypernetwork-dir <path>` | Hypernetworks |
| `--realesrgan-models-path <path>` | RealESRGAN upscalers (required for [upscaling](api/filters.md#upscaling-images)) |
| `--gfpgan-models-path <path>` | GFPGAN face-restoration models (type registered; no route yet) |
| `--codeformer-models-path <path>` | CodeFormer face-restoration models (type registered; no route yet) |

### Performance and VRAM flags

These placement flags are passed through to stable-diffusion.cpp when the inference
context is created.

| Flag | Default | Description |
|---|---|---|
| `--diffusion-fa` | off | Enable diffusion flash attention |
| `--offload-to-cpu` | off | Offload model parameters to CPU |
| `--vae-tiling` | off | Enable tiled VAE decode |
| `--vae-tile-size <size>` | `256x256` | VAE tile size in pixels, format `[X]x[Y]` (a single number applies to both) |
| `--vae-on-cpu` | off | Keep the VAE on CPU |
| `--clip-on-cpu` | off | Keep CLIP (text encoder) on CPU |
| `--control-net-cpu` | off | Keep ControlNet on CPU |
| `--chroma-disable-dit-mask` | off | Disable the DiT mask for Chroma models |

## Recommended flag presets

These mirror the flags the [Easy Diffusion UI](https://github.com/easydiffusion/easydiffusion)
adds for its VRAM usage levels:

| VRAM level | Flags |
|---|---|
| Low | `--diffusion-fa --chroma-disable-dit-mask --offload-to-cpu --vae-tiling` |
| Balanced | `--diffusion-fa --chroma-disable-dit-mask --offload-to-cpu --vae-tiling --vae-tile-size 512x512` |
| High | `--diffusion-fa --chroma-disable-dit-mask` |

With `--vae-tiling` but no explicit `--vae-tile-size`, the default tile size of
`256x256` is used (the low-VRAM preset), while the balanced preset raises it to
`512x512` for faster decodes.

## Verifying the server is up

```bash
curl http://127.0.0.1:8188/v1/internal/ping
```

The server responds with `200 OK`. From there, continue with
[API conventions](api/index.md).