# sdkit v3

**sdkit** (**s**table **d**iffusion **kit**) v3 is a Stable Diffusion server built on
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp). It exposes an
HTTP API that follows the conventions of the Automatic1111 / Forge webui
(`/v1/sdapi/v1/...`), so existing clients and integrations work with minimal changes.

The server runs locally (bound to `127.0.0.1`), loads models from directories you
configure on the command line, and performs:

- **txt2img** - text-to-image generation
- **img2img** - image-to-image generation, including masks and reference images
- **Upscaling** - RealESRGAN upscaling of one or more images
- **ControlNet** - conditioning image preprocessing (Canny) and ControlNet-guided
  generation

Model weights are consumed as GGUF or safetensors files, and the inference core runs
entirely in C++ with CPU, CUDA, Vulkan, and Metal backends.

!!! warning "Work in progress"

    v3 is not ready for production and the API can change at any time.

    - Report bugs and feature requests on
      [GitHub issues](https://github.com/easydiffusion/sdkit/issues).
    - Ask questions and get help on the
      [Easy Diffusion Discord](https://discord.gg/Z5q3HCAqzF).

## What's in this documentation

| Page | Description |
|---|---|
| [Installation](installation.md) | How to download prebuilt binaries with the release-manifest flow, and how to build from source. |
| [Running the server](server.md) | Every command-line flag accepted by the `sdkit` server, plus recommended presets and how to verify the server is up. |
| [API conventions](api/index.md) | Base URL, JSON conventions, base64 images, error format, task IDs, and blocking behavior. |
| [Models & options](api/models.md) | Reading and setting server options (`sd_model_checkpoint` etc.) and rescanning model directories at runtime. |
| [Image generation](api/generate.md) | Full parameter reference for `txt2img` and `img2img`, sampler/scheduler name mapping, progress polling, and interruption. |
| [Filters, upscaling & ControlNet](api/filters.md) | Upscaling images with RealESRGAN, generating ControlNet conditioning images, and passing ControlNet inside generation requests. |
| [Notes & troubleshooting](appendix.md) | Common errors and fixes, debugging tips, and where to get help. |

## Quick start

1. [Download the binaries](installation.md) and start the server (see
   [Running the server](server.md)).
2. Select a model:

    ```bash
    curl -X POST http://127.0.0.1:8188/v1/sdapi/v1/options \
         -H "Content-Type: application/json" \
         -d '{"sd_model_checkpoint": "model.safetensors"}'
    ```

3. Generate an image:

    ```bash
    curl -X POST http://127.0.0.1:8188/v1/sdapi/v1/txt2img \
         -H "Content-Type: application/json" \
         -d '{"prompt": "a photograph of an astronaut riding a horse", "steps": 20}'
    ```

See [API conventions](api/index.md) for the response shape and
[Image generation](api/generate.md) for the full parameter list.