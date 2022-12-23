# sdkit
**sdkit** (**s**table **d**iffusion **kit**) is an easy-to-use library for using Stable Diffusion in your AI Art projects. It is fast, feature-packed, and memory-efficient.

[![Discord Server](https://img.shields.io/discord/1014774730907209781?label=Discord)](https://discord.com/invite/u9yhsFmEkB)

*New: Stable Diffusion 2.1 is now supported!*

This is a community project, so please feel free to contribute (and use in your project)!

![t2i](https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/assets/stable-samples/txt2img/768/merged-0006.png)

# Why?
The goal is to let you be productive quickly (at your AI art project), so it bundles Stable Diffusion along with commonly-used features (like GFPGAN for face restoration, RealESRGAN for upscaling, k-samplers, support for loading custom VAEs and hypernetworks etc).

Advanced features include a model-downloader (with a database of commonly used models), support for running in parallel on multiple GPUs, auto-scanning for malicious models etc.

# Installation
1. On Windows and Linux only: `pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116`
2. Run `pip install sdkit`

We don't support Mac yet, but we'd love to accept code contributions!

# Example
### Local model
A simple example for generating an image from a Stable Diffusion model file (already present on the disk):
```python
import sdkit
from sdkit.models import load_model
from sdkit.generate import generate_images
from sdkit.utils import save_images, log

context = sdkit.Context()

# set the path to the model file on the disk (.ckpt or .safetensors file)
context.model_paths['stable-diffusion'] = 'D:\\path\\to\\512-base-ema.ckpt'
load_model(context, 'stable-diffusion')

# generate the image
images = generate_images(context, prompt='Photograph of an astronaut riding a horse', seed=42, width=512, height=512)

# save the image
save_images(images, dir_path='D:\\path\\to\\images\\directory')

log.info("Generated images!")
```

### Auto-download a known model
A simple example for automatically downloading a known Stable Diffusion model file:
```python
import sdkit
from sdkit.models import download_models, resolve_downloaded_model_path, load_model
from sdkit.generate import generate_images
from sdkit.utils import save_images

context = sdkit.Context()

download_models(context, models={'stable-diffusion': '1.5-pruned-emaonly'}) # downloads the known "SD 1.5-pruned-emaonly" model

context.model_paths['stable-diffusion'] = resolve_downloaded_model_path(context, 'stable-diffusion', '1.5-pruned-emaonly')
load_model(context, 'stable-diffusion')

images = generate_images(context, prompt='Photograph of an astronaut riding a horse', seed=42, width=512, height=512)
save_images(images, dir_path='D:\\path\\to\\images\\directory')
```

Please see the list of [examples](examples), to learn how to use the other features (like filters, VAE, Hypernetworks, memory optimizations, running on multiple GPUs etc).

# API
Please see the [API Reference](https://github.com/easydiffusion/sdkit/wiki/API) page for a detailed summary.

Broadly, the API contains 5 modules:
```python
sdkit.models # load/unloading models, downloading known models, scanning models
sdkit.generate # generating images
sdkit.filter # face restoration, upscaling
sdkit.train # model merge, and (in the future) more training methods
sdkit.utils
```

And a `sdkit.Context` object is passed around, which encapsulates the data related to the runtime (e.g. `device` and `vram_optimizations`) as well as references to the loaded model files and paths. `Context` is a thread-local object.

# Models DB
Click here to see the [list of known models](sdkit/models/models_db).

sdkit includes a database of known models and their configurations. This lets you download a known model with a single line of code. You can customize where it saves the downloaded model.

Additionally, sdkit will attempt to automatically determine the configuration for a given model (when loading from disk). For e.g. if an SD 2.1 model is being loaded, sdkit will automatically know to use `fp32` for `attn_precision`. If an SD 2.0 v-type model is being loaded, sdkit will automatically know to use the `v2-inference-v.yaml` configuration. It does this by matching the quick-hash of the given model file, with the list of known quick-hashes.

For models that don't match a known hash (e.g. custom models), or to override the config file, you can set the path to the config file in `context.model_paths`. e.g. `context.model_paths['stable-diffusion'] = 'path/to/config.yaml'`

# FAQ
## Does it have all the cool features?
It has a lot of features! It was born out of a popular Stable Diffusion UI, splitting out the battle-tested core engine into `sdkit`.

**Features include:** SD 2.1, txt2img, img2img, inpainting, multiple GPU support, gfpgan (fix faces), realesrgan (upscale), 14 samplers (including k-samplers), custom VAE, custom hypernetworks, low-memory optimizations, model merging, safetensor support, picklescan, etc. [Click here to see all the features](Features)

📢 We're looking to add support for *textual inversion embeddings*, *codeformer*, *seamless tiling*, and *outpainting*. We'd love code contributions for these!

## Is it fast?
It is pretty fast, and close to the fastest. For the same image, `sdkit` took 5.5 seconds, while `automatic1111` webui took 4.95 seconds. 📢 We're looking for code contributions to make `sdkit` even faster!

`xformers` is supported experimentally, which will make `sdkit` even faster.

**Details of the benchmark:**

Windows 11, NVIDIA 3060 12GB, 512x512 image, sd-v1-4.ckpt, euler_a sampler, number of steps: 25, seed 42, guidance scale 7.5.

No xformers. No VRAM optimizations for low-memory usage.

| | Time taken | Iterations/sec | Peak VRAM usage |
| --- | --- | --- | --- |
| `sdkit` | 5.5 sec | 6.0 it/s | 5.1 GB |
| `automatic1111` webui | 4.95 sec | 6.15 it/s | 5.1 GB |

## Does it work on lower-end GPUs, or without GPUs?
Yes. It works on NVIDIA GPUs with atleast 3GB of VRAM. Otherwise, it can run entirely on the CPU, for PCs without a compatible GPU. Running on the CPU will be *very* slow, but it'll work.

📢 We don't support AMD yet (it'll run in CPU-mode), but we're looking for code contributions for AMD support!

## Why not just use diffusers?
You can certainly use diffusers. `sdkit` is just a different attempt at a productive toolkit, so use `sdkit` if you find its features useful.

`sdkit`:
1. is a fresh attempt at a simple, lightweight toolkit for Stable Diffusion projects.
2. natively includes frequently-used projects like GFPGAN and RealESRGAN
3. works with the popular `.ckpt` and `.safetensors` model format (instead of only the diffusers format)
4. includes memory optimizations for low-end GPUs
5. built-in support for running on multiple GPUs
6. can download models from any server
7. auto-scans for malicious models
8. includes 14 samplers (including k-samplers)
9. born out of the needs of the new Stable Diffusion AI Art scene, starting Aug 2022

This is not to say that `diffusers` can't do these. The easy-to-use API of `diffusers` is an inspiration for `sdkit`.

# Who is using sdkit?
* [cmdr2 UI](https://github.com/cmdr2/stable-diffusion-ui) for Stable Diffusion.

If your project is using sdkit, you can add it to this list. Please feel free to open a pull request (or let us know at our [Discord community](https://discord.com/invite/u9yhsFmEkB)).

# Contributing
We'd love to accept code contributions. Please feel free to drop by our [Discord community](https://discord.com/invite/u9yhsFmEkB)!

📢 We're looking for code contributions for these features (or anything else you'd like to work on):
- CodeFormer upscaling (please maintain the required copyright notices)
- Using custom Textual Inversion embeddings
- Seamless tiling
- Outpainting
- Mac support
- AMD support
- Allow other samplers for img2img (instead of only DDIM)

# Credits
* Stable Diffusion: https://github.com/Stability-AI/stablediffusion
* GFPGAN: https://github.com/TencentARC/GFPGAN
* RealESRGAN: https://github.com/xinntao/Real-ESRGAN
* k-diffusion: https://github.com/crowsonkb/k-diffusion
* Code contributors and artists on the cmdr2 UI: https://github.com/cmdr2/stable-diffusion-ui and Discord (https://discord.com/invite/u9yhsFmEkB)
* Lots of contributors on the internet

# Disclaimer
The authors of this project are not responsible for any content generated using this project.

The license of this software forbids you from sharing any content that:
- Violates any laws.
- Produces any harm to a person or persons.
- Disseminates (spreads) any personal information that would be meant for harm.
- Spreads misinformation.
- Target vulnerable groups. 

For the full list of restrictions please read [the License](LICENSE). You agree to these terms by using this software.
