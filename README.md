# sdkit
**sdkit** (**s**table **d**iffusion **kit**) is an easy-to-use library for using Stable Diffusion in your AI Art projects. It is fast, feature-packed, and memory-efficient.

[![Discord Server](https://img.shields.io/discord/1014774730907209781?label=Discord)](https://discord.com/invite/u9yhsFmEkB)

*New: Stable Diffusion 2.1 is now supported!*

This is a community project, so please feel free to contribute (and use in your project)!

![t2i](https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/assets/stable-samples/txt2img/768/merged-0006.png)

# Why?
The goal is to let you be productive quickly (at your AI art project), so it bundles Stable Diffusion along with commonly-used features (like GFPGAN and CodeFormer for face restoration, RealESRGAN for upscaling, k-samplers, support for loading custom VAEs and hypernetworks, NSFW filter etc).

Advanced features include a model-downloader (with a database of commonly used models), support for running in parallel on multiple GPUs, auto-scanning for malicious models etc. [Full list of features](https://github.com/easydiffusion/sdkit/wiki/Features)

# Installation
Tested with Python 3.8. Supports Windows, Linux and Mac.

**Windows/Linux:**
1. `pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116`
2. Run `pip install sdkit`

**Mac:**
1. Run `pip install sdkit`

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

Please see the list of [examples](https://github.com/easydiffusion/sdkit/tree/main/examples), to learn how to use the other features (like filters, VAE, Hypernetworks, memory optimizations, running on multiple GPUs etc).

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
Click here to see the [list of known models](https://github.com/easydiffusion/sdkit/tree/main/sdkit/models/models_db).

sdkit includes a database of known models and their configurations. This lets you download a known model with a single line of code. You can customize where it saves the downloaded model.

Additionally, sdkit will attempt to automatically determine the configuration for a given model (when loading from disk). For e.g. if an SD 2.1 model is being loaded, sdkit will automatically know to use `fp32` for `attn_precision`. If an SD 2.0 v-type model is being loaded, sdkit will automatically know to use the `v2-inference-v.yaml` configuration. It does this by matching the quick-hash of the given model file, with the list of known quick-hashes.

For models that don't match a known hash (e.g. custom models), or if you want to override the config file, you can set the path to the config file in `context.model_paths`. e.g. `context.model_paths['stable-diffusion'] = 'path/to/config.yaml'`

# FAQ
## Does it have all the cool features?
It has a lot of features! It was born out of a popular Stable Diffusion UI, splitting out the battle-tested core engine into `sdkit`.

**Features include:** SD 2.1, txt2img, img2img, inpainting, NSFW filter, multiple GPU support, Mac Support, GFPGAN and CodeFormer (fix faces), RealESRGAN (upscale), 19 samplers (including k-samplers and UniPC), custom VAE, custom hypernetworks, low-memory optimizations, model merging, safetensor support, picklescan, etc. [Click here to see the full list of features](https://github.com/easydiffusion/sdkit/wiki/Features).

📢 We're looking to add support for *textual inversion embeddings*, *AMD support*, *ControlNet*, *Pix2Pix*, and *outpainting*. We'd love code contributions for these!

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
Yes. It works on NVIDIA/Mac GPUs with atleast 2GB of VRAM. For PCs without a compatible GPU, it can run entirely on the CPU. Running on the CPU will be *very* slow, but atleast you'll be able to try it out!

📢 We don't support AMD yet (it'll run in CPU-mode), but we're looking for code contributions for AMD support!

## Why not just use diffusers?
You can certainly use diffusers. `sdkit` is infact using `diffusers` internally (currently in beta), so you can think of `sdkit` as a convenient API and a collection of tools, focused on Stable Diffusion projects.

`sdkit`:
1. is a simple, lightweight toolkit for Stable Diffusion projects.
2. natively includes frequently-used projects like GFPGAN, CodeFormer and RealESRGAN.
3. works with the popular `.ckpt` and `.safetensors` model format.
4. includes memory optimizations for low-end GPUs.
5. built-in support for running on multiple GPUs.
6. can download models from any server.
7. auto-scans for malicious models.
8. includes 19 samplers (including k-samplers).
9. born out of the needs of the new Stable Diffusion AI Art scene, starting Aug 2022.

# Who is using sdkit?
* [Easy Diffusion (cmdr2 UI)](https://github.com/cmdr2/stable-diffusion-ui) for Stable Diffusion.

If your project is using sdkit, you can add it to this list. Please feel free to open a pull request (or let us know at our [Discord community](https://discord.com/invite/u9yhsFmEkB)).

# Contributing
We'd love to accept code contributions. Please feel free to drop by our [Discord community](https://discord.com/invite/u9yhsFmEkB)!

📢 We're looking for code contributions for these features (or anything else you'd like to work on):
- Using custom Textual Inversion embeddings.
- Outpainting.
- ControlNet.
- Pix2Pix.
- AMD support.

If you'd like to set up a developer version on your PC (to contribute code changes), please follow [these instructions](https://github.com/easydiffusion/sdkit/blob/main/CONTRIBUTING.md).

Instructions for running automated tests: [Running Tests](tests/README.md).

# Credits
* Stable Diffusion: https://github.com/Stability-AI/stablediffusion
* CodeFormer: https://github.com/sczhou/CodeFormer (license: https://github.com/sczhou/CodeFormer/blob/master/LICENSE)
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

For the full list of restrictions please read [the License](https://github.com/easydiffusion/sdkit/blob/main/LICENSE). You agree to these terms by using this software.
