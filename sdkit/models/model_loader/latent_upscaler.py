import torch
from diffusers import StableDiffusionLatentUpscalePipeline

from sdkit import Context
from sdkit.utils import log


def load_model(context: Context, **kwargs):
    dtype = torch.float16 if context.half_precision else torch.float32
    upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(
        "stabilityai/sd-x2-latent-upscaler", torch_dtype=dtype, revision="1f2883b22f62cfb29323770ba9cfdcf601757bd3"
    )
    if context.vram_usage_level == "high":
        upscaler.to(context.device)
    else:
        if "cuda" in context.device:
            upscaler.enable_sequential_cpu_offload()

        upscaler.enable_attention_slicing(1)

    return upscaler


def unload_model(context: Context, **kwargs):
    pass
