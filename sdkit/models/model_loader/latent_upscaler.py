import torch
from diffusers import StableDiffusionLatentUpscalePipeline

from sdkit import Context
from sdkit.utils import log



def load_model(context: Context, **kwargs):
    log.info('Load latent upscale model')
    dtype = torch.float16 if context.half_precision else torch.float32
    upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained("stabilityai/sd-x2-latent-upscaler", torch_dtype=dtype)
    upscaler.to(context.device)

    if context._vram_usage_level != "high":
        #upscaler.enable_sequential_cpu_offload()   # GPU?
        upscaler.enable_attention_slicing()   # GPU?
    log.info('Loading completed')

    return upscaler


def unload_model(context: Context, **kwargs):
    pass
