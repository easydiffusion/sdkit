import torch
from torch import Tensor

from sdkit import Context
from sdkit.utils import log

from . import default_samplers, k_samplers, unipc_samplers


def make_samples(
    context: Context,
    sampler_name: str = None,
    seed: int = 42,
    batch_size: int = 1,
    shape: tuple = (),
    steps: int = 50,
    cond: Tensor = None,
    uncond: Tensor = None,
    guidance_scale: float = 0.8,
    callback=None,
    **kwargs,
):
    """
    Common args:
    * context: Context
    * sampler_name: str
    * seed: int
    * batch_size: int - number of images to generate in parallel
    * shape: tuple
    * steps: int - number of inference steps
    * cond: Tensor - conditioning from the prompt
    * uncond: Tensor - unconditional conditioning from the negative prompt
    * guidance_scale: float
    * callback: function - signature: `callback(x_samples: Tensor, i: int)`

    additional args for txt2img:

    additional args for img2img:
    * init_image_latent: Tensor
    * mask: Tensor
    * prompt_strength: float - between 0 and 1. Use 0 to ignore the prompt entirely, or 1 to ignore the init image entirely
    """
    sampler_module = None
    if sampler_name in default_samplers.samplers:
        sampler_module = default_samplers
    if sampler_name in k_samplers.samplers:
        sampler_module = k_samplers
    if sampler_name in unipc_samplers.samplers:
        sampler_module = unipc_samplers
    if sampler_module is None:
        raise RuntimeError(f'Unknown sampler "{sampler_name}"!')

    noise = make_some_noise(seed, batch_size, shape, context.device)

    return sampler_module.sample(
        context, sampler_name, noise, batch_size, shape, steps, cond, uncond, guidance_scale, callback, **kwargs
    )


def make_some_noise(seed, batch_size, shape, device):
    b1, b2, b3 = shape
    img_shape = (1, b1, b2, b3)
    tens = []
    log.info(f"seeds used = {[seed+s for s in range(batch_size)]}")

    for _ in range(batch_size):
        torch.manual_seed(seed)
        tens.append(torch.randn(img_shape, device=device))
        seed += 1
    noise = torch.cat(tens)
    del tens

    return noise
