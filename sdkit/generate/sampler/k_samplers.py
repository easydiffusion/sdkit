import k_diffusion.external
import k_diffusion.sampling as k_samplers
import torch
import torch.nn as nn
from torch import Tensor

from sdkit import Context

samplers = {
    "euler_a": k_samplers.sample_euler_ancestral,
    "euler": k_samplers.sample_euler,
    "lms": k_samplers.sample_lms,
    "heun": k_samplers.sample_heun,
    "dpm2": k_samplers.sample_dpm_2,
    "dpm2_a": k_samplers.sample_dpm_2_ancestral,
    "dpmpp_2s_a": k_samplers.sample_dpmpp_2s_ancestral,
    "dpmpp_2m": k_samplers.sample_dpmpp_2m,
    "dpmpp_sde": k_samplers.sample_dpmpp_sde,
    "dpm_fast": k_samplers.sample_dpm_fast,
    "dpm_adaptive": k_samplers.sample_dpm_adaptive,
}


def sample(
    context: Context,
    sampler_name: str = None,
    noise: Tensor = None,
    batch_size: int = 1,
    shape: tuple = (),
    steps: int = 50,
    cond: Tensor = None,
    uncond: Tensor = None,
    guidance_scale: float = 0.8,
    callback=None,
    **kwargs,
):
    model = context.models["stable-diffusion"]
    denoiser = (
        k_diffusion.external.CompVisVDenoiser if model.parameterization == "v" else k_diffusion.external.CompVisDenoiser
    )
    wrapped_model = DenoiserWrap(denoiser(model))
    sigmas = wrapped_model.inner_model.get_sigmas(steps)

    sample_fn = samplers.get(sampler_name)
    x_latent = noise  # because we only use DDIM for img2img
    x_latent *= sigmas[0]

    params = {
        "model": wrapped_model,
        "x": x_latent,
        "callback": (lambda info: callback(info["x"], info["i"])) if callback is not None else None,
        "extra_args": {
            "uncond": uncond,
            "cond": cond,
            "guidance_scale": guidance_scale,
        },
    }

    if sampler_name in ("dpm_fast", "dpm_adaptive"):
        params["sigma_min"] = sigmas[-2]  # sigmas is sorted. the last element is 0, which isn't allowed
        params["sigma_max"] = sigmas[0]

        if sampler_name == "dpm_fast":
            params["n"] = steps - 1
    else:
        params["sigmas"] = sigmas

    return sample_fn(**params)


# based on https://github.com/XmYx/waifu-diffusion-gradio-hosted-by-colab-en/blob/main/scripts/kdiff.py#L109
class DenoiserWrap(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, guidance_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        return uncond + (cond - uncond) * guidance_scale
