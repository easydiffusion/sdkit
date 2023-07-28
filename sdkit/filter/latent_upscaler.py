import torch

from sdkit import Context


def apply(
    context: Context, image, prompt="", negative_prompt="", seed=172, num_inference_steps=50, guidance_scale=0, **kwargs
):
    upscaler = context.models["latent_upscaler"]

    options = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "generator": torch.manual_seed(seed),
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
    }

    return upscaler(image=image, **options).images[0]
