import torch

from sdkit import Context


def apply(context: Context, image, latent_upscaler_options=None, **kwargs):
    upscaler = context.models["latent_upscaler"]

    options = {}
    if latent_upscaler_options is not None:
        options["prompt"] = latent_upscaler_options["prompt"]
        options["negative_prompt"] = latent_upscaler_options["negative_prompt"]

        options["generator"] = torch.manual_seed(int(latent_upscaler_options["seed"]))

        options["num_inference_steps"] = int(latent_upscaler_options["num_inference_steps"])
        options["guidance_scale"] = float(latent_upscaler_options["guidance_scale"])
    else:
        options["prompt"] = ""
        options["generator"] = torch.manual_seed(172)
        options["num_inference_steps"] = 50
        options["guidance_scale"] = 0

    return upscaler(image=image, **options).images[0]
