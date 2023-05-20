from sdkit import Context
from sdkit.utils import gc, log

from . import (
    codeformer,
    gfpgan,
    hypernetwork,
    latent_upscaler,
    lora,
    nsfw_checker,
    realesrgan,
    stable_diffusion,
    vae,
)

models = {
    "stable-diffusion": stable_diffusion,
    "codeformer": codeformer,
    "gfpgan": gfpgan,
    "realesrgan": realesrgan,
    "vae": vae,
    "hypernetwork": hypernetwork,
    "nsfw_checker": nsfw_checker,
    "lora": lora,
    "latent_upscaler": latent_upscaler
}


def load_model(context: Context, model_type: str, **kwargs):
    if context.model_paths.get(model_type) is None and model_type != "nsfw_checker":
        return

    if model_type in context.models:
        unload_model(context, model_type)

    log.info(f"loading {model_type} model from {context.model_paths.get(model_type)} to device: {context.device}")

    context.models[model_type] = models[model_type].load_model(context, **kwargs)

    log.info(f"loaded {model_type} model from {context.model_paths.get(model_type)} to device: {context.device}")

    # reload dependent models
    if model_type == "stable-diffusion":
        load_model(context, "vae")
        load_model(context, "hypernetwork")
        if "lora" in context.models and hasattr(context, "_last_lora_alpha"):
            del context._last_lora_alpha
        load_model(context, "lora")


def unload_model(context: Context, model_type: str, **kwargs):
    if model_type not in context.models:
        return

    if context.test_diffusers:
        models[model_type].unload_model(context)
        del context.models[model_type]
    else:
        del context.models[model_type]
        models[model_type].unload_model(context)

    gc(context)

    log.info(f"unloaded {model_type} model from device: {context.device}")
