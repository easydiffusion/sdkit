from sdkit import Context
from sdkit.utils import gc, log

from . import gfpgan, hypernetwork, realesrgan, stable_diffusion, vae

models = {
    "stable-diffusion": stable_diffusion,
    "gfpgan": gfpgan,
    "realesrgan": realesrgan,
    "vae": vae,
    "hypernetwork": hypernetwork,
}


def load_model(context: Context, model_type: str, **kwargs):
    if context.model_paths.get(model_type) is None:
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


def unload_model(context: Context, model_type: str, **kwargs):
    if model_type not in context.models:
        return

    del context.models[model_type]
    models[model_type].unload_model(context)
    gc(context)

    log.info(f"unloaded {model_type} model from device: {context.device}")
