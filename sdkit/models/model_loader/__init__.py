from sdkit import Context
from sdkit.utils import gc, log

import importlib
import traceback
from threading import Lock


class Loader:
    pass


model_load_lock = Lock()


def _get_module(model_type):
    models = {  # model_type -> local_module_name
        "stable-diffusion": "stable_diffusion",
        "codeformer": "codeformer",
        "gfpgan": "gfpgan",
        "realesrgan": "realesrgan",
        "vae": "vae",
        "hypernetwork": "hypernetwork",
        "nsfw_checker": "nsfw_checker",
        "lora": "lora",
        "latent_upscaler": "latent_upscaler",
        "controlnet": "controlnet",
        "embeddings": "embeddings",
    }
    if model_type not in models:
        return

    module_name = models[model_type]

    return importlib.import_module("." + module_name, __name__)


def load_model(context: Context, model_type: str, **kwargs):
    if context.test_diffusers:
        from . import diffusers_bugfixes

    if context.model_paths.get(model_type) is None and model_type != "nsfw_checker":
        return

    if model_type in context.models:
        unload_model(context, model_type)

    with model_load_lock:
        # only allow one model to load at a time, regardless of how many threads are running
        # this works around a thread-unsafe behavior of accelerate: https://github.com/huggingface/diffusers/issues/4296

        log.info(f"loading {model_type} model from {context.model_paths.get(model_type)} to device: {context.device}")

        context.models[model_type] = get_loader_module(model_type).load_model(context, **kwargs)

    log.info(f"loaded {model_type} model from {context.model_paths.get(model_type)} to device: {context.device}")

    # reload dependent models
    if model_type == "stable-diffusion":
        for m in ("vae", "hypernetwork", "lora", "embeddings"):
            try:
                if m == "lora" and "lora" in context.models and hasattr(context, "_last_lora_alpha"):
                    del context._last_lora_alpha
                load_model(context, m)
            except Exception as e:
                log.error(f"Could not load dependent model: {m}")
                traceback.print_exc()
                if m in context.models:
                    del context.models[m]

                gc(context)


def unload_model(context: Context, model_type: str, **kwargs):
    if model_type not in context.models:
        return

    if context.test_diffusers:
        get_loader_module(model_type).unload_model(context)
        del context.models[model_type]
    else:
        del context.models[model_type]
        get_loader_module(model_type).unload_model(context)

    gc(context)

    log.info(f"unloaded {model_type} model from device: {context.device}")


def get_loader_module(model_type):
    module = _get_module(model_type)
    if module is None:
        from . import controlnet_filters

        if model_type in controlnet_filters.filters:
            module = Loader()
            module.load_model = controlnet_filters.make_load_model(model_type)
            module.unload_model = controlnet_filters.make_unload_model(model_type)

    return module
