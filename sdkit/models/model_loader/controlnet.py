import os
from pathlib import Path

from sdkit import Context
from sdkit.utils import hash_file_quick, log
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_controlnet_from_original_ckpt


def load_model(context: Context, **kwargs):
    controlnet_path = context.model_paths["controlnet"]
    controlnet_paths = controlnet_path if isinstance(controlnet_path, list) else [controlnet_path]

    controlnets = [load_controlnet(context, path) for path in controlnet_paths]
    controlnets = controlnets[0] if len(controlnets) == 1 else controlnets
    return controlnets


def load_controlnet(context, controlnet_path):
    import torch
    from sdkit.models import get_model_info_from_db
    from sdkit.models import models_db

    controlnet_base_path = os.path.splitext(controlnet_path)[0]
    controlnet_config_path = controlnet_base_path + ".yaml"
    if not os.path.exists(controlnet_config_path):
        quick_hash = hash_file_quick(controlnet_path)
        model_info = get_model_info_from_db(quick_hash=quick_hash)
        if not model_info:
            raise RuntimeError(
                "Can't find a yaml config file for the ControlNet! Please download the YAML config file (from where you downloaded the model), and put it next to the model. Please use the same name as the model file, e.g. 'foo.yaml' if the model is 'foo.safetensors'"
            )

        controlnet_config_path = model_info.get("config_url")
        models_db_path = Path(models_db.__file__).parent
        controlnet_config_path = models_db_path / controlnet_config_path

    log.info(f"Using controlnet config: {controlnet_config_path}")

    controlnet = download_controlnet_from_original_ckpt(
        controlnet_path,
        controlnet_config_path,
        from_safetensors=".safetensors" in controlnet_path,
        device="cpu",
    )
    controlnet.set_attention_slice(1)
    controlnet = controlnet.to(context.device, dtype=torch.float16 if context.half_precision else torch.float32)
    return controlnet


def unload_model(context: Context, **kwargs):
    pass
