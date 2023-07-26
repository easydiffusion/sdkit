import os

from sdkit import Context
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_controlnet_from_original_ckpt


def load_model(context: Context, **kwargs):
    controlnet_path = context.model_paths["controlnet"]
    controlnet_paths = controlnet_path if isinstance(controlnet_path, list) else [controlnet_path]

    controlnets = [load_controlnet(context, path) for path in controlnet_paths]
    controlnets = controlnets[0] if len(controlnets) == 1 else controlnets
    return controlnets


def load_controlnet(context, controlnet_path):
    import torch

    controlnet_base_path = os.path.splitext(controlnet_path)[0]
    controlnet_config_path = controlnet_base_path + ".yaml"

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
