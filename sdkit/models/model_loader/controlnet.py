import os
from pathlib import Path

from sdkit import Context
from sdkit.utils import hash_file_quick, log


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
    from sdkit.utils import load_tensor_file

    from accelerate import cpu_offload

    import json
    from diffusers import ControlNetModel
    from diffusers.pipelines.stable_diffusion.convert_from_ckpt import convert_controlnet_checkpoint

    controlnet_state_dict = load_tensor_file(controlnet_path)
    while "state_dict" in controlnet_state_dict:
        controlnet_state_dict = controlnet_state_dict["state_dict"]

    controlnet = None
    controlnet_config_path = None

    if "mid_block.attentions.0.transformer_blocks.0.attn2.to_k.weight" in controlnet_state_dict:  # diffusers format
        v = controlnet_state_dict["mid_block.attentions.0.transformer_blocks.0.attn2.to_k.weight"]
        if v.shape[-1] in (768, 1024):
            controlnet_config_path = "configs/controlnet/controlnet_sd1.5.json"
        elif v.shape[-1] == 2048:
            controlnet_config_path = "configs/controlnet/controlnet_sdxl.json"

        if controlnet_config_path:
            models_db_path = Path(models_db.__file__).parent
            controlnet_config_path = models_db_path / controlnet_config_path

            with open(controlnet_config_path, "r") as f:
                controlnet_config = json.load(f)

            controlnet = ControlNetModel(**controlnet_config)
            controlnet.load_state_dict(controlnet_state_dict)
    else:
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

        from omegaconf import OmegaConf

        controlnet_config = OmegaConf.load(controlnet_config_path)

        controlnet = convert_controlnet_checkpoint(
            controlnet_state_dict, controlnet_config, controlnet_path, 512, None, False
        )

    # memory optimizations

    if context.vram_usage_level == "low" and "cuda" in context.device:
        controlnet = controlnet.to("cpu", torch.float16 if context.half_precision else torch.float32)

        offload_buffers = len(controlnet._parameters) > 0
        cpu_offload(controlnet, context.device, offload_buffers=offload_buffers)
    else:
        controlnet = controlnet.to(context.device, torch.float16 if context.half_precision else torch.float32)

    controlnet.set_attention_slice(1)

    try:
        import xformers

        controlnet.enable_xformers_memory_efficient_attention()
    except:
        pass

    # /memory optimizations

    return controlnet


def unload_model(context: Context, **kwargs):
    pass
