import os
import tempfile
import traceback

from sdkit import Context
from sdkit.utils import load_tensor_file, log

"""
The VAE model overwrites the state_dict of model.first_stage_model.

We keep a copy of the original first-stage state_dict when a SD model is loaded,
and restore that copy if the custom VAE is unloaded.
"""


def load_model(context: Context, **kwargs):
    vae_model_path = context.model_paths.get("vae")

    try:
        vae = load_tensor_file(vae_model_path)
        vae = vae["state_dict"]

        if context.test_diffusers:
            from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
                convert_ldm_vae_checkpoint,
                create_vae_diffusers_config,
            )

            # the ckpt converter requires the VAE dict in the original SD style
            vae_converted = {}
            for key, value in vae.items():
                vae_converted["first_stage_model." + key] = value

            vae = vae_converted

            model = context.models["stable-diffusion"]
            m = model["default"]
            image_size = m.vae.sample_size

            original_config = model["config"]
            vae_config = create_vae_diffusers_config(original_config, image_size=image_size)
            vae_dict = convert_ldm_vae_checkpoint(vae, vae_config)

            log.info("Loading diffusers vae")
        else:
            vae_dict = {k: v for k, v in vae.items() if k[0:4] != "loss"}

        if context.half_precision:
            for key in vae_dict.keys():
                vae_dict[key] = vae_dict[key].half()

        _set_vae(context, vae_dict)

        del vae_dict
        return {}  # we don't need to access this again
    except:
        log.error(traceback.format_exc())
        log.error(f"Could not load VAE: {vae_model_path}")


def move_model_to_cpu(context: Context):
    pass


def unload_model(context: Context, **kwargs):
    base_vae = _get_base_model_vae(context)
    _set_vae(context, base_vae)


def _set_vae(context: Context, vae: dict):
    if "stable-diffusion" not in context.models:
        return

    model = context.models["stable-diffusion"]

    if context.test_diffusers:
        m = model["default"]
        m.vae.load_state_dict(vae, strict=False)
    else:
        model.first_stage_model.load_state_dict(vae, strict=False)


def _get_base_model_vae(context: Context):
    base_vae = os.path.join(tempfile.gettempdir(), "sd-base-vae.safetensors")
    return load_tensor_file(base_vae)
