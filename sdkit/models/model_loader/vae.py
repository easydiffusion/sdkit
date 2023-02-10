"""
    VAE model loader
"""
import os
import traceback
import tempfile

from sdkit import Context
from sdkit.utils import log, load_tensor_file


"""
The VAE model overwrites the state_dict of model.first_stage_model.

We keep a copy of the original first-stage state_dict when a SD model is loaded,
and restore that copy if the custom VAE is unloaded.
"""


def load_model(context: Context, **kwargs):
    """
        Load the model.
    """
    vae_model_path = context.model_paths.get('vae')

    try:
        vae = load_tensor_file(vae_model_path)
        vae_dict = {
            k: v
            for k, v in vae["state_dict"].items()
            if k[0:4] != "loss"
        }
        if context.half_precision:
            for key in vae_dict.keys():
                vae_dict[key] = vae_dict[key].half()

        _set_vae(context, vae_dict)

        del vae_dict
        # we don't need to access this again
        return {}
    except:
        log.error('%s', traceback.format_exc())
        log.error('Could not load VAE: %s', vae_model_path)
    return None


def move_model_to_cpu(_: Context):
    """
        Move the model to CPU.

        # TODO: Implement this?
    """


def unload_model(context: Context, **__):
    """
        Unload the model.
    """
    base_vae = _get_base_model_vae(context)
    _set_vae(context, base_vae)


def _set_vae(context: Context, vae: dict):
    """
        Set the VAE state_dict.
    """
    if 'stable-diffusion' not in context.models:
        return

    model = context.models['stable-diffusion']
    model.first_stage_model.load_state_dict(vae, strict=False)


def _get_base_model_vae(_: Context):
    """
        Get the base model VAE.
    """
    base_vae = os.path.join(tempfile.gettempdir(), 'sd-base-vae.safetensors')
    return load_tensor_file(base_vae)
