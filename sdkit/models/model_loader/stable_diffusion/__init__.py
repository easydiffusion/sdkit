import os
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from transformers import logging as tr_logging
from torch.nn.functional import silu
import ldm.modules.attention
import ldm.modules.diffusionmodules.model
import tempfile
from urllib.parse import urlparse

from sdkit import Context
from sdkit.utils import load_tensor_file, save_tensor_file, hash_file_quick, download_file, log

tr_logging.set_verbosity_error() # suppress unnecessary logging

def load_model(context: Context, scan_model=True, **kwargs):
    from . import optimizations
    from sdkit.models import scan_model as scan_model_fn

    model_path = context.model_paths.get('stable-diffusion')
    config_file_path = get_model_config_file(context)
    if config_file_path is None:
        raise Exception(f'Unknown model! No config file path specified in context.model_configs for the "stable-diffusion" model!')

    log.info(f'using config: {config_file_path}')
    config = OmegaConf.load(config_file_path)
    config.model.params.unet_config.params.use_fp16 = context.half_precision

    extra_config = config.get('extra', {})
    attn_precision = extra_config.get('attn_precision', 'fp16')

    if scan_model:
        scan_result = scan_model_fn(model_path)
        if scan_result.issues_count > 0 or scan_result.infected_files > 0:
            raise Exception(f'Model scan failed! Potentially infected model: {model_path}')

    sd = load_tensor_file(model_path)
    sd = sd['state_dict'] if 'state_dict' in sd else sd

    model = instantiate_from_config(config.model)
    _, _ = model.load_state_dict(sd, strict=False)
    if context.half_precision: model = model.half()
    optimizations.send_to_device(context, model)
    model.eval()
    del sd

    # optimize CrossAttention.forward() for faster performance, and lower VRAM usage
    ldm.modules.attention.CrossAttention.forward = optimizations.make_attn_forward(context, attn_precision=attn_precision)
    ldm.modules.diffusionmodules.model.nonlinearity = silu

    # save the model vae into a temp folder (used for restoring the default VAE, if a custom VAE is unloaded)
    save_tensor_file(model.first_stage_model.state_dict(), os.path.join(tempfile.gettempdir(), 'sd-base-vae.safetensors'))

    # optimizations.print_model_size_breakdown(model)

    return model

def unload_model(context: Context, **kwargs):
    context.module_in_gpu = None # don't keep a dangling reference, prevents gc

def get_model_config_file(context: Context):
    from sdkit.models import get_model_info_from_db

    if 'stable-diffusion' in context.model_configs:
        return context.model_configs['stable-diffusion']

    model_path = context.model_paths['stable-diffusion']
    quick_hash = hash_file_quick(model_path)

    model_info = get_model_info_from_db(quick_hash=quick_hash)
    if model_info is None:
        return
    config_url = model_info.get('config_url')
    if config_url is None:
        return

    config_file_name = os.path.basename(urlparse(config_url).path)
    model_dir_name = os.path.dirname(model_path)
    config_file_path = os.path.join(model_dir_name, config_file_name)

    if not os.path.exists(config_file_path):
        download_file(config_url, config_file_path)

    return config_file_path
