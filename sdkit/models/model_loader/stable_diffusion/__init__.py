import os
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from transformers import logging as tr_logging
from torch.nn.functional import silu
import ldm.modules.attention
import ldm.modules.diffusionmodules.model
import tempfile
from urllib.parse import urlparse
from pathlib import Path

from sdkit import Context
from sdkit.utils import load_tensor_file, save_tensor_file, hash_file_quick, download_file, log

tr_logging.set_verbosity_error() # suppress unnecessary logging

def load_model(context: Context, scan_model=True, check_for_config_with_same_name=True, **kwargs):
    from . import optimizations
    from sdkit.models import scan_model as scan_model_fn

    model_path = context.model_paths.get('stable-diffusion')
    config_file_path = get_model_config_file(context, check_for_config_with_same_name)

    if scan_model:
        scan_result = scan_model_fn(model_path)
        if scan_result.issues_count > 0 or scan_result.infected_files > 0:
            raise Exception(f'Model scan failed! Potentially infected model: {model_path}')

    # load the model file
    sd = load_tensor_file(model_path)
    sd = sd['state_dict'] if 'state_dict' in sd else sd

    # try to guess the config, if no config file was given
    # check if a key specific to SD 2.0 is missing
    if config_file_path is None and 'cond_stage_model.model.ln_final.bias' not in sd.keys():
        # try using an SD 1.4 config
        from sdkit.models import get_model_info_from_db
        sd_v1_4_info = get_model_info_from_db(model_type='stable-diffusion', model_id='1.4')
        config_file_path = resolve_model_config_file_path(sd_v1_4_info, model_path)

    # load the config
    if config_file_path is None:
        raise Exception(f'Unknown model! No config file path specified in context.model_configs for the "stable-diffusion" model!')

    log.info(f'using config: {config_file_path}')
    config = OmegaConf.load(config_file_path)
    config.model.params.unet_config.params.use_fp16 = context.half_precision

    extra_config = config.get('extra', {})
    attn_precision = extra_config.get('attn_precision', 'fp16' if context.half_precision else 'fp32')
    log.info(f'using attn_precision: {attn_precision}')

    # instantiate the model
    model = instantiate_from_config(config.model)
    _, _ = model.load_state_dict(sd, strict=False)

    model = model.half() if context.half_precision else model.float()

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

def get_model_config_file(context: Context, check_for_config_with_same_name):
    from sdkit.models import get_model_info_from_db

    if context.model_configs.get('stable-diffusion') is not None:
        return context.model_configs['stable-diffusion']

    model_path = context.model_paths['stable-diffusion']

    if check_for_config_with_same_name:
        model_name_path = os.path.splitext(model_path)[0]
        model_config_path = f'{model_name_path}.yaml'
        if os.path.exists(model_config_path):
            return model_config_path

    quick_hash = hash_file_quick(model_path)
    model_info = get_model_info_from_db(quick_hash=quick_hash)

    return resolve_model_config_file_path(model_info, model_path)

def resolve_model_config_file_path(model_info, model_path):
    if model_info is None:
        return
    config_url = model_info.get('config_url')
    if config_url is None:
        return

    if config_url.startswith('http'):
        config_file_name = os.path.basename(urlparse(config_url).path)
        model_dir_name = os.path.dirname(model_path)
        config_file_path = os.path.join(model_dir_name, config_file_name)

        if not os.path.exists(config_file_path):
            download_file(config_url, config_file_path)
    else:
        from sdkit.models import models_db
        models_db_path = Path(models_db.__file__).parent
        config_file_path = models_db_path/config_url

    return config_file_path
