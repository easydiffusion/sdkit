import os

import torch

# hack for basicsr https://github.com/XPixelGroup/BasicSR/pull/650
# credit: https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/14186/files

import sys

try:
    import torchvision.transforms.functional_tensor
except ImportError:
    try:
        import torchvision.transforms.functional as functional

        sys.modules["torchvision.transforms.functional_tensor"] = functional
    except ImportError:
        pass
# /hack

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

from sdkit import Context
from sdkit.utils import is_cpu_device


def load_model(context: Context, **kwargs):
    model_path = context.model_paths.get("realesrgan")

    RealESRGAN_models = {
        "RealESRGAN_x4plus": RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
        "RealESRGAN_x4plus_anime_6B": RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4
        ),
    }

    model_to_use = os.path.basename(model_path)
    model_to_use, _ = os.path.splitext(model_to_use)
    model_to_use = RealESRGAN_models[model_to_use]

    half = context.half_precision
    model = RealESRGANer(
        device=context.torch_device,
        scale=4,
        model_path=model_path,
        model=model_to_use,
        pre_pad=0,
        half=half,
        tile=256,
    )
    if is_cpu_device(context.torch_device):
        model.model.to(context.torch_device)

    model.model.name = model_to_use

    return model


def unload_model(context: Context, **kwargs):
    pass
