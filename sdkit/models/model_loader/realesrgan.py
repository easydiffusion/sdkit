import os

import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

from sdkit import Context


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
        device=torch.device(context.device),
        scale=4,
        model_path=model_path,
        model=model_to_use,
        pre_pad=0,
        half=half,
        tile=256,
    )
    if "cuda" not in context.device:
        model.model.to(context.device)

    model.model.name = model_to_use

    return model


def unload_model(context: Context, **kwargs):
    pass
