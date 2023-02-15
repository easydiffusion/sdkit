import torch
from gfpgan import GFPGANer

from sdkit import Context


def load_model(context: Context, **kwargs):
    model_path = context.model_paths.get("gfpgan")

    # hack for a bug in facexlib: https://github.com/xinntao/facexlib/pull/19/files
    from facexlib.detection import retinaface

    retinaface.device = torch.device(context.device)

    return GFPGANer(
        device=torch.device(context.device),
        model_path=model_path,
        upscale=1,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
    )


def unload_model(context: Context, **kwargs):
    pass
