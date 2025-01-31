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

from gfpgan import GFPGANer

from sdkit import Context


def load_model(context: Context, **kwargs):
    model_path = context.model_paths.get("gfpgan")

    from sdkit.filter import gfpgan as gfpgan_filter

    with gfpgan_filter.gfpgan_temp_device_lock:
        # hack for a bug in facexlib: https://github.com/xinntao/facexlib/pull/19/files
        from facexlib.detection import retinaface

        retinaface.device = context.torch_device

        return GFPGANer(
            device=context.torch_device,
            model_path=model_path,
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
        )


def unload_model(context: Context, **kwargs):
    pass
