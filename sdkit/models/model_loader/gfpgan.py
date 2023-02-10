"""
    gfpgan model loader
"""
import torch
from gfpgan import GFPGANer

from sdkit import Context
from sdkit.utils import log

# hack for a bug in facexlib: https://github.com/xinntao/facexlib/pull/19/files
try:
    from facexlib.detection import retinaface
except ImportError:
    retinaface = None
    log.warning('facexlib not found. Please install facexlib')


def load_model(context: Context, **__):
    """ Load the model. """
    model_path = context.model_paths.get('gfpgan')

    if retinaface:
        retinaface.device = torch.device(context.device)

    return GFPGANer(
        device=torch.device(context.device),
        model_path=model_path,
        upscale=1,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=None
    )


def unload_model(_: Context, **__) -> None:
    """
        Unload the model from the context.

        # TODO: Implement this?
    """
