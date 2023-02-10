"""
    This filter uses the gfpgan model to enhance the image.
"""

import torch
import numpy as np
from PIL import Image
from threading import Lock

from sdkit import Context
from sdkit.utils import log
from sdkit.utils.image_utils import convert_img_to_np_array, \
    convert_np_array_to_img

# hack for a bug in facexlib: https://github.com/xinntao/facexlib/pull/19/files
try:
    from facexlib.detection import retinaface
except ImportError:
    log.warning('facexlib not found. Please install facexlib')

# workaround: gfpgan currently can only start on one device at a time.
gfpgan_temp_device_lock = Lock()


def apply(
    context: Context,
    image: Image.Image,
    **__
) -> Image.Image:
    """
        Apply the filter to the image.
    """
    # This lock is only ever used here. No need to use timeout for the request.
    # Should never deadlock.

    # Wait for any other devices to complete before starting.
    with gfpgan_temp_device_lock:
        retinaface.device = torch.device(context.device)

        image = convert_img_to_np_array(image)

        _, _, output = context.models['gfpgan'].enhance(
            image,
            has_aligned=False,
            only_center_face=False,
            paste_back=True
        )
        return convert_np_array_to_img(output)
