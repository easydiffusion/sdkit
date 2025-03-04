from threading import Lock

import numpy as np
import torch
from PIL import Image

from sdkit import Context

gfpgan_temp_device_lock = Lock()  # workaround: gfpgan currently can only start on one device at a time.


def apply(context: Context, image, **kwargs):
    # this lock is also used in the model loader for gfpgan
    with gfpgan_temp_device_lock:  # Wait for any other devices to complete before starting.
        # hack for a bug in facexlib: https://github.com/xinntao/facexlib/pull/19/files
        from facexlib.detection import retinaface

        retinaface.device = context.torch_device

        image = image.convert("RGB")
        image = np.array(image, dtype=np.uint8)[..., ::-1]

        _, _, output = context.models["gfpgan"].enhance(
            image, has_aligned=False, only_center_face=False, paste_back=True
        )
        output = output[:, :, ::-1]
        output = Image.fromarray(output)

        return output
