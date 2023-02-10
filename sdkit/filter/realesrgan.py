"""
    Apply Real-ESRGAN to an image.
"""

import numpy as np
from PIL import Image

from sdkit import Context


def apply(
    context: Context,
    image: Image.Image,
    scale=4,
    **__
) -> Image.Image:
    """
        Apply the filter to the image.
    """
    image = image.convert('RGB')
    image = np.array(image, dtype=np.uint8)[..., ::-1]

    output, _ = context.models['realesrgan'].enhance(image, outscale=scale)
    output = output[:, :, ::-1]
    return Image.fromarray(output)
