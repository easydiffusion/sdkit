"""
    Latent utils
"""
import numpy as np
import torch
from PIL import Image, ImageOps
from einops import repeat, rearrange

from sdkit import Context


def img_to_tensor(
    img: Image.Image,
    batch_size,
    device,
    half_precision: bool,
    shift_range=False,
    unsqueeze=False
):
    """ Convert PIL image to torch tensor. """
    if img is None:
        return None

    _img: np.ndarray = np.array(img).astype(np.float32) / 255.0
    _img = _img[None].transpose(0, 3, 1, 2)
    _img = torch.from_numpy(_img)
    _img = 2. * _img - 1. if shift_range else _img
    _img = _img.to(device)

    if device != "cpu" and half_precision:
        _img = _img.half()

    if unsqueeze:
        _img = _img[0][0].unsqueeze(0).repeat(4, 1, 1).unsqueeze(0)

    _img = repeat(_img, '1 ... -> b ...', b=batch_size)

    return _img


def get_image_latent_and_mask(
    context: Context,
    image: Image.Image,
    mask: Image.Image,
    desired_width,
    desired_height,
    batch_size
):
    """
    Assumes model is on the correct device
    """
    from sdkit.utils import resize_img
    if image is None or 'stable-diffusion' not in context.models:
        return None, None

    model = context.models['stable-diffusion']

    image = image.convert('RGB')
    image = resize_img(image, desired_width, desired_height, clamp_to_64=True)
    image = img_to_tensor(image, batch_size, context.device, 
                                 context.half_precision, shift_range=True)
    # move to latent space
    image = model.get_first_stage_encoding(model.encode_first_stage(image))

    if mask is None:
        return image, None

    mask = mask.convert('RGB')
    mask = resize_img(mask, image.shape[3], image.shape[2])
    mask = ImageOps.invert(mask)
    mask = img_to_tensor(mask, batch_size, context.device, 
                                context.half_precision, unsqueeze=True)

    return image, mask


def latent_samples_to_images(context: Context, samples):
    """
        Latent samples to images
    """
    model = context.models['stable-diffusion']

    if context.half_precision and samples.dtype != torch.float16:
        samples = samples.half()

    samples = model.decode_first_stage(samples)
    samples = torch.clamp((samples + 1.0) / 2.0, min=0.0, max=1.0)

    images = []
    for sample in samples:
        sample = 255.0 * rearrange(sample.cpu().numpy(), "c h w -> h w c")
        sample = sample.astype(np.uint8)
        images.append(Image.fromarray(sample))

    return images
