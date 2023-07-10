from einops import rearrange, repeat

from sdkit import Context
from sdkit.utils import log


def to_tensor(x, device, dtype=None):
    import torch
    import numpy as np

    if dtype is None:
        dtype = torch.float32

    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device=device, dtype=dtype)
    elif isinstance(x, list):
        if all(isinstance(item, torch.Tensor) for item in x):
            return torch.stack(x).to(device=device, dtype=dtype)
        elif all(isinstance(item, np.ndarray) for item in x):
            return [torch.from_numpy(item).to(device=device, dtype=dtype) for item in x]
    elif isinstance(x, tuple):
        return tuple(torch.from_numpy(item).to(device=device, dtype=dtype) for item in x)
    else:
        log.debug(f"X:{x} and X's type{type(x)}")
        return torch.tensor(x).to(device=device, dtype=dtype)


def img_to_tensor(img, batch_size, device, half_precision: bool, shift_range=False, unsqueeze=False):
    from PIL import Image, ImageOps
    import torch
    import numpy as np

    if img is None:
        return None

    img = np.array(img).astype(np.float32) / 255.0
    img = img[None].transpose(0, 3, 1, 2)
    img = torch.from_numpy(img)
    img = 2.0 * img - 1.0 if shift_range else img
    img = img.to(device)

    if "cuda" in device and half_precision:
        img = img.half()

    if unsqueeze:
        img = img[0][0].unsqueeze(0).repeat(4, 1, 1).unsqueeze(0)

    img = repeat(img, "1 ... -> b ...", b=batch_size)

    return img


def get_image_latent_and_mask(context: Context, image, mask, desired_width, desired_height, batch_size):
    """
    Assumes model is on the correct device
    """
    from .image_utils import resize_img
    from PIL import Image, ImageOps

    if image is None or "stable-diffusion" not in context.models:
        return None, None

    model = context.models["stable-diffusion"]

    image = image.convert("RGB")
    image = resize_img(image, desired_width, desired_height, clamp_to_64=True)
    image = img_to_tensor(image, batch_size, context.device, context.half_precision, shift_range=True)
    image = model.get_first_stage_encoding(model.encode_first_stage(image))  # move to latent space

    if mask is None:
        return image, None

    mask = mask.convert("RGB")
    mask = resize_img(mask, image.shape[3], image.shape[2])
    mask = ImageOps.invert(mask)
    mask = img_to_tensor(mask, batch_size, context.device, context.half_precision, unsqueeze=True)

    return image, mask


def latent_samples_to_images(context: Context, samples):
    import torch
    import numpy as np
    from PIL import Image, ImageOps

    model = context.models["stable-diffusion"]

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


def diffusers_latent_samples_to_images(context: Context, latent_samples):
    import torch

    @torch.no_grad()
    def apply():
        samples, model = latent_samples
        samples = model.vae.decode(samples / model.vae.config.scaling_factor, return_dict=False)[0]
        do_denormalize = [True] * samples.shape[0]
        images = model.image_processor.postprocess(samples, output_type="pil", do_denormalize=do_denormalize)
        images = [img.convert("RGB") for img in images]
        return images

    return apply()


def tensor_to_bitmap(tensor):
    "Generates a grayscale bitmap from the given tensor"
    import torch
    import numpy as np
    from PIL import Image, ImageOps

    assert np.ndim(tensor) < 5
    if np.ndim(tensor) == 4:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    if np.ndim(tensor) == 3:
        return [tensor_to_bitmap(tensor[i]) for i in range(tensor.shape[0])]
    minVal = torch.min(tensor)
    maxVal = torch.max(tensor)
    delta = maxVal - minVal
    tensor = (tensor - minVal) / delta * 255
    tensor = np.array(tensor.cpu(), dtype=np.uint8)
    return Image.fromarray(tensor)
