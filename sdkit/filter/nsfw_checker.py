import torch
from PIL import ImageFilter

from sdkit import Context


def apply(context: Context, image, blur_radius: float = 75, **kwargs):
    safety_checker, feature_extractor = context.models["nsfw_checker"]

    images = [torch.Tensor([0])]  # just a dummy array, the real info is in `safety_checker_input``

    safety_checker_input = feature_extractor(image, return_tensors="pt").to("cpu")
    _, has_nsfw_concept = safety_checker(images=images, clip_input=safety_checker_input.pixel_values)

    if has_nsfw_concept[0]:
        image = image.filter(ImageFilter.GaussianBlur(blur_radius))

    return image
