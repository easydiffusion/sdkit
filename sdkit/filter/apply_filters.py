from sdkit import Context
from sdkit.utils import base64_str_to_img, gc, log

import importlib


class Filter:
    pass


def _get_module(model_type):
    modules = {  # filter_name -> local_module_name
        "gfpgan": "gfpgan",
        "realesrgan": "realesrgan",
        "nsfw_checker": "nsfw_checker",
        "codeformer": "codeformer",
        "latent_upscaler": "latent_upscaler",
    }
    if model_type not in modules:
        return

    module_name = modules[model_type]
    base_package = ".".join(__name__.split(".")[:-1])

    return importlib.import_module("." + module_name, base_package)


def apply_filters(context: Context, filters, images, **kwargs):
    """
    * context: Context
    * filters: filter_type (string) or list of strings
    * images: str or PIL.Image or list of str/PIL.Image - image to filter. if a string is passed, it needs to be a base64-encoded image

    returns: [PIL.Image] - list of filtered images
    """
    images = images if isinstance(images, list) else [images]
    filters = filters if isinstance(filters, list) else [filters]

    return [apply_filter_single_image(context, filters, image, **kwargs) for image in images]


def apply_filter_single_image(context, filters, image, **kwargs):
    image = base64_str_to_img(image) if isinstance(image, str) else image

    for filter_type in filters:
        log.info(f"Applying {filter_type}...")
        gc(context)

        image = get_filter_module(filter_type).apply(context, image, **kwargs)

    return image


def get_filter_module(filter_type):
    module = _get_module(filter_type)
    if module is None:  # use the default filter implementation

        def default_apply(context: Context, image, **kwargs):
            model = context.models[filter_type]
            return model(image)

        module = Filter()
        module.apply = default_apply

    return module
