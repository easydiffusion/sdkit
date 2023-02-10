"""
    Apply filters to images
"""

from typing import Dict, List, Union
from types import ModuleType
from PIL.Image import Image
from sdkit import Context
from sdkit.filter import gfpgan, realesrgan
from sdkit.utils import base64_str_to_img, log, gc


# Your module must contain a function called apply.
FILTER_MODULES: Dict[str, ModuleType] = {
    'gfpgan': gfpgan,
    'realesrgan': realesrgan,
}


def apply_filters(
    context: Context,
    filters: Union[str, List[str]],
    images: Union[str, List[Union[str, Image]]],
    **kwargs
) -> list[Image]:
    '''
    * context: Context
    * filters: filter_type (string) or list of strings
    * images: str or PIL.Image or list of str/PIL.Image - image to filter.
        if a string is passed, it needs to be a base64-encoded image

    returns: PIL.Image - filtered image
    '''
    _images: List[str | Image] = images \
        if isinstance(images, list) else [images]
    _filters: List[str] = filters if isinstance(filters, list) else [filters]

    return [
        apply_filter_single_image(context, _filters, image, **kwargs)
        for image in _images
    ]


def apply_filter_single_image(
    context: Context,
    filters: List[str],
    image: Union[str, Image],
    **kwargs
) -> Image:
    """
        Apply filters to a single image
    """
    _image: Image = base64_str_to_img(image) \
        if isinstance(image, str) else image

    for filter_type in filters:
        log.info('Applying %s...', filter_type)
        gc(context)

        _image = FILTER_MODULES[filter_type].apply(context, _image, **kwargs)

    return _image
