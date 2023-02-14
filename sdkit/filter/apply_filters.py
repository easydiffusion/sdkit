from . import gfpgan, realesrgan

from sdkit import Context
from sdkit.utils import base64_str_to_img, log, gc

filter_modules = {
    'gfpgan': gfpgan,
    'realesrgan': realesrgan,
}

def apply_filters(context: Context, filters, images, **kwargs):
    '''
    * context: Context
    * filters: filter_type (string) or list of strings
    * images: str or PIL.Image or list of str/PIL.Image - image to filter. if a string is passed, it needs to be a base64-encoded image

    returns: PIL.Image - filtered image
    '''
    images = images if isinstance(images, list) else [images]
    filters = filters if isinstance(filters, list) else [filters]

    return [apply_filter_single_image(context, filters, image, **kwargs) for image in images]

def apply_filter_single_image(context, filters, image, **kwargs):
    image = base64_str_to_img(image) if isinstance(image, str) else image

    for filter_type in filters:
        log.info(f'Applying {filter_type}...')
        gc(context)

        image = filter_modules[filter_type].apply(context, image, **kwargs)

    return image
