import base64
import re
from io import BytesIO


# https://stackoverflow.com/a/61114178
def img_to_base64_str(img, output_format="PNG", output_quality=75, output_lossless=False):
    buffered = img_to_buffer(img, output_format, output_quality=output_quality, output_lossless=output_lossless)
    return buffer_to_base64_str(buffered, output_format)


def img_to_buffer(img, output_format="PNG", output_quality=75, output_lossless=False):
    buffered = BytesIO()
    if output_format.upper() == "PNG":
        img.save(buffered, format=output_format)
    elif output_format.upper() == "WEBP":
        img.save(buffered, format=output_format, quality=output_quality, lossless=output_lossless)
    else:
        img.save(buffered, format=output_format, quality=output_quality)
    buffered.seek(0)
    return buffered


def buffer_to_base64_str(buffered, output_format="PNG"):
    buffered.seek(0)
    img_byte = buffered.getvalue()
    mime_type = f"image/{output_format.lower()}"
    img_str = f"data:{mime_type};base64," + base64.b64encode(img_byte).decode()
    return img_str


def base64_str_to_buffer(img_str):
    img_str = re.sub(r"^data:image/[a-z]+;base64,", "", img_str)
    data = base64.b64decode(img_str)
    buffered = BytesIO(data)
    return buffered


def base64_str_to_img(img_str):
    from PIL import Image

    buffered = base64_str_to_buffer(img_str)
    img = Image.open(buffered)
    return img


def resize_img(img, desired_width, desired_height, clamp_to_64=False):
    from PIL import Image

    w, h = img.size

    if desired_width is not None and desired_height is not None:
        w, h = desired_width, desired_height

    if clamp_to_64:
        w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64

    return img.resize((w, h), resample=Image.Resampling.LANCZOS)


def apply_color_profile(orig_image, image_to_modify):
    from PIL import Image
    import cv2
    import numpy as np
    from skimage import exposure

    reference = cv2.cvtColor(np.asarray(orig_image), cv2.COLOR_RGB2LAB)
    image_to_modify = cv2.cvtColor(np.asarray(image_to_modify), cv2.COLOR_RGB2LAB)
    matched = exposure.match_histograms(image_to_modify, reference, channel_axis=2)

    return Image.fromarray(cv2.cvtColor(matched, cv2.COLOR_LAB2RGB).astype("uint8"))
