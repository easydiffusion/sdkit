import numpy as np
import cv2
from skimage import exposure
import base64
from io import BytesIO
from PIL import Image

# https://stackoverflow.com/a/61114178
def img_to_base64_str(img, output_format="PNG", output_quality=75):
    buffered = img_to_buffer(img, output_format, output_quality=output_quality)
    return buffer_to_base64_str(buffered, output_format)

def img_to_buffer(img, output_format="PNG", output_quality=75):
    buffered = BytesIO()
    if output_format.upper() == "JPEG":
        img.save(buffered, format=output_format, quality=output_quality)
    else:
        img.save(buffered, format=output_format)
    buffered.seek(0)
    return buffered

def buffer_to_base64_str(buffered, output_format="PNG"):
    buffered.seek(0)
    img_byte = buffered.getvalue()
    mime_type = "image/png" if output_format.lower() == "png" else "image/jpeg"
    img_str = f"data:{mime_type};base64," + base64.b64encode(img_byte).decode()
    return img_str

def base64_str_to_buffer(img_str):
    mime_type = "image/png" if img_str.startswith("data:image/png;") else "image/jpeg"
    img_str = img_str[len(f"data:{mime_type};base64,"):]
    data = base64.b64decode(img_str)
    buffered = BytesIO(data)
    return buffered


def base64_str_to_img(img_str) -> Image.Image:
    buffered = base64_str_to_buffer(img_str)
    return Image.open(buffered)


def resize_img(img: Image, desired_width, desired_height, clamp_to_64=False):
    w, h = img.size

    if desired_width is not None and desired_height is not None:
        w, h = desired_width, desired_height

    if clamp_to_64:
        w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64

    return img.resize((w, h), resample=Image.Resampling.LANCZOS)

def apply_color_profile(orig_image: Image, image_to_modify: Image):
    reference = cv2.cvtColor(np.asarray(orig_image), cv2.COLOR_RGB2LAB)
    image_to_modify = cv2.cvtColor(np.asarray(image_to_modify), cv2.COLOR_RGB2LAB)
    matched = exposure.match_histograms(image_to_modify, reference, channel_axis=2)

    return Image.fromarray(cv2.cvtColor(matched, cv2.COLOR_LAB2RGB).astype("uint8"))


def convert_img_to_np_array(image: Image.Image) -> np.ndarray:
    """
        Convert a PIL image to a numpy array.
    """
    image = image.convert('RGB')
    return np.array(image, dtype=np.uint8)[..., ::-1]


def convert_np_array_to_img(arr: np.ndarray) -> Image.Image:
    """
        Convert a numpy array to a PIL image
    """
    return Image.fromarray(arr[:, :, ::-1])
