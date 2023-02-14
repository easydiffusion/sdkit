import json
import os

import piexif
import piexif.helper
import safetensors.torch
import torch
from PIL import Image, PngImagePlugin
from PIL.PngImagePlugin import PngInfo


def load_tensor_file(path):
    if path.lower().endswith(".safetensors"):
        return safetensors.torch.load_file(path, device="cpu")
    else:
        return torch.load(path, map_location="cpu")


def save_tensor_file(data, path):
    if path.lower().endswith(".safetensors"):
        return safetensors.torch.save_file(data, path, metadata={"format": "pt"})
    else:
        return torch.save(data, path)


def save_images(images: list, dir_path: str, file_name="image", output_format="JPEG", output_quality=75):
    """
    * images: a list of of PIL.Image images to save
    * dir_path: the directory path where the images will be saved
    * file_name: the file name to save. Can be a string or a function.
        if a string, the actual file name will be `{file_name}_{index}`.
        if a function, the callback function will be passed the `index` (int),
          and the returned value will be used as the actual file name. e.g `def fn(i): return 'foo' + i`
    * output_format: 'JPEG' or 'PNG'
    * output_quality: an integer between 0 and 100, used for JPEG
    """
    if dir_path is None:
        return
    os.makedirs(dir_path, exist_ok=True)

    for i, img in enumerate(images):
        actual_file_name = file_name(i) if callable(file_name) else f"{file_name}_{i}"
        path = os.path.join(dir_path, actual_file_name)
        img.save(f"{path}.{output_format.lower()}", quality=output_quality)


def save_dicts(entries: list, dir_path: str, file_name="data", output_format="txt", file_format=""):
    """
    * entries: a list of dictionaries
    * dir_path: the directory path where the files will be saved
    * file_name: the file name to save. Can be a string or a function.
        if a string, the actual file name will be `{file_name}_{index}`.
        if a function, the callback function will be passed the `index` (int),
          and the returned value will be used as the actual file name. e.g `def fn(i): return 'foo' + i`
    * output_format: 'txt', 'json', or 'embed'
        if 'embed', the metadata will be embedded in PNG files in tEXt chunks, and as EXIF UserComment for JPEG files
    """
    if dir_path is None:
        return
    os.makedirs(dir_path, exist_ok=True)

    for i, metadata in enumerate(entries):
        actual_file_name = file_name(i) if callable(file_name) else f"{file_name}_{i}"
        path = os.path.join(dir_path, actual_file_name)

        if output_format.lower() == "embed" and file_format.lower() == "png":
            targetImage = Image.open(f"{path}.{file_format.lower()}")
            embedded_metadata = PngInfo()
            for key, val in metadata.items():
                embedded_metadata.add_text(key, str(val))
            targetImage.save(f"{path}.{file_format.lower()}", pnginfo=embedded_metadata)
        elif output_format.lower() == "embed" and file_format.lower() == "jpeg":
            targetImage = Image.open(f"{path}.{file_format.lower()}")
            user_comment = json.dumps(metadata)
            exif_dict = {
                "Exif": {piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(user_comment, encoding="unicode")}
            }
            exif_bytes = piexif.dump(exif_dict)
            targetImage.save(f"{path}.{file_format.lower()}", exif=exif_bytes)
        else:
            with open(f"{path}.{output_format.lower()}", "w", encoding="utf-8") as f:
                if output_format.lower() == "txt":
                    for key, val in metadata.items():
                        f.write(f"{key}: {val}\n")
                elif output_format.lower() == "json":
                    json.dump(metadata, f, indent=2)
