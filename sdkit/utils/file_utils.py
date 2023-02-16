import json
import os

import piexif
import piexif.helper
import safetensors.torch
import torch
from PIL import Image
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
        raise ValueError("No directory specified")
    os.makedirs(dir_path, exist_ok=True)

    for i, metadata in enumerate(entries):
        actual_file_name = file_name(i) if callable(file_name) else f"{file_name}_{i}"
        path = os.path.join(dir_path, actual_file_name)

        file_format = file_format.lower()
        output_format = output_format.lower()

        if output_format == "embed":
            if file_format == "png":
                save_png_metadata(file_format, metadata, path)
            elif file_format == "jpeg":
                save_jpeg_exif(file_format, metadata, path)
            else:
                raise ValueError(f"Unknown format image type: {file_format}")
        else:
            save_text_metadata(output_format, metadata, path)


def save_text_metadata(output_format: str, metadata: dict, path: str):
    """Save the metadata in a text file either as text or as json

    Args:
      output_format (str): the file format (choice: text or json)
      metadata (dict): the metadata to save to the file
      path (str): path the output file
    """
    with open(f"{path}.{output_format}", "w", encoding="utf-8") as f:
        if output_format == "txt":
            for key, val in metadata.items():
                f.write(f"{key}: {val}\n")
        elif output_format == "json":
            json.dump(metadata, f, indent=2)
        else:
            raise ValueError(f"Unknown format file type {output_format}")


def save_png_metadata(file_format: str, metadata: dict, path: str):
    """Save the metadata in a PNG

    Args:
        file_format (str): file format
        metadata (dict): the metadata to save to the file
        path (str): path to the image file
    """
    target_image = Image.open(f"{path}.{file_format}")
    embedded_metadata = PngInfo()
    for key, val in metadata.items():
        embedded_metadata.add_text(key, str(val))
    target_image.save(f"{path}.{file_format}", pnginfo=embedded_metadata)


def save_jpeg_exif(file_format: str, metadata: dict, path: str):
    """Save the metadata as EXIF in a JPEG

    Args:
        file_format (str): file format
        metadata (dict): the metadata to save to the file
        path (str): path to the image file
    """
    target_image = Image.open(f"{path}.{file_format}")
    user_comment = json.dumps(metadata)
    exif_dict = {"Exif": {piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(user_comment, encoding="unicode")}}
    exif_bytes = piexif.dump(exif_dict)
    target_image.save(f"{path}.{file_format}", exif=exif_bytes)
