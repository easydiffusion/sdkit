import json
import os


def load_tensor_file(path):
    import torch
    import safetensors.torch

    if not isinstance(path, str):
        path = str(path)

    if path.lower().endswith(".safetensors"):
        return safetensors.torch.load_file(path, device="cpu")
    else:
        return torch.load(path, map_location="cpu")


def save_tensor_file(data, path):
    import torch
    import safetensors.torch

    if path.lower().endswith(".safetensors"):
        return safetensors.torch.save_file(data, path, metadata={"format": "pt"})
    else:
        return torch.save(data, path)


def save_images(
    images: list,
    dir_path: str,
    file_name="image",
    output_format="JPEG",
    output_quality=75,
    output_lossless=False,
):
    """
    * images: a list of of PIL.Image images to save
    * dir_path: the directory path where the images will be saved
    * file_name: the file name to save. Can be a string or a function.
        if a string, the actual file name will be `{file_name}_{index}`.
        if a function, the callback function will be passed the `index` (int),
          and the returned value will be used as the actual file name. e.g `def fn(i): return 'foo' + i`
    * output_format: 'JPEG', 'PNG', or 'WEBP'
    * output_quality: an integer between 0 and 100, used for JPEG and WEBP
    * output_lossless: whether to save lossless images (WEBP only)
    """
    if dir_path is None:
        return
    os.makedirs(dir_path, exist_ok=True)

    for i, img in enumerate(images):
        actual_file_name = file_name(i) if callable(file_name) else f"{file_name}_{i}"
        path = os.path.join(dir_path, actual_file_name)
        output_lossless = output_lossless and output_format.lower() == "webp"
        img.save(f"{path}.{output_format.lower()}", quality=output_quality, lossless=output_lossless)


def save_dicts(entries: list, dir_path: str, file_name="data", output_format="txt", file_format=""):
    """
    * entries: a list of dictionaries
    * dir_path: the directory path where the files will be saved
    * file_name: the file name to save. Can be a string or a function.
        if a string, the actual file name will be `{file_name}_{index}`.
        if a function, the callback function will be passed the `index` (int),
          and the returned value will be used as the actual file name. e.g `def fn(i): return 'foo' + i`
    * output_format: 'txt', 'json', or 'embed'
        if 'embed', the metadata will be embedded in PNG files in tEXt chunks, and as EXIF UserComment for JPEG and WEBP files
    """
    from PIL import Image
    import piexif
    import piexif.helper
    from PIL.PngImagePlugin import PngInfo

    if dir_path is None:
        return
    os.makedirs(dir_path, exist_ok=True)

    for i, metadata in enumerate(entries):
        actual_file_name = file_name(i) if callable(file_name) else f"{file_name}_{i}"
        path = os.path.join(dir_path, actual_file_name)

        if output_format.lower() == "embed":
            if file_format.lower() == "png":
                targetImage = Image.open(f"{path}.{file_format.lower()}")
                embedded_metadata = PngInfo()
                for key, val in metadata.items():
                    embedded_metadata.add_text(key, str(val))
                targetImage.save(f"{path}.{file_format.lower()}", pnginfo=embedded_metadata)
            else:
                user_comment = json.dumps(metadata)
                exif_dict = {
                    "Exif": {
                        piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(user_comment, encoding="unicode")
                    }
                }
                exif_bytes = piexif.dump(exif_dict)
                piexif.insert(exif_bytes, f"{path}.{file_format.lower()}")
        else:
            with open(f"{path}.{output_format.lower()}", "w", encoding="utf-8") as f:
                if output_format.lower() == "txt":
                    for key, val in metadata.items():
                        f.write(f"{key}: {val}\n")
                elif output_format.lower() == "json":
                    json.dump(metadata, f, indent=2)
