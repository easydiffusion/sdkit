import os
import json
import torch
import safetensors.torch

def load_tensor_file(path):
    if path.lower().endswith(".safetensors"):
        return safetensors.torch.load_file(path, device="cpu")
    else:
        return torch.load(path, map_location="cpu")

def save_tensor_file(data, path):
    if path.lower().endswith(".safetensors"):
        return safetensors.torch.save_file(data, path, metadata={"format": "pt"})
    else:
        return torch.load(path, map_location="cpu")

def save_images(images: list, dir_path: str, file_name='image', output_format='JPEG', output_quality=75):
    '''
    * images: a list of of PIL.Image images to save
    * dir_path: the directory path where the images will be saved
    * file_name: the file name to save. Can be a string or a function.
        if a string, the actual file name will be `{file_name}_{index}`.
        if a function, the callback function will be passed the `index` (int),
          and the returned value will be used as the actual file name. e.g `def fn(i): return 'foo' + i`
    * output_format: 'JPEG' or 'PNG'
    * output_quality: an integer between 0 and 100, used for JPEG
    '''
    if dir_path is None: return
    os.makedirs(dir_path, exist_ok=True)

    for i, img in enumerate(images):
        actual_file_name = file_name(i) if callable(file_name) else f'{file_name}_{i}'
        path = os.path.join(dir_path, actual_file_name)
        img.save(f'{path}.{output_format.lower()}', quality=output_quality)

def save_dicts(entries: list, dir_path: str, file_name='data', output_format='txt'):
    '''
    * entries: a list of dictionaries
    * dir_path: the directory path where the files will be saved
    * file_name: the file name to save. Can be a string or a function.
        if a string, the actual file name will be `{file_name}_{index}`.
        if a function, the callback function will be passed the `index` (int),
          and the returned value will be used as the actual file name. e.g `def fn(i): return 'foo' + i`
    * output_format: 'txt' or 'json'
    '''
    if dir_path is None: return
    os.makedirs(dir_path, exist_ok=True)

    for i, metadata in enumerate(entries):
        actual_file_name = file_name(i) if callable(file_name) else f'{file_name}_{i}'
        path = os.path.join(dir_path, actual_file_name)
        with open(f'{path}.{output_format.lower()}', 'w', encoding='utf-8') as f:
            if output_format.lower() == 'txt':
                for key, val in metadata.items():
                    f.write(f'{key}: {val}\n')
            elif output_format.lower() == 'json':
                json.dump(metadata, f)
