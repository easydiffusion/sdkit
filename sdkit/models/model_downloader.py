import os
from urllib.parse import urlparse

from sdkit.utils import download_file, log

def download_models(models: dict, download_base_dir: str=None, subdir_for_model_type=True):
    '''
    Downloads the requested models (and config files) based on the SDKit models database.
    Resumes incomplete downloads, and shows a progress bar.

    Args:
    * models: dict of {string: string or list}. Models to download of `model_type: model_ids`.
              E.g. `{'stable-diffusion': ['1.4', '1.5-inpainting'], 'gfpgan': '1.3'}`
              `model_ids` can be a single string, or a list of strings.
    * download_base_dir: str - base directory inside which the models are downloaded.
              Note: The actual download directorypath will include the `model_type` if `subdir_for_model_type` is `True`.
              If `download_base_dir` is `None`, it'll use the `~/.cache/sdkit` folder, where `~` is the home or user-folder.
    * subdir_for_model_type: bool - default True. Saves the downloaded model in a subdirectory (named with the model_type).
              For e.g. if `download_base_dir` is `D:\\models`, then a `stable-diffusion` type model is downloaded to
              `D:\\models\\stable-diffusion`, a `hypernetwork` type model is downloaded to `D:\\models\\hypernetwork` and so on.
    '''
    for model_type, model_ids in models.items():
        model_ids = model_ids if isinstance(model_ids, list) else [model_ids]

        for model_id in model_ids:
            download_model(model_type, model_id, download_base_dir, subdir_for_model_type)

def download_model(model_type: str, model_id: str, download_base_dir: str=None, subdir_for_model_type=True):
    '''
    Downloads the requested model (and config file) based on the SDKit models database.
    Resumes incomplete downloads, and shows a progress bar.

    Args:
    * model_type: str
    * model_id: str
    * download_base_dir: str - base directory inside which the models are downloaded.
              Note: The actual download directory path will include the `model_type` if `subdir_for_model_type` is `True`.
              If `download_base_dir` is `None`, it'll use the `~/.cache/sdkit` folder, where `~` is the home or user-folder.
    * subdir_for_model_type: bool - default True. Saves the downloaded model in a subdirectory (named with the model_type).
              For e.g. if `download_base_dir` is `D:\\models`, then a `stable-diffusion` type model is downloaded to
              `D:\\models\\stable-diffusion`, a `hypernetwork` type model is downloaded to `D:\\models\\hypernetwork` and so on.
    '''
    download_base_dir = get_actual_base_dir(model_type, download_base_dir, subdir_for_model_type)
    try:
        model_url, model_file_name = get_url_and_filename(model_type, model_id, url_key='url')
        config_url, config_file_name = get_url_and_filename(model_type, model_id, url_key='config_url')

        if model_url is None:
            log.warn(f'No download url found for model {model_type} {model_id}')
            return

        out_path = os.path.join(download_base_dir, model_file_name)
        download_file(model_url, out_path)

        if config_url:
            out_path = os.path.join(download_base_dir, config_file_name)
            download_file(config_url, out_path)
    except Exception as e:
        log.exception(e)

def resolve_downloaded_model_path(model_type: str, model_id: str, download_base_dir: str=None, subdir_for_model_type=True):
    '''
    Gets the path to the downloaded model file. Returns `None` if a file doesn't exist at the calculated path.

    Args:
    * model_type: str
    * model_id: str
    * download_base_dir: str - base directory inside which the models are downloaded.
              Note: The actual download directory path will include the `model_type` if `subdir_for_model_type` is `True`.
              If `download_base_dir` is `None`, it'll use the `~/.cache/sdkit` folder, where `~` is the home or user-folder.
    * subdir_for_model_type: bool - default True. Saves the downloaded model in a subdirectory (named with the model_type).
              For e.g. if `download_base_dir` is `D:\\models`, then a `stable-diffusion` type model is downloaded to
              `D:\\models\\stable-diffusion`, a `hypernetwork` type model is downloaded to `D:\\models\\hypernetwork` and so on.
    '''
    download_base_dir = get_actual_base_dir(model_type, download_base_dir, subdir_for_model_type)
    _, file_name = get_url_and_filename(model_type, model_id, url_key='url')
    if file_name is None:
        return

    file_path = os.path.join(download_base_dir, file_name)
    return file_path if os.path.exists(file_path) else None

def get_actual_base_dir(model_type, download_base_dir, subdir_for_model_type):
    download_base_dir = os.path.join('~', '.cache', 'sdkit') if download_base_dir is None else download_base_dir
    download_base_dir = os.path.join(download_base_dir, model_type) if subdir_for_model_type else download_base_dir
    return os.path.abspath(download_base_dir)

def get_url_and_filename(model_type, model_id, url_key='url'):
    from sdkit.models import get_model_info_from_db

    model_info = get_model_info_from_db(model_type=model_type, model_id=model_id)
    url = model_info.get(url_key)
    if url is None:
        return None, None

    file_name = os.path.basename(urlparse(url).path)
    return url, file_name
