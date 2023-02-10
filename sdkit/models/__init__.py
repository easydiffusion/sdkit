"""
    This module contains functions for loading and
     unloading models.
"""
from sdkit.models.model_loader import (
    load_model,
    unload_model,
)

from sdkit.models.models_db import (
    get_model_info_from_db,
    get_models_db,
)

from sdkit.models.model_downloader import (
    download_model,
    download_models,
    resolve_downloaded_model_path,
)

from sdkit.models.scan_models import scan_model

__all__ = [
    'download_model',
    'download_models',
    'get_model_info_from_db',
    'get_models_db',
    'load_model',
    'resolve_downloaded_model_path',
    'scan_model',
    'unload_model',
]
