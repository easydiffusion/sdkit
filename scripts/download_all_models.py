import os
import sys

from sdkit.models import download_models
from sdkit.models import get_models_db

if len(sys.argv) < 2:
    print('Error: need to provide a folder path as the first argument')
    exit(1)

db = get_models_db()

models_to_download = {model_type: list(models.keys()) for model_type, models in db.items()}
download_dir = sys.argv[1]

os.makedirs(download_dir, exist_ok=True)

if os.path.exists(download_dir):
    download_models(models=models_to_download, download_base_dir=download_dir)
