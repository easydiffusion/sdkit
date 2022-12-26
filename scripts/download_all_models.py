import os
import sys
import argparse
import requests

from sdkit.models import download_model, resolve_downloaded_model_path
from sdkit.models import get_models_db
from sdkit.utils import hash_file_quick

parser = argparse.ArgumentParser()
parser.add_argument('--models-dir', required=True, help="Folder path where the models will be downloaded, with a subdir for each model type")
parser.add_argument('--hash-only', action='store_true', default=False, help="Don't download, just calculate the hashes of the models in the downloaded dir")
args = parser.parse_args()

if len(sys.argv) < 2:
    print('Error: need to provide a folder path as the first argument')
    exit(1)

db = get_models_db()

models_to_download = {model_type: list(models.keys()) for model_type, models in db.items()}
download_dir = args.models_dir

os.makedirs(download_dir, exist_ok=True)

if os.path.exists(download_dir):
    for model_type, model_ids in models_to_download.items():
        for model_id in model_ids:
            if not args.hash_only:
                download_model(model_type, model_id, download_base_dir=download_dir)

            model_path = resolve_downloaded_model_path(model_type=model_type, model_id=model_id, download_base_dir=download_dir)
            quick_hash = hash_file_quick(model_path)
            model_info = db[model_type][model_id]
            expected_quick_hash = model_info['quick_hash']
            expected_size = int(requests.get(model_info['url'], stream=True).headers['content-length'])
            actual_size = os.path.getsize(model_path)

            if quick_hash != expected_quick_hash:
                print(f'''ERROR! {model_type} {model_id}:
  expected hash:\t{expected_quick_hash}
  actual:\t\t{quick_hash}
  expected size:\t{expected_size}
  actual size:\t\t{actual_size}''')
