'''
Utility script for calculating quick hashes for all the entries in the models db.

Usage:
python print_quick_hashes.py
'''
from sdkit.utils import hash_url_quick
import stable_diffusion, realesrgan, gfpgan

all_models = {
    'stable-diffusion': stable_diffusion.models,
    'gfpgan': gfpgan.models,
    'realesrgan': realesrgan.models,
}

print('Printing quick-hashes for only those URLs that do not match the configured quick-hash')

for model_type, models in all_models.items():
    print(f'{model_type} models:')

    for model_id, model_info in models.items():
        url = model_info['url']
        quick_hash = hash_url_quick(url)
        if quick_hash != model_info.get('quick_hash'):
            print(f'{model_id} = {quick_hash}')
