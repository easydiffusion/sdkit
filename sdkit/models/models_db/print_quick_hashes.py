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

for model_type, models in all_models.items():
    print(f'{model_type} models:')

    for model_id, model_info in models.items():
        url = model_info['url']
        quick_hash = hash_url_quick(url)
        print(f'{model_id} = {quick_hash}')
