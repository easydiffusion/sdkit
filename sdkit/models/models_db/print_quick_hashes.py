'''
Utility script for calculating quick hashes for all the entries in the models db.

Usage:
python print_quick_hashes.py
'''
import argparse

from sdkit.utils import hash_url_quick
import stable_diffusion, realesrgan, gfpgan

parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument("--diff-only", action="store_true", help="Only show entries if the calculated quick-hash doesn't match the stored quick-hash")
parser.set_defaults(diff_only=False)
args = parser.parse_args()

all_models = {
    'stable-diffusion': stable_diffusion.models,
    'gfpgan': gfpgan.models,
    'realesrgan': realesrgan.models,
}
hashes_found = {}

if args.diff_only:
    print('Printing quick-hashes for only those URLs that do not match the configured quick-hash')

for model_type, models in all_models.items():
    print(f'{model_type} models:')

    for model_id, model_info in models.items():
        url = model_info['url']
        quick_hash = hash_url_quick(url)
        if not args.diff_only or quick_hash != model_info.get('quick_hash'):
            print(f'{model_id} = {quick_hash}')
        if quick_hash in hashes_found:
            print(f'HASH CONFLICT! {quick_hash} already maps to {hashes_found[quick_hash]}')
        else:
            hashes_found[quick_hash] = url
