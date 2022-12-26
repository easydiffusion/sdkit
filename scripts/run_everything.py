'''
Runs all the models in the models db, with all the samplers.
'''
import os
import argparse

from sdkit import Context
from sdkit.models import get_models_db, resolve_downloaded_model_path, load_model
from sdkit.generate import generate_images
from sdkit.generate.sampler import default_samplers, k_samplers
from sdkit.utils import save_images, log

# args
parser = argparse.ArgumentParser()
parser.add_argument('--models-dir', type=str, help="Path to the directory containing all the models, with subdirs for each model-type")
parser.add_argument('--out-dir', type=str, help="Path to the directory to save the generated images")
args = parser.parse_args()

# setup
models_db = get_models_db()
samplers = list(default_samplers.samplers.keys()) + list(k_samplers.samplers.keys())

# run stable diffusion tests
sd_models = models_db['stable-diffusion']
context = Context()

for model_id, model_info in sd_models.items():
    context.model_paths['stable-diffusion'] = resolve_downloaded_model_path(model_type='stable-diffusion', model_id=model_id, download_base_dir=args.models_dir)
    load_model(context, 'stable-diffusion', scan_model=False)

    min_size = model_info['metadata']['min_size']

    out_dir_path = os.path.join(args.out_dir, model_id)

    for sampler_name in samplers:
        log.info(f'Model: {model_id}, Sampler: {sampler_name}')
        images = generate_images(
            context,
            prompt='Photograph of an astronaut riding a horse',
            seed=42,
            width=min_size, height=min_size,
            sampler_name=sampler_name
        )
        save_images(images, dir_path=out_dir_path, file_name=sampler_name)
