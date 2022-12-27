'''
Runs all the models in the models db, with all the samplers.
'''
import os
import argparse

# args
parser = argparse.ArgumentParser()
parser.add_argument('--models-dir', type=str, required=True, help="Path to the directory containing all the models, with subdirs for each model-type")
parser.add_argument('--out-dir', type=str, required=True, help="Path to the directory to save the generated images")
parser.add_argument('--skip-models', type=str, default=None, help="Comma-separated list of model ids (without spaces) to skip")
args = parser.parse_args()

# setup
from sdkit import Context
from sdkit.models import get_models_db, resolve_downloaded_model_path, load_model
from sdkit.generate import generate_images
from sdkit.generate.sampler import default_samplers, k_samplers
from sdkit.utils import log

models_db = get_models_db()
samplers = list(default_samplers.samplers.keys()) + list(k_samplers.samplers.keys())

skip_models = []
if args.skip_models is not None:
    skip_models = args.skip_models.split(',')

# run stable diffusion tests
sd_models = models_db['stable-diffusion']
context = Context()

for model_id, model_info in sd_models.items():
    if model_id in skip_models:
        log.info(f'skipping {model_id} as requested')
        continue

    out_dir_path = os.path.join(args.out_dir, model_id)
    # check if this model should be skipped (already done)
    if os.path.exists(out_dir_path):
        skip = True
        for sampler_name in samplers:
            img_path = os.path.join(out_dir_path, f'{sampler_name}_0.jpeg')
            if not os.path.exists(img_path):
                skip = False
                break
        if skip:
            log.info(f'skipping {model_id}, since images for all the samplers already exist in {out_dir_path}')
            continue

    context.model_paths['stable-diffusion'] = resolve_downloaded_model_path(model_type='stable-diffusion', model_id=model_id, download_base_dir=args.models_dir)
    load_model(context, 'stable-diffusion', scan_model=False)
    os.makedirs(out_dir_path)

    min_size = model_info['metadata']['min_size']

    for sampler_name in samplers:
        img_path = os.path.join(out_dir_path, f'{sampler_name}_0.jpeg')
        if os.path.exists(img_path):
            log.info(f'Skipping sampler {sampler_name} since it has already been processed at {img_path}')
            continue

        log.info(f'Model: {model_id}, Sampler: {sampler_name}')
        images = generate_images(
            context,
            prompt='Photograph of an astronaut riding a horse',
            seed=42,
            width=min_size, height=min_size,
            sampler_name=sampler_name
        )
        images[0].save(img_path)
