'''
Runs the desired Stable Diffusion models against the desired samplers. Saves the output images
to disk, along with the peak RAM and VRAM usage, as well as the sampling performance.
'''
import os
import time
import argparse
from sdkit.utils import log

# args
parser = argparse.ArgumentParser()
parser.add_argument('--models-dir', type=str, required=True, help="Path to the directory containing the Stable Diffusion models")
parser.add_argument('--out-dir', type=str, required=True, help="Path to the directory to save the generated images and test results")
parser.add_argument('--models', default='all', help="Comma-separated list of model filenames (without spaces) to test. Default: all")
parser.add_argument('--exclude-models', default=set(), help="Comma-separated list of model filenames (without spaces) to skip")
parser.add_argument('--samplers', default='all', help="Comma-separated list of sampler names (without spaces) to test. Default: all")
parser.add_argument('--exclude-samplers', default=set(), help="Comma-separated list of sampler names (without spaces) to skip")
parser.add_argument('--vram-usage-levels', default='balanced', help="Comma-separated list of VRAM usage levels. Allowed values: low, balanced, high")
parser.add_argument('--skip-completed', default=False, help="Skips a model or sampler if it has already been tested (i.e. an output image exists for it)")
parser.add_argument('--width', default=-1, type=int, help="Specify the image width. Defaults to 512 or 768 (if the model requires 768)")
parser.add_argument('--height', default=-1, type=int, help="Specify the image height. Defaults to 512 or 768 (if the model requires 768)")
parser.add_argument('--live-perf', action="store_true", help="Print the RAM and VRAM usage stats every few seconds")
parser.set_defaults(live_perf=False)
args = parser.parse_args()

if args.models != 'all': args.models = set(args.models.split(','))
if args.exclude_models != set(): args.exclude_models = set(args.exclude_models.split(','))
if args.samplers != 'all': args.samplers = set(args.samplers.split(','))
if args.exclude_samplers != set(): args.exclude_samplers = set(args.exclude_samplers.split(','))

# setup
log.info('Starting..')
from sdkit.models import load_model
from sdkit.generate.sampler import default_samplers, k_samplers

sd_models = set([f for f in os.listdir(args.models_dir) if os.path.splitext(f)[1] in ('.ckpt', '.safetensors')])
all_samplers = set(default_samplers.samplers.keys()) | set(k_samplers.samplers.keys())
args.vram_usage_levels = args.vram_usage_levels.split(',')

models_to_test = sd_models if args.models == 'all' else args.models
models_to_test -= args.exclude_models
samplers_to_test = all_samplers if args.samplers == 'all' else args.samplers
samplers_to_test -= args.exclude_samplers
vram_usage_levels_to_test = args.vram_usage_levels

# setup the test
from sdkit import Context
from sdkit.generate import generate_images
from sdkit.models import get_model_info_from_db
from sdkit.utils import hash_file_quick

VRAM_USAGE_LEVEL_TO_OPTIMIZATIONS = {
    'balanced': {'KEEP_FS_AND_CS_IN_CPU', 'SET_ATTENTION_STEP_TO_4'},
    'low': {'KEEP_ENTIRE_MODEL_IN_CPU'},
    'high': {},
}
perf_results = [['model_filename', 'vram_usage_level', 'sampler_name', 'max_ram (GB)', 'max_vram (GB)', 'image_size', 'time_taken (s)', 'speed (it/s)', 'test_status']]
perf_results_file = f'perf_results_{time.time()}.csv'

# print test info
log.info('---')
log.info(f'Models actually being tested: {models_to_test}')
log.info(f'Samplers actually being tested: {samplers_to_test}')
log.info(f'VRAM usage levels being tested: {vram_usage_levels_to_test}')
log.info('---')
log.info(f'Available models: {sd_models}')
log.info(f'Available samplers: {all_samplers}')
log.info('---')

# run the test
def run_test():
    for model_filename in models_to_test:
        model_dir_path = os.path.join(args.out_dir, model_filename)

        if args.skip_completed and is_model_already_tested(model_dir_path):
            log.info(f'skipping model {model_filename} since it has already been processed at {model_dir_path}')
            continue

        for vram_usage_level in vram_usage_levels_to_test:
            context = Context()
            context.vram_optimizations = VRAM_USAGE_LEVEL_TO_OPTIMIZATIONS[vram_usage_level]

            # setup the model
            out_dir_path = os.path.join(model_dir_path, vram_usage_level)
            os.makedirs(out_dir_path, exist_ok=True)

            try:
                context.model_paths['stable-diffusion'] = os.path.join(args.models_dir, model_filename)
                load_model(context, 'stable-diffusion', scan_model=False)
            except Exception as e:
                log.exception(e)
                perf_results.append([model_filename, vram_usage_level, 'n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'error'])
                log_perf_results()
                continue

            min_size = get_min_size(context.model_paths['stable-diffusion'])
            width = min_size if args.width == -1 else args.width
            height = min_size if args.height == -1 else args.height

            # run a warm-up, before running the actual samplers
            log.info('Warming up..')
            try:
                generate_images(context, prompt='Photograph of an astronaut riding a horse', num_inference_steps=10, seed=42, width=width, height=height, sampler_name='euler_a')
            except:
                pass

            # run the actual test
            run_samplers(context, model_filename, out_dir_path, width, height, vram_usage_level)

            del context

def run_samplers(context, model_filename, out_dir_path, width, height, vram_usage_level):
    from threading import Thread, Event
    from queue import Queue

    for sampler_name in samplers_to_test:
        # setup
        img_path = os.path.join(out_dir_path, f'{sampler_name}_0.jpeg')
        if args.skip_completed and os.path.exists(img_path):
            log.info(f'skipping sampler {sampler_name} since it has already been processed at {img_path}')
            continue

        log.info(f'Model: {model_filename}, Sampler: {sampler_name}')

        # start profiling
        prof_thread_stop_event = Event()
        ram_usage = Queue()
        vram_usage = Queue()
        prof_thread = Thread(target=profiling_thread, kwargs={
            'device': context.device,
            'prof_thread_stop_event': prof_thread_stop_event,
            'ram_usage': ram_usage,
            'vram_usage': vram_usage,
        })
        prof_thread.start()

        t, speed = time.time(), 0

        # start rendering
        try:
            images = generate_images(
                context,
                prompt='Photograph of an astronaut riding a horse',
                seed=42,
                num_inference_steps=25,
                width=width, height=height,
                sampler_name=sampler_name
            )
            t = time.time() - t
            speed = 25 / t
            test_status = 'success'

            images[0].save(img_path)
        except Exception as e:
            test_status = 'error'
            log.exception(e)
            t = 0
        finally:
            # stop profiling
            prof_thread_stop_event.set()
            prof_thread.join()

        perf_results.append([model_filename, vram_usage_level, sampler_name, f'{max(ram_usage.queue):.1f}', f'{max(vram_usage.queue):.1f}', f'{width}x{height}', f'{t:.1f}', f'{speed:.1f}', test_status])

        log_perf_results()

def profiling_thread(device, prof_thread_stop_event, ram_usage, vram_usage):
    import torch
    import time
    import psutil

    while not prof_thread_stop_event.is_set():
        ram_used = psutil.Process().memory_info().rss / 10**9
        vram_free, vram_total = torch.cuda.mem_get_info(device)
        vram_used = (vram_total - vram_free) / 10**9

        ram_usage.put(ram_used)
        vram_usage.put(vram_used)

        if args.live_perf:
            log.info(f'System RAM (used): {ram_used} GB')
            log.info(f'GPU RAM (used): {vram_used} GB')

        time.sleep(1)

def is_model_already_tested(out_dir_path):
    if not os.path.exists(out_dir_path):
        return False

    sampler_files = list(map(lambda x: os.path.join(out_dir_path, f'{x}_0.jpeg'), samplers_to_test))
    images_exist = list(map(lambda x: os.path.exists(x), sampler_files))
    return all(images_exist)

def get_min_size(model_path, default_size=512):
    model_info = get_model_info_from_db(quick_hash=hash_file_quick(model_path))

    return model_info['metadata']['min_size'] if model_info is not None else default_size

def log_perf_results():
    print('\n-- Performance summary --')
    import pandas as pd
    df = pd.DataFrame(data=perf_results)
    df = df.rename(columns=df.iloc[0]).drop(df.index[0])
    print(df)
    print('')

    df = pd.DataFrame(data=perf_results)
    out_file = os.path.join(args.out_dir, perf_results_file)
    df.to_csv(out_file, header=False, index=False)

    print(f'Written performance summary to {out_file}\n')

run_test()
