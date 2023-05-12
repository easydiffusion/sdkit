"""
Runs the desired Stable Diffusion models against the desired samplers. Saves the output images
to disk, along with the peak RAM and VRAM usage, as well as the sampling performance.

Mandatory arguments: `--models-dir` and `--out-dir`

Example:
`python path/to/run_everything.py --models-dir /path/to/models/stable-diffusion --out-dir /path/to/output-dir`

By default, these arguments will run all the samplers against all the model files found inside the `--models-dir` directory.

To specify the models and/or samplers to use, specify a comma-separated list of models and/or samplers. For e.g. `--models sd-v1-4.ckpt,512-base-ema.safetensors` and/or `--samplers plms,euler_a,ddim`

You can also set a comma-separated list of VRAM usage levels to run. By default it'll run against `balanced`, but you can specify `--vram-usage-levels low,balanced,high` to run against all three.

There are a few more options (`--prompt`, `--seed`, `--init-image` etc), which you can explore using `--help`).
"""
import argparse
import os
import time

from sdkit.utils import get_device_usage, log

# args
parser = argparse.ArgumentParser()
parser.add_argument(
    "--prompt",
    type=str,
    default="Photograph of an astronaut riding a horse",
    help="Prompt to use for generating the image",
)
parser.add_argument("--seed", type=int, default=42, help="Seed to use for generating the image")
parser.add_argument(
    "--models-dir", type=str, required=True, help="Path to the directory containing the Stable Diffusion models"
)
parser.add_argument(
    "--out-dir", type=str, required=True, help="Path to the directory to save the generated images and test results"
)
parser.add_argument(
    "--models", default="all", help="Comma-separated list of model filenames (without spaces) to test. Default: all"
)
parser.add_argument(
    "--exclude-models",
    default=set(),
    help="Comma-separated list of model filenames (without spaces) to skip. Supports wildcards (without commas), for e.g. --exclude-models *.safetensors, or --exclude-models sd-1-4*",
)
parser.add_argument(
    "--samplers", default="all", help="Comma-separated list of sampler names (without spaces) to test. Default: all"
)
parser.add_argument(
    "--exclude-samplers", default=set(), help="Comma-separated list of sampler names (without spaces) to skip"
)
parser.add_argument(
    "--init-image", default=None, help="Path to an initial image to use. Only works with DDIM sampler (for now)."
)
parser.add_argument(
    "--vram-usage-levels",
    default="balanced",
    help="Comma-separated list of VRAM usage levels. Allowed values: low, balanced, high",
)
parser.add_argument(
    "--skip-completed",
    default=False,
    help="Skips a model or sampler if it has already been tested (i.e. an output image exists for it)",
)
parser.add_argument("--steps", default=25, type=int, help="Number of inference steps to run for each sampler")
parser.add_argument(
    "--sizes",
    default="auto",
    type=str,
    help="Comma-separated list of image sizes (width x height). No spaces. E.g. 512x512 or 512x512,1024x768. Defaults to what the model needs (512x512 or 768x768, if the model requires 768)",
)
parser.add_argument(
    "--device", default="cuda:0", type=str, help="Specify the device to run on. E.g. cpu or cuda:0 or cuda:1 etc"
)
parser.add_argument("--max-vram", default=None, type=float, help="Max VRAM (in GiB) this process is allowed to use")
parser.add_argument("--live-perf", action="store_true", help="Print the RAM and VRAM usage stats every few seconds")
parser.add_argument("--diffusers", action="store_true", help="Use the new diffusers backend")
parser.set_defaults(live_perf=False)
parser.set_defaults(diffusers=False)
args = parser.parse_args()

if args.models != "all":
    args.models = set(args.models.split(","))
if args.exclude_models != set() and "*" not in args.exclude_models:
    args.exclude_models = set(args.exclude_models.split(","))
if args.samplers != "all":
    args.samplers = set(args.samplers.split(","))
if args.exclude_samplers != set():
    args.exclude_samplers = set(args.exclude_samplers.split(","))
if args.sizes != "auto":
    args.sizes = [tuple(map(lambda x: int(x), size.split("x"))) for size in args.sizes.split(",")]

# setup
log.info("Starting..")
from sdkit.generate.sampler import default_samplers, k_samplers, unipc_samplers
from sdkit.models import load_model

sd_models = set([f for f in os.listdir(args.models_dir) if os.path.splitext(f)[1] in (".ckpt", ".safetensors")])
all_samplers = (
    set(default_samplers.samplers.keys()) | set(k_samplers.samplers.keys()) | set(unipc_samplers.samplers.keys())
)
args.vram_usage_levels = args.vram_usage_levels.split(",")

if isinstance(args.exclude_models, str) and "*" in args.exclude_models:
    import fnmatch

    args.exclude_models = set(fnmatch.filter(sd_models, args.exclude_models))

models_to_test = sd_models if args.models == "all" else args.models
models_to_test -= args.exclude_models
samplers_to_test = all_samplers if args.samplers == "all" else args.samplers
samplers_to_test -= args.exclude_samplers
vram_usage_levels_to_test = args.vram_usage_levels

if args.init_image is not None:
    if not os.path.exists(args.init_image):
        log.error(f"Error! Could not an initial image at the path specified: {args.init_image}")
        exit(1)

    if samplers_to_test != {"ddim"} and not args.diffusers:
        log.error('We only support the "ddim" sampler for img2img right now!')
        exit(1)

    all_samplers = {"ddim"}

# setup the test
import torch
from sdkit import Context
from sdkit.generate import generate_images
from sdkit.models import get_model_info_from_db
from sdkit.utils import hash_file_quick


def restrict():
    if args.max_vram is not None:
        device_usage = get_device_usage(args.device)
        total_vram = device_usage[4]
        vram_fraction = max(args.max_vram, 0) / total_vram

        log.info(
            f"Restricting process VRAM usage to {args.max_vram} GiB, out of the total {total_vram:.1f} GiB on {args.device}"
        )
        torch.cuda.set_per_process_memory_fraction(vram_fraction, device=args.device)


restrict()

perf_results = [
    [
        "model_filename",
        "model_load_time",
        "vram_usage_level",
        "sampler_name",
        "max_ram (GB)",
        "max_vram (GB)",
        "image_size",
        "time_taken (s)",
        "speed (it/s)",
        "render_test",
        "ram_usage",
        "vram_usage",
    ]
]
perf_results_file = f"perf_results_{time.time()}.csv"

# print test info
log.info("---")
log.info(f"Models actually being tested: {models_to_test}")
log.info(f"Samplers actually being tested: {samplers_to_test}")
log.info(f"VRAM usage levels being tested: {vram_usage_levels_to_test}")
log.info(f"Image sizes being tested: {args.sizes}")
log.info("---")
log.info(f"Available models: {sd_models}")
log.info(f"Available samplers: {all_samplers}")
log.info("---")

model_load_time = 0


# run the test
def run_test():
    global model_load_time

    for model_filename in models_to_test:
        model_dir_path = os.path.join(args.out_dir, model_filename)

        if args.skip_completed and is_model_already_tested(model_dir_path):
            log.info(f"skipping model {model_filename} since it has already been processed at {model_dir_path}")
            continue

        for vram_usage_level in vram_usage_levels_to_test:
            context = Context()
            context.device = args.device
            context.vram_usage_level = vram_usage_level
            context.test_diffusers = args.diffusers

            # setup the model
            out_dir_path = os.path.join(model_dir_path, vram_usage_level)
            os.makedirs(out_dir_path, exist_ok=True)

            model_load_time = 0

            try:
                context.model_paths["stable-diffusion"] = os.path.join(args.models_dir, model_filename)
                t = time.time()
                load_model(context, "stable-diffusion", scan_model=False)
                model_load_time = time.time() - t
            except Exception as e:
                log.exception(e)
                perf_results.append(
                    [
                        model_filename,
                        model_load_time,
                        vram_usage_level,
                        "n/a",
                        "n/a",
                        "n/a",
                        "n/a",
                        "n/a",
                        "n/a",
                        False,
                        [],
                        [],
                    ]
                )
                log_perf_results()
                continue

            # run a warm-up, before running the actual samplers
            log.info("Warming up..")
            try:
                generate_images(
                    context,
                    prompt="Photograph of an astronaut riding a horse",
                    num_inference_steps=4,
                    seed=42,
                    width=512,
                    height=512,
                    sampler_name="plms",
                )
            except:
                pass

            if args.sizes == "auto":
                min_size = get_min_size(context.model_paths["stable-diffusion"])
                sizes = [(min_size, min_size)]
            else:
                sizes = args.sizes

            # run the actual test
            for width, height in sizes:
                run_samplers(context, model_filename, out_dir_path, width, height, vram_usage_level)

            del context


def run_samplers(context, model_filename, out_dir_path, width, height, vram_usage_level):
    from queue import Queue
    from threading import Event, Thread

    for sampler_name in samplers_to_test:
        # setup
        img_path = os.path.join(out_dir_path, f"{sampler_name}_0.jpeg")
        if args.skip_completed and os.path.exists(img_path):
            log.info(f"skipping sampler {sampler_name} since it has already been processed at {img_path}")
            continue

        log.info(
            f"Model: {model_filename}, Sampler: {sampler_name}, Size: {width}x{height}, VRAM Usage Level: {vram_usage_level}"
        )

        # start profiling
        if "cuda" in args.device:
            torch.cuda.reset_peak_memory_stats(args.device)

        prof_thread_stop_event = Event()
        ram_usage = Queue()
        vram_usage = Queue()
        prof_thread = Thread(
            target=profiling_thread,
            kwargs={
                "device": context.device,
                "prof_thread_stop_event": prof_thread_stop_event,
                "ram_usage": ram_usage,
                "vram_usage": vram_usage,
            },
        )
        prof_thread.start()

        t, speed = time.time(), 0

        # start rendering
        try:
            images = generate_images(
                context,
                prompt=args.prompt,
                seed=args.seed,
                num_inference_steps=args.steps,
                width=width,
                height=height,
                sampler_name=sampler_name,
                init_image=args.init_image,
            )
            t = time.time() - t
            speed = args.steps / t
            render_success = True

            images[0].save(img_path)

            if not images[0].getbbox():
                log.error("Image is fully black! Precision issue!")
                render_success = False
        except Exception as e:
            render_success = False
            log.exception(e)
            t = 0
        finally:
            # stop profiling
            prof_thread_stop_event.set()
            prof_thread.join()

        vram_peak = get_device_usage(context.device)[5]

        perf_results.append(
            [
                model_filename,
                f"{model_load_time:.1f}",
                vram_usage_level,
                sampler_name,
                f"{max(ram_usage.queue):.1f}",
                f"{vram_peak:.1f}",
                f"{width}x{height}",
                f"{t:.1f}",
                f"{speed:.1f}",
                render_success,
                list(ram_usage.queue),
                list(vram_usage.queue),
            ]
        )

        log_perf_results()


def profiling_thread(device, prof_thread_stop_event, ram_usage, vram_usage):
    import time

    while not prof_thread_stop_event.is_set():
        cpu_used, ram_used, ram_total, vram_used, vram_total, vram_peak = get_device_usage(
            device, log_info=args.live_perf
        )

        ram_usage.put(ram_used)
        vram_usage.put(vram_used)

        time.sleep(0.3)


def is_model_already_tested(out_dir_path):
    if not os.path.exists(out_dir_path):
        return False

    sampler_files = list(map(lambda x: os.path.join(out_dir_path, f"{x}_0.jpeg"), samplers_to_test))
    images_exist = list(map(lambda x: os.path.exists(x), sampler_files))
    return all(images_exist)


def get_min_size(model_path, default_size=512):
    model_info = get_model_info_from_db(quick_hash=hash_file_quick(model_path))

    return model_info["metadata"]["min_size"] if model_info is not None else default_size


def log_perf_results():
    from importlib.metadata import version

    import numpy as np
    import pandas as pd

    pd.set_option("display.max_rows", 1000)

    print("\n-- Performance summary --")
    print(f"sdkit version: {version('sdkit')}")
    print(f"stable-diffusion-sdkit version: {version('stable-diffusion-sdkit')}")
    print(f"torch version: {version('torch')}")
    if args.diffusers:
        print(f"diffusers version: {version('diffusers')}")
    print(f"Device: {args.device}")
    print(f"Num inference steps: {args.steps}")
    print("")

    df = pd.DataFrame(data=perf_results)
    df = df.rename(columns=df.iloc[0]).drop(df.index[0])
    df = df.sort_values(by=["image_size", "model_filename"], ascending=False)
    df = df.reset_index(drop=True)

    df["render_test"] = df["render_test"].apply(lambda is_pass: "pass" if is_pass else "FAIL")

    out_file = os.path.join(args.out_dir, perf_results_file)
    df.to_csv(out_file, index=False)

    # print the summary
    del df["vram_usage"]
    del df["ram_usage"]

    print(df)
    print("")

    print(f"Written the performance summary to {out_file}\n")


run_test()
