import argparse
import os
from collections import namedtuple

parser = argparse.ArgumentParser()
parser.add_argument(
    "--models-dir", type=str, required=True, help="Path to the directory containing the Stable Diffusion models"
)
args = parser.parse_args()

from sdkit import Context
from sdkit.generate import generate_images
from sdkit.models import load_model
from sdkit.utils import get_device_usage, log

DeviceUsage = namedtuple("DeviceUsage", ["cpu_used", "ram_used", "ram_total", "vram_used", "vram_total", "vram_peak"])

c = Context()

log.info("Starting..")
usage_start = DeviceUsage(*get_device_usage(c.device, log_info=True))

c.model_paths["stable-diffusion"] = os.path.join(args.models_dir, "1.x", "sd-v1-4.ckpt")
load_model(c, "stable-diffusion")

log.info("Loaded the model..")
usage_model_load = DeviceUsage(*get_device_usage(c.device, log_info=True))

try:
    images = generate_images(c, prompt="Photograph of an astronaut riding a horse")
except Exception as e:
    log.exception(e)

log.info("Generated the image..")
usage_after_render = DeviceUsage(*get_device_usage(c.device, log_info=True))

print("")
log.info(
    f"VRAM trend: {usage_start.vram_used:.1f} (start) GiB to {usage_model_load.vram_used:.1f} GiB (before render) to {usage_after_render.vram_used:.1f} GiB (after render)"
)
print("")

max_expected_vram = usage_model_load.vram_used + 0.3
if usage_after_render.vram_used > max_expected_vram:
    log.error(
        f"Test failed! VRAM after render was expected to be below {max_expected_vram:.1f} GiB, but was {usage_after_render.vram_used:.1f} GiB!"
    )
    exit(1)
else:
    log.info("Test passed!")
