from PIL import Image

from sdkit import Context
from sdkit.generate import generate_images
from sdkit.models import load_model
from sdkit import utils


from common import (
    TEST_DATA_FOLDER,
    assert_images_same,
)

EXPECTED_DIR = f"{TEST_DATA_FOLDER}/expected_images/tiling"

context = None

orig_has_amd = None


def setup_module():
    global context

    context = Context()
    context.test_diffusers = True
    context.model_paths["stable-diffusion"] = "models/stable-diffusion/sd-v1-4.ckpt"
    load_model(context, "stable-diffusion")


"""
def setup_function():
    global orig_has_amd

    orig_has_amd = utils.has_amd_gpu


def teardown_function():
    utils.has_amd_gpu = orig_has_amd
"""


def test_1_0__tiling_x():
    prompt="big four-master sailing vessel, stormy waves, at night"
    image = generate_images(context, prompt, seed=42, width=512, height=512, tiling="x")[0]

    expected_image = Image.open(f"{EXPECTED_DIR}/1.4-tiling-x.png")
    assert_images_same(image, expected_image, "tiling_test1.0")
