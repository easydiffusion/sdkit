from PIL import Image

from sdkit import Context
from sdkit.generate import generate_images
from sdkit.models import load_model
from sdkit import utils


from common import (
    TEST_DATA_FOLDER,
    assert_images_same,
)

EXPECTED_DIR = f"{TEST_DATA_FOLDER}/expected_images/stable-diffusion"

context = None

orig_has_amd = None


def setup_module():
    global context

    context = Context()
    context.test_diffusers = True
    context.model_paths["stable-diffusion"] = "models/stable-diffusion/sd-v1-4.ckpt"


def setup_function():
    global orig_has_amd

    orig_has_amd = utils.has_amd_gpu


def teardown_function():
    utils.has_amd_gpu = orig_has_amd


def test_1_0__directml__stable_diffusion_1_4_txt2img_works__64x64():
    utils.has_amd_gpu = lambda: True
    context.device = "cpu"

    load_model(context, "stable-diffusion")

    image = generate_images(context, "Photograph of an astronaut riding a horse", seed=42, width=64, height=64)[0]

    expected_image = Image.open(f"{EXPECTED_DIR}/1.4-txt-euler_a-42-64x64-cpu.png")
    assert_images_same(image, expected_image, "directml_test1.0")


def test_1_1__tensorRT__stable_diffusion_1_4_txt2img_works__64x64():
    context.device = "cuda:0"
    load_model(context, "stable-diffusion", convert_to_tensorrt=True)

    image = generate_images(context, "Photograph of an astronaut riding a horse", seed=42, width=64, height=64)[0]

    expected_image = Image.open(f"{EXPECTED_DIR}/1.4-txt-euler_a-42-64x64-cuda.png")
    assert_images_same(image, expected_image, "tensorRT_test1.1")
