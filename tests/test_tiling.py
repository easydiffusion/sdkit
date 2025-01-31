from PIL import Image

from sdkit import Context
from sdkit.generate import generate_images
from sdkit.models import load_model


from common import (
    TEST_DATA_FOLDER,
    USE_DIFFUSERS,
    assert_images_same,
)

EXPECTED_DIR = f"{TEST_DATA_FOLDER}/expected_images/tiling"

context = None


def setup_module():
    global context

    context = Context()
    context.test_diffusers = USE_DIFFUSERS
    context.model_paths["stable-diffusion"] = "models/stable-diffusion/1.x/sd-v1-4.ckpt"
    load_model(context, "stable-diffusion")


def test_1_0__tiling_x():
    prompt = "big four-master sailing vessel, stormy waves, at night"
    image = generate_images(context, prompt, seed=42, width=512, height=512, tiling="x")[0]

    expected_image = Image.open(f"{EXPECTED_DIR}/1.4-tiling-x.png")
    assert_images_same(image, expected_image, "tiling_test1.0")


def test_1_1__tiling_xy():
    prompt = "Photorealistic multicolored blobs of paint on a white canvas"
    image = generate_images(context, prompt, seed=265319673, width=768, height=768, tiling="xy")[0]

    expected_image = Image.open(f"{EXPECTED_DIR}/1.4-tiling-768-xy.png")
    assert_images_same(image, expected_image, "tiling_test1.1")
