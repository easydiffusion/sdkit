from PIL import Image

from sdkit import Context
from sdkit.generate import generate_images
from sdkit.models import load_model
from sdkit.utils import load_tensor_file

import os

from common import (
    TEST_DATA_FOLDER,
    assert_images_same,
)

EXPECTED_DIR = f"{TEST_DATA_FOLDER}/expected_images/lora"

context = None


def setup_module():
    global context

    context = Context()
    context.test_diffusers = True


def test_load_sd_1_4():
    context.model_paths["stable-diffusion"] = "models/stable-diffusion/sd-v1-4.ckpt"
    load_model(context, "stable-diffusion")


# section 1 - SD at different resolutions and samplers
def check_lora_image(lora_alpha, expected_image, test_name):
    image = generate_images(
        context,
        "1boy, muscular, full armor, armor, shoulder plates, angry, looking at viewer, spiked hair, white hair,",
        seed=42,
        width=512,
        height=512,
        lora_alpha=lora_alpha,
    )[0]

    expected_image = Image.open(f"{EXPECTED_DIR}/{expected_image}")
    assert_images_same(image, expected_image, test_name)


def test_1_0__image_without_lora__sd_1_4():
    check_lora_image(
        lora_alpha=0,
        expected_image="lora-not-loaded.png",
        test_name="test1.0",
    )


def test_load_single_lora():
    lora_path = "models/lora/oot-254.safetensors"

    if not os.path.exists(lora_path):
        raise FileNotFoundError(
            f"LoRA model not found. Please download from https://civitai.com/api/download/models/23655 to {lora_path}"
        )

    model = context.models["stable-diffusion"]
    pipe = model["default"]

    lora = load_tensor_file(lora_path)
    pipe.load_lora_weights(lora)

    model["_lora_loaded"] = True


def test_1_1__single_a1111_lora__sd_1_4__alpha_0():
    check_lora_image(
        lora_alpha=0,
        expected_image="lora-single-alpha0.png",
        test_name="test1.1",
    )


def test_1_2__single_a1111_lora__sd_1_4__alpha_0_5():
    check_lora_image(
        lora_alpha=0.5,
        expected_image="lora-single-alpha0.5.png",
        test_name="test1.2",
    )


def test_1_3__single_a1111_lora__sd_1_4__alpha_1():
    check_lora_image(
        lora_alpha=1,
        expected_image="lora-single-alpha1.png",
        test_name="test1.3",
    )


def test_1_4__single_a1111_lora__sd_1_4__alpha_0_5():
    check_lora_image(
        lora_alpha=0.5,
        expected_image="lora-single-alpha0.5.png",
        test_name="test1.4",
    )


def test_1_5__single_a1111_lora__sd_1_4__alpha_0():
    check_lora_image(
        lora_alpha=0,
        expected_image="lora-single-alpha0.png",
        test_name="test1.5",
    )
