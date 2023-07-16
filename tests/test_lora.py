from PIL import Image

from sdkit import Context
from sdkit.generate import generate_images
from sdkit.models import load_model, unload_model
from sdkit.utils import load_tensor_file, download_file, save_tensor_file

import os

from common import (
    TEST_DATA_FOLDER,
    assert_images_same,
)

EXPECTED_DIR = f"{TEST_DATA_FOLDER}/expected_images/lora"

context = None

saved_weights = None


def setup_module():
    global context

    context = Context()
    context.test_diffusers = True


def test_load_sd_1_4():
    global saved_weights

    context.model_paths["stable-diffusion"] = "models/stable-diffusion/sd-v1-4.ckpt"
    load_model(context, "stable-diffusion")


def check_lora_image(expected_image, test_name, lora_alpha=0, prefix=""):
    prompt = (
        prefix
        + "1boy, muscular, full armor, armor, shoulder plates, angry, looking at viewer, spiked hair, white hair,"
    )

    image = generate_images(context, prompt, seed=42, width=512, height=512, lora_alpha=lora_alpha)[0]

    expected_image = Image.open(f"{EXPECTED_DIR}/{expected_image}")
    assert_images_same(image, expected_image, "lora_" + test_name)


# section 0 - no lora
def test_0_0__image_without_lora__sd_1_4():
    check_lora_image(
        expected_image="lora-not-loaded.png",
        test_name="test0.0",
    )


# section 1 - single lora
def test_load_single_lora():
    lora_path = "models/lora/oot-254.safetensors"
    lora_url = "https://civitai.com/api/download/models/23655"

    if not os.path.exists(lora_path):
        download_file(lora_url, lora_path)

    context.model_paths["lora"] = lora_path  # testing a non-array path assignment
    load_model(context, "lora")


def test_1_1__single_a1111_lora__sd_1_4__alpha_0():
    check_lora_image(
        expected_image="lora-not-loaded.png",
        test_name="test1.1",
        lora_alpha=0,
    )


def test_1_2__single_a1111_lora__sd_1_4__alpha_0_5():
    check_lora_image(
        expected_image="lora-single-alpha0.5.png",
        test_name="test1.2",
        lora_alpha=0.5,
    )


def test_1_3__single_a1111_lora__sd_1_4__alpha_1():
    check_lora_image(
        expected_image="lora-single-alpha1.png",
        test_name="test1.3",
        lora_alpha=1,
    )


def test_1_4__single_a1111_lora__sd_1_4__alpha_0_5():
    check_lora_image(
        expected_image="lora-single-alpha0.5.png",
        test_name="test1.4",
        lora_alpha=0.5,
    )


def test_1_5__single_a1111_lora__sd_1_4__alpha_0():
    check_lora_image(
        expected_image="lora-not-loaded.png",
        test_name="test1.5",
        lora_alpha=0,
    )


def test_1_6__single_a1111_lora__sd_1_4__alpha_1():
    check_lora_image(
        expected_image="lora-single-alpha1.png",
        test_name="test1.6",
        lora_alpha=1,
    )


def test_1_7__lora_continues_to_stay_loaded_after_sd_model_reloads():
    # the last test before this one should have a non-zero alpha, to test if the correct alpha is used after reloading SD

    test_load_sd_1_4()

    check_lora_image(
        expected_image="lora-single-alpha0.5.png",
        test_name="test1.7a",
        lora_alpha=0.5,
    )
    check_lora_image(
        expected_image="lora-single-alpha1.png",
        test_name="test1.7b",
        lora_alpha=1,
    )


def test_unload_single_lora():
    unload_model(context, "lora")


def test_1_8__single_a1111_lora__sd_1_4__non_zero_alpha_after_unloading_lora():
    check_lora_image(
        expected_image="lora-not-loaded.png",
        test_name="test1.8",
        lora_alpha=1,
    )


# section 2 - multiple lora
def test_2_0__multiple_a1111_lora__sd_1_4__without_lora():
    check_lora_image(
        expected_image="lora-two-not-loaded.png",
        test_name="test2.0",
        lora_alpha=[0, 0],
        prefix="inkSketch ",
    )


def test_load_multiple_lora():
    loras = [
        ("models/lora/oot-254.safetensors", "https://civitai.com/api/download/models/23655"),
        ("models/lora/inkSketch_V1.5.safetensors", "https://civitai.com/api/download/models/105284"),
    ]

    for lora_path, lora_url in loras:
        if not os.path.exists(lora_path):
            download_file(lora_url, lora_path)

    context.model_paths["lora"] = [path for path, _ in loras]  # testing an array path assignment
    load_model(context, "lora")


def test_2_1__multiple_a1111_lora__sd_1_4__alpha_half_0():
    check_lora_image(
        expected_image="lora-two-alpha0.5_0.png",
        test_name="test2.1",
        lora_alpha=[0.5, 0],
        prefix="inkSketch ",
    )


def test_2_2__multiple_a1111_lora__sd_1_4__alpha_half_half():
    check_lora_image(
        expected_image="lora-two-alpha0.5_0.5.png",
        test_name="test2.2",
        lora_alpha=[0.5, 0.5],
        prefix="inkSketch ",
    )


def test_2_3__multiple_a1111_lora__sd_1_4__alpha_0_1():
    check_lora_image(
        expected_image="lora-two-alpha0_0.5.png",
        test_name="test2.3",
        lora_alpha=[0, 0.5],
        prefix="inkSketch ",
    )


def test_2_4__multiple_a1111_lora__sd_1_4__alpha_0_0():
    check_lora_image(
        expected_image="lora-two-not-loaded.png",
        test_name="test2.4",
        lora_alpha=[0, 0],
        prefix="inkSketch ",
    )


# section 3 - misc
def vram_usage_level_test(vram_usage_level, test_name):
    unload_model(context, "stable-diffusion")
    unload_model(context, "lora")

    context.vram_usage_level = vram_usage_level

    test_load_sd_1_4()
    test_load_single_lora()

    check_lora_image(
        expected_image="lora-single-alpha0.5.png",
        test_name=test_name + "a",
        lora_alpha=0.5,
    )
    check_lora_image(
        expected_image="lora-single-alpha1.png",
        test_name=test_name + "b",
        lora_alpha=1,
    )

    # reload SD
    test_load_sd_1_4()
    check_lora_image(
        expected_image="lora-single-alpha1.png",
        test_name=test_name + "c",
        lora_alpha=1,
    )


def test_3_1__vram_mode__works_on_low():
    vram_usage_level_test(vram_usage_level="low", test_name="test3.1")


def test_3_2__vram_mode__works_on_balanced():
    vram_usage_level_test(vram_usage_level="balanced", test_name="test3.2")


def test_3_3__vram_mode__works_on_high():
    vram_usage_level_test(vram_usage_level="high", test_name="test3.3")
