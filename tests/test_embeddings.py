from PIL import Image

from sdkit import Context
from sdkit.generate import generate_images
from sdkit.models import load_model, unload_model
from sdkit.utils import load_tensor_file, download_file, save_tensor_file

import os

from common import (
    TEST_DATA_FOLDER,
    USE_DIFFUSERS,
    assert_images_same,
)

EXPECTED_DIR = f"{TEST_DATA_FOLDER}/expected_images/embeddings"

context = None


def setup_module():
    global context

    context = Context()
    context.test_diffusers = USE_DIFFUSERS


def test_load_sd_1_4():
    context.model_paths["stable-diffusion"] = "models/stable-diffusion/1.x/sd-v1-4.ckpt"
    load_model(context, "stable-diffusion")


def check_embedding_image(expected_image, test_name, prefix=""):
    prefix += " " if prefix != "" and not prefix.endswith(" ") else ""
    prompt = (
        prefix
        + "1boy, muscular, full armor, armor, shoulder plates, angry, looking at viewer, spiked hair, white hair,"
    )

    image = generate_images(context, prompt, seed=43, width=512, height=512)[0]

    expected_image = Image.open(f"{EXPECTED_DIR}/{expected_image}")
    assert_images_same(image, expected_image, "embedding_" + test_name)


# section 0 - no embedding
def test_0_0__image_without_embedding__sd_1_4__with_trigger_word():
    check_embedding_image(
        expected_image="no-embedding.png",
        test_name="test0.0",
        prefix="charturnerv2",
    )


# section 1 - single embedding
def test_load_single_embedding():
    embedding_path = "models/embeddings/charturnerv2.pt"
    embedding_url = "https://civitai.com/api/download/models/8387"

    if not os.path.exists(embedding_path):
        download_file(embedding_url, embedding_path)

    context.model_paths["embeddings"] = embedding_path  # testing a non-array path assignment
    load_model(context, "embeddings")


def test_1_1__single_a1111_embedding__sd_1_4__with_trigger_word():
    check_embedding_image(
        expected_image="embedding.png",
        test_name="test1.1",
        prefix="charturnerv2",
    )


def test_1_2__single_a1111_embedding__sd_1_4__with_trigger_word_again():
    check_embedding_image(
        expected_image="embedding.png",
        test_name="test1.2",
        prefix="charturnerv2",
    )


def test_1_3__embedding_continues_to_stay_loaded_after_sd_model_reloads():
    test_load_sd_1_4()

    check_embedding_image(
        expected_image="embedding.png",
        test_name="test1.3a",
        prefix="charturnerv2",
    )
    check_embedding_image(
        expected_image="embedding.png",
        test_name="test1.3b",
        prefix="charturnerv2",
    )


def test_unload_single_embedding():
    unload_model(context, "embeddings")  # has no effect at present


# section 2 - multiple embeddings
def test_load_multiple_embedding():
    embeddings = [
        ("models/embeddings/charturnerv2.pt", "https://civitai.com/api/download/models/8387"),
        ("models/embeddings/sketchpad.pt", "https://civitai.com/api/download/models/122772"),
    ]

    for embedding_path, embedding_url in embeddings:
        if not os.path.exists(embedding_path):
            download_file(embedding_url, embedding_path)

    context.model_paths["embeddings"] = [path for path, _ in embeddings]  # testing an array path assignment
    load_model(context, "embeddings")


def test_2_1__multiple_a1111_embedding__sd_1_4__first_prefix_only():
    check_embedding_image(
        expected_image="embedding.png",
        test_name="test2.1",
        prefix="charturnerv2",
    )


def test_2_2__multiple_a1111_embedding__sd_1_4__second_prefix_only():
    check_embedding_image(
        expected_image="embedding-multiple-second-only.png",
        test_name="test2.2",
        prefix="sketchpad",
    )


def test_2_3__multiple_a1111_embedding__sd_1_4__both_prefixes():
    check_embedding_image(
        expected_image="embedding-multiple-both.png",
        test_name="test2.3",
        prefix="charturnerv2 sketchpad",
    )


# section 3 - sdxl embedding
def test_load_sdxl():
    context.model_paths["stable-diffusion"] = "models/stable-diffusion/xl/sd_xl_base_1.0.safetensors"
    load_model(context, "stable-diffusion")


def test_3_1__without_embedding__sdxl__with_trigger_word():
    check_embedding_image(
        expected_image="no-embedding-sdxl.png",
        test_name="test3.1",
        prefix="annev",
    )


def test_load_single_sdxl_embedding():
    embedding_path = "models/embeddings/annev.safetensors"
    embedding_url = "https://civitai.com/api/download/models/138153"

    if not os.path.exists(embedding_path):
        download_file(embedding_url, embedding_path)

    context.model_paths["embeddings"] = embedding_path
    load_model(context, "embeddings")


def test_3_2__single_a1111_embedding__sdxl__with_trigger_word():
    check_embedding_image(
        expected_image="embedding-sdxl.png",
        test_name="test3.2",
        prefix="annev",
    )


# section 4 - misc
def vram_usage_level_test(vram_usage_level, test_name):
    unload_model(context, "stable-diffusion")
    unload_model(context, "embeddings")

    context.vram_usage_level = vram_usage_level

    test_load_sd_1_4()
    test_load_single_embedding()

    check_embedding_image(
        expected_image="embedding.png",
        test_name=test_name + "a",
        prefix="charturnerv2",
    )

    # reload SD
    test_load_sd_1_4()
    check_embedding_image(
        expected_image="embedding.png",
        test_name=test_name + "b",
        prefix="charturnerv2",
    )


def test_4_1__vram_mode__works_on_low():
    vram_usage_level_test(vram_usage_level="low", test_name="test4.1")


def test_4_2__vram_mode__works_on_balanced():
    vram_usage_level_test(vram_usage_level="balanced", test_name="test4.2")


def test_4_3__vram_mode__works_on_high():
    vram_usage_level_test(vram_usage_level="high", test_name="test4.3")
