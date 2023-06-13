from PIL import Image

import torch

from sdkit import Context
from sdkit.generate import generate_images
from sdkit.models import load_model


from common import (
    TEST_DATA_FOLDER,
    get_image_for_device,
    get_tensor_for_device,
    assert_images_same,
    run_test_on_multiple_devices,
)

EXPECTED_DIR = f"{TEST_DATA_FOLDER}/expected_images/stable-diffusion"

context = None


def setup_module():
    global context

    context = Context()
    context.test_diffusers = True
    context.model_paths["stable-diffusion"] = "models/stable-diffusion/sd-v1-4.ckpt"

    load_model(context, "stable-diffusion")


# section 1 - SD 1.4 at different resolutions and samplers
def test_1_0__stable_diffusion_1_4_works__64x64():
    image = generate_images(context, "Photograph of an astronaut riding a horse", seed=42, width=64, height=64)[0]

    expected_image = Image.open(f"{EXPECTED_DIR}/1.4-euler_a-42-64x64-cuda.png")
    assert_images_same(image, expected_image, "stable_diffusion_test1.0")


def test_1_1__stable_diffusion_1_4_works__512x512():
    image = generate_images(context, "Photograph of an astronaut riding a horse", seed=42, width=512, height=512)[0]

    expected_image = Image.open(f"{EXPECTED_DIR}/1.4-euler_a-42-512x512-cuda.png")
    assert_images_same(image, expected_image, "stable_diffusion_test1.1")


def stable_diffusion_works_on_multiple_devices_in_parallel_test(vram_usage_level: str):
    def task(context: Context):
        context.test_diffusers = True
        context.vram_usage_level = vram_usage_level
        context.model_paths["stable-diffusion"] = "models/stable-diffusion/sd-v1-4.ckpt"

        load_model(context, "stable-diffusion")

        image = generate_images(context, "Photograph of an astronaut riding a horse", seed=42, width=512, height=512)[0]

        expected_image = get_image_for_device(f"{EXPECTED_DIR}/1.4-euler_a-42-512x512", context.device)
        assert_images_same(image, expected_image, f"stable_diffusion_test1.10_{context.device.replace(':', '')}")

    # emulate multiple GPUs by running one thread on the CPU, and one on the GPU
    run_test_on_multiple_devices(task, ["cuda:0"])


def test_1_10a__stable_diffusion_works_on_multiple_devices__low_VRAM():
    stable_diffusion_works_on_multiple_devices_in_parallel_test(vram_usage_level="low")


def test_1_10b__stable_diffusion_works_on_multiple_devices__balanced_VRAM():
    stable_diffusion_works_on_multiple_devices_in_parallel_test(vram_usage_level="balanced")


def test_1_10c__stable_diffusion_works_on_multiple_devices__high_VRAM():
    stable_diffusion_works_on_multiple_devices_in_parallel_test(vram_usage_level="high")


def test_2_0__misc__compel_parses_prompts():
    compel = context.models["stable-diffusion"]["compel"]
    embeds = compel("Photograph of an astronaut riding a horse")

    expected_embeds = f"{TEST_DATA_FOLDER}/expected_embeds/prompt-photograph_of_an_astronaut_riding_a_horse"
    expected_embeds = get_tensor_for_device(expected_embeds, context.device)

    assert embeds.device == torch.device(context.device)
    assert torch.equal(embeds, expected_embeds)


def compel_parses_prompts_on_multiple_devices_in_parallel_test(vram_usage_level: str):
    def task(context: Context):
        context.test_diffusers = True
        context.vram_usage_level = vram_usage_level
        context.model_paths["stable-diffusion"] = "models/stable-diffusion/sd-v1-4.ckpt"

        load_model(context, "stable-diffusion")

        compel = context.models["stable-diffusion"]["compel"]
        embeds = compel("Photograph of an astronaut riding a horse")

        expected_embeds = f"{TEST_DATA_FOLDER}/expected_embeds/prompt-photograph_of_an_astronaut_riding_a_horse"
        expected_embeds = get_tensor_for_device(expected_embeds, context.device)

        assert embeds.device == torch.device(context.device)
        assert torch.equal(embeds, expected_embeds)

    run_test_on_multiple_devices(task, ["cuda:0", "cpu"])


def test_2_1a__misc__compel_parses_prompts_on_multiple_devices__low_VRAM_usage():
    compel_parses_prompts_on_multiple_devices_in_parallel_test(vram_usage_level="low")


def test_2_1b__misc__compel_parses_prompts_on_multiple_devices__balanced_VRAM_usage():
    compel_parses_prompts_on_multiple_devices_in_parallel_test(vram_usage_level="balanced")


def test_2_1c__misc__compel_parses_prompts_on_multiple_devices__high_VRAM_usage():
    compel_parses_prompts_on_multiple_devices_in_parallel_test(vram_usage_level="high")
