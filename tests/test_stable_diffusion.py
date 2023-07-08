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
def test_1_0__stable_diffusion_1_4_txt2img_works__64x64():
    image = generate_images(context, "Photograph of an astronaut riding a horse", seed=42, width=64, height=64)[0]

    expected_image = Image.open(f"{EXPECTED_DIR}/1.4-txt-euler_a-42-64x64-cuda.png")
    assert_images_same(image, expected_image, "stable_diffusion_test1.0")


def test_1_1__stable_diffusion_1_4_txt2img_works__512x512():
    image = generate_images(context, "Photograph of an astronaut riding a horse", seed=42, width=512, height=512)[0]

    expected_image = Image.open(f"{EXPECTED_DIR}/1.4-txt-euler_a-42-512x512-cuda.png")
    assert_images_same(image, expected_image, "stable_diffusion_test1.1")


def test_1_2__stable_diffusion_1_4_img2img_works__512x512():
    init_img = Image.open(f"{TEST_DATA_FOLDER}/input_images/dog-512x512.png")
    image = generate_images(context, "Lion sitting on a bench", seed=42, width=512, height=512, init_image=init_img)[0]

    expected_image = Image.open(f"{EXPECTED_DIR}/1.4-img-euler_a-42-512x512-cuda.png")
    assert_images_same(image, expected_image, "stable_diffusion_test1.2")


def test_1_3__stable_diffusion_1_4_inpainting_works__512x512():
    init_img = Image.open(f"{TEST_DATA_FOLDER}/input_images/dog-512x512.png")
    mask = Image.open(f"{TEST_DATA_FOLDER}/input_images/dog_mask-512x512.png")
    image = generate_images(
        context, "Lion sitting on a bench", seed=43, width=512, height=512, init_image=init_img, init_image_mask=mask
    )[0]

    expected_image = Image.open(f"{EXPECTED_DIR}/1.4-inpaint-euler_a-43-512x512-cuda.png")
    assert_images_same(image, expected_image, "stable_diffusion_test1.3")


def stable_diffusion_works_on_multiple_devices_in_parallel_test(model, vram_usage_level, test_name, args={}):
    init_args(args)

    model_file, model_ver = model

    def task(context: Context):
        context.test_diffusers = True
        context.vram_usage_level = vram_usage_level
        context.model_paths["stable-diffusion"] = f"models/stable-diffusion/{model_file}"

        load_model(context, "stable-diffusion")

        image = generate_images(context, **args)[0]

        if args.get("init_image"):
            expected_image = "inpaint" if args.get("init_image_mask") else "img"
        else:
            expected_image = "txt"

        expected_image = f"{EXPECTED_DIR}/{model_ver}-{expected_image}-{args['sampler_name']}-{args['seed']}-{args['width']}x{args['height']}"
        expected_image = get_image_for_device(expected_image, context.device)
        assert_images_same(image, expected_image, f"stable_diffusion_{test_name}_{context.device.replace(':', '')}")

    # emulate multiple GPUs by running one thread on the CPU, and one on the GPU
    run_test_on_multiple_devices(task, ["cuda:0", "cpu"])


def test_1_10a__stable_diffusion_txt2img_works_on_multiple_devices__low_VRAM():
    stable_diffusion_works_on_multiple_devices_in_parallel_test(("sd-v1-4.ckpt", "1.4"), "low", "test1.10a")


def test_1_10b__stable_diffusion_txt2img_works_on_multiple_devices__balanced_VRAM():
    stable_diffusion_works_on_multiple_devices_in_parallel_test(("sd-v1-4.ckpt", "1.4"), "balanced", "test1.10b")


def test_1_10c__stable_diffusion_txt2img_works_on_multiple_devices__high_VRAM():
    stable_diffusion_works_on_multiple_devices_in_parallel_test(("sd-v1-4.ckpt", "1.4"), "high", "test1.10c")


def sd_1_4_image_test(model, vram_usage_mode, test_name, inpaint=False):
    args = {
        "prompt": "Lion sitting on a bench",
        "seed": 43,
        "width": 512,
        "height": 512,
        "init_image": Image.open(f"{TEST_DATA_FOLDER}/input_images/dog-512x512.png"),
    }
    if inpaint:
        args["init_image_mask"] = Image.open(f"{TEST_DATA_FOLDER}/input_images/dog_mask-512x512.png")

    stable_diffusion_works_on_multiple_devices_in_parallel_test(model, vram_usage_mode, test_name, args)


def test_1_11a__stable_diffusion_img2img_works_on_multiple_devices__low_VRAM():
    sd_1_4_image_test(("sd-v1-4.ckpt", "1.4"), "low", "test1.11a")


def test_1_11a__stable_diffusion_img2img_works_on_multiple_devices__balanced_VRAM():
    sd_1_4_image_test(("sd-v1-4.ckpt", "1.4"), "balanced", "test1.11b")


def test_1_11a__stable_diffusion_img2img_works_on_multiple_devices__high_VRAM():
    sd_1_4_image_test(("sd-v1-4.ckpt", "1.4"), "high", "test1.11c")


def test_1_12a__stable_diffusion_legacy_inpaint_works_on_multiple_devices__low_VRAM():
    sd_1_4_image_test(("sd-v1-4.ckpt", "1.4"), "low", "test1.12a", inpaint=True)


def test_1_12a__stable_diffusion_legacy_inpaint_works_on_multiple_devices__balanced_VRAM():
    sd_1_4_image_test(("sd-v1-4.ckpt", "1.4"), "balanced", "test1.12b", inpaint=True)


def test_1_12a__stable_diffusion_legacy_inpaint_works_on_multiple_devices__high_VRAM():
    sd_1_4_image_test(("sd-v1-4.ckpt", "1.4"), "high", "test1.12c", inpaint=True)


def test_1_13a__stable_diffusion_inpaint_model_works_on_multiple_devices__low_VRAM():
    sd_1_4_image_test(("512-inpainting-ema.ckpt", "2.0"), "low", "test1.13a", inpaint=True)


def test_1_13b__stable_diffusion_inpaint_model_works_on_multiple_devices__balanced_VRAM():
    sd_1_4_image_test(("512-inpainting-ema.ckpt", "2.0"), "balanced", "test1.13b", inpaint=True)


def test_1_13c__stable_diffusion_inpaint_model_works_on_multiple_devices__high_VRAM():
    sd_1_4_image_test(("512-inpainting-ema.ckpt", "2.0"), "high", "test1.13c", inpaint=True)


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


def init_args(args: dict):
    args["prompt"] = args.get("prompt", "Photograph of an astronaut riding a horse")
    args["seed"] = args.get("seed", 42)
    args["width"] = args.get("width", 512)
    args["height"] = args.get("height", 512)
    args["sampler_name"] = args.get("sampler_name", "euler_a")
