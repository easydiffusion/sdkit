from PIL import Image

import torch

from sdkit import Context
from sdkit.generate import generate_images
from sdkit.models import load_model
from sdkit.utils import diffusers_latent_samples_to_images, img_to_buffer


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


# section 1 - SD at different resolutions and samplers
def test_1_0__stable_diffusion_1_4_txt2img_works__64x64():
    image = generate_images(context, "Photograph of an astronaut riding a horse", seed=42, width=64, height=64)[0]

    expected_image = Image.open(f"{EXPECTED_DIR}/1.4-txt-euler_a-42-64x64-cuda.png")
    assert_images_same(image, expected_image, "test1.0")


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
            test_type = "inpaint" if args.get("init_image_mask") else "img"
        else:
            test_type = "txt"

        expected_image = f"{EXPECTED_DIR}/{model_ver}-{test_type}-{args['sampler_name']}-{args['seed']}-{args['width']}x{args['height']}"
        if vram_usage_level == "high" and test_type == "img":
            expected_image += "-high"
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


def test_1_11b__stable_diffusion_img2img_works_on_multiple_devices__balanced_VRAM():
    sd_1_4_image_test(("sd-v1-4.ckpt", "1.4"), "balanced", "test1.11b")


def test_1_11c__stable_diffusion_img2img_works_on_multiple_devices__high_VRAM():
    sd_1_4_image_test(("sd-v1-4.ckpt", "1.4"), "high", "test1.11c")


def test_1_12a__stable_diffusion_legacy_inpaint_works_on_multiple_devices__low_VRAM():
    sd_1_4_image_test(("sd-v1-4.ckpt", "1.4"), "low", "test1.12a", inpaint=True)


def test_1_12b__stable_diffusion_legacy_inpaint_works_on_multiple_devices__balanced_VRAM():
    sd_1_4_image_test(("sd-v1-4.ckpt", "1.4"), "balanced", "test1.12b", inpaint=True)


def test_1_12c__stable_diffusion_legacy_inpaint_works_on_multiple_devices__high_VRAM():
    sd_1_4_image_test(("sd-v1-4.ckpt", "1.4"), "high", "test1.12c", inpaint=True)


def test_1_13a__stable_diffusion_inpaint_model_works_on_multiple_devices__low_VRAM():
    sd_1_4_image_test(("512-inpainting-ema.ckpt", "2.0"), "low", "test1.13a", inpaint=True)


def test_1_13b__stable_diffusion_inpaint_model_works_on_multiple_devices__balanced_VRAM():
    sd_1_4_image_test(("512-inpainting-ema.ckpt", "2.0"), "balanced", "test1.13b", inpaint=True)


def test_1_13c__stable_diffusion_inpaint_model_works_on_multiple_devices__high_VRAM():
    sd_1_4_image_test(("512-inpainting-ema.ckpt", "2.0"), "high", "test1.13c", inpaint=True)


def test_1_14a__stable_diffusion_2_0_txt2img_works__64x64():
    context.model_paths["stable-diffusion"] = "models/stable-diffusion/official/2.0/512-base-ema.ckpt"

    load_model(context, "stable-diffusion")

    image = generate_images(context, "Photograph of an astronaut riding a horse", seed=42, width=64, height=64)[0]

    expected_image = Image.open(f"{EXPECTED_DIR}/2.0-txt-euler_a-42-64x64-cuda.png")
    assert_images_same(image, expected_image, "test1.14a")


def test_1_14b__stable_diffusion_2_1_txt2img_works__64x64():
    context.model_paths["stable-diffusion"] = "models/stable-diffusion/official/2.1/v2-1_512-ema-pruned.safetensors"

    load_model(context, "stable-diffusion")

    image = generate_images(context, "Photograph of an astronaut riding a horse", seed=42, width=64, height=64)[0]

    expected_image = Image.open(f"{EXPECTED_DIR}/2.1-txt-euler_a-42-64x64-cuda.png")
    assert_images_same(image, expected_image, "test1.14b")


def make_live_preview_callback(width, height):
    def on_step(samples, i, *args):
        images = diffusers_latent_samples_to_images(context, (samples, args[0]))
        assert images is not None
        for img in images:
            w, h = img.size
            assert w == width
            assert h == height
            buf = img_to_buffer(img, output_format="JPEG")
            assert buf is not None

    return on_step


def test_1_15a__live_preview__stable_diffusion_1_4_txt2img_works__64x64():
    image = generate_images(
        context,
        "Photograph of an astronaut riding a horse",
        seed=42,
        width=64,
        height=64,
        callback=make_live_preview_callback(64, 64),
    )[0]

    expected_image = Image.open(f"{EXPECTED_DIR}/1.4-txt-euler_a-42-64x64-cuda.png")
    assert_images_same(image, expected_image, "test1.15a")


def test_1_15b__live_preview__stable_diffusion_1_4_img2img_works__64x64():
    image = generate_images(
        context,
        "Lion sitting on a bench",
        seed=42,
        width=64,
        height=64,
        callback=make_live_preview_callback(64, 64),
        init_image=Image.open(f"{TEST_DATA_FOLDER}/input_images/dog-512x512.png"),
    )[0]

    expected_image = Image.open(f"{EXPECTED_DIR}/1.4-img-euler_a-42-64x64-cuda.png")
    assert_images_same(image, expected_image, "test1.15b")


def test_1_15c__live_preview__stable_diffusion_1_4_inpaint_works__64x64():
    image = generate_images(
        context,
        "Lion sitting on a bench",
        seed=42,
        width=64,
        height=64,
        callback=make_live_preview_callback(64, 64),
        init_image=Image.open(f"{TEST_DATA_FOLDER}/input_images/dog-512x512.png"),
        init_image_mask=Image.open(f"{TEST_DATA_FOLDER}/input_images/dog_mask-512x512.png"),
    )[0]

    expected_image = Image.open(f"{EXPECTED_DIR}/1.4-inpaint-euler_a-42-64x64-cuda.png")
    assert_images_same(image, expected_image, "test1.15c")


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
