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


def test_sd_1_4_loads():
    context.model_paths["stable-diffusion"] = "models/stable-diffusion/sd-v1-4.ckpt"
    load_model(context, "stable-diffusion")


# section 1
def test_1_1__stable_diffusion_1_4_txt2img_works__512x512():
    image = generate_images(context, "Photograph of an astronaut riding a horse", seed=42, width=512, height=512)[0]

    expected_image = Image.open(f"{EXPECTED_DIR}/1.4-txt-euler_a-42-512x512-cuda.png")
    assert_images_same(image, expected_image, "stable_diffusion_test1.1")


def test_1_2__stable_diffusion_1_4_img2img_works__512x512():
    init_img = Image.open(f"{TEST_DATA_FOLDER}/input_images/dog-512x512.png")
    image = generate_images(context, "Lion sitting on a bench", seed=42, width=512, height=512, init_image=init_img)[0]

    expected_image = Image.open(f"{EXPECTED_DIR}/1.4-img-euler_a-42-512x512-cuda.png")
    assert_images_same(image, expected_image, "stable_diffusion_test1.2")


def test_1_3a__stable_diffusion_1_4_inpainting_works__512x512():
    init_img = Image.open(f"{TEST_DATA_FOLDER}/input_images/dog-512x512.png")
    mask = Image.open(f"{TEST_DATA_FOLDER}/input_images/dog_mask-512x512.png")
    image = generate_images(
        context, "Lion sitting on a bench", seed=43, width=512, height=512, init_image=init_img, init_image_mask=mask
    )[0]

    expected_image = Image.open(f"{EXPECTED_DIR}/1.4-inpaint-euler_a-43-512x512-cuda.png")
    assert_images_same(image, expected_image, "stable_diffusion_test1.3a")


def test_1_3b__stable_diffusion_1_4_inpainting_works_with_strict_mask_border__512x512():
    init_img = Image.open(f"{TEST_DATA_FOLDER}/input_images/dog-512x512.png")
    mask = Image.open(f"{TEST_DATA_FOLDER}/input_images/dog_mask-512x512.png")
    image = generate_images(
        context,
        "Lion sitting on a bench",
        seed=43,
        width=512,
        height=512,
        init_image=init_img,
        init_image_mask=mask,
        strict_mask_border=True,
    )[0]

    expected_image = Image.open(f"{EXPECTED_DIR}/1.4-inpaint-euler_a-43-512x512-cuda-strict_mask.png")
    assert_images_same(image, expected_image, "stable_diffusion_test1.3b")


def stable_diffusion_works_on_multiple_devices_in_parallel_test(
    model, vram_usage_level, test_name, args={}, devices=["cuda:0", "cpu"]
):
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
        if vram_usage_level == "high" and test_type == "txt":
            expected_image += "-high"
        expected_image = get_image_for_device(expected_image, context.device)
        assert_images_same(image, expected_image, f"stable_diffusion_{test_name}_{context.device.replace(':', '')}")

    # emulate multiple GPUs by running one thread on the CPU, and one on the GPU
    run_test_on_multiple_devices(task, devices)


def test_1_4__stable_diffusion_txt2img_works_on_cpu():
    stable_diffusion_works_on_multiple_devices_in_parallel_test(
        ("sd-v1-4.ckpt", "1.4"), "low", "test1.4", devices=["cpu"]
    )


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


def init_args(args: dict):
    args["prompt"] = args.get("prompt", "Photograph of an astronaut riding a horse")
    args["seed"] = args.get("seed", 42)
    args["width"] = args.get("width", 512)
    args["height"] = args.get("height", 512)
    args["sampler_name"] = args.get("sampler_name", "euler_a")


# SD XL
def test_2_0__sdxl__loads_base_model():
    context.model_paths["stable-diffusion"] = "models/stable-diffusion/official/sd_xl_base_1.0.safetensors"
    load_model(context, "stable-diffusion")


## full tests (768x768)
def test_2_2a__sdxl_txt2img_works__768x768():
    image = generate_images(context, "Photograph of an astronaut riding a horse", seed=42, width=768, height=768)[0]

    expected_image = Image.open(f"{EXPECTED_DIR}/xl-txt-euler_a-42-768x768-cuda.png")
    assert_images_same(image, expected_image, "test2.2a")


def test_2_2b__sdxl_img2img_works__768x768():
    init_img = Image.open(f"{TEST_DATA_FOLDER}/input_images/dog-512x512.png")
    image = generate_images(context, "Lion sitting on a bench", seed=42, width=768, height=768, init_image=init_img)[0]

    expected_image = Image.open(f"{EXPECTED_DIR}/xl-img-euler_a-42-768x768-cuda.png")
    assert_images_same(image, expected_image, "test2.2b")


def test_2_3c__sdxl_inpainting_works__768x768():
    init_img = Image.open(f"{TEST_DATA_FOLDER}/input_images/dog-512x512.png")
    mask = Image.open(f"{TEST_DATA_FOLDER}/input_images/dog_mask-512x512.png")
    image = generate_images(
        context, "Lion sitting on a bench", seed=43, width=768, height=768, init_image=init_img, init_image_mask=mask
    )[0]

    expected_image = Image.open(f"{EXPECTED_DIR}/xl-inpaint-euler_a-43-768x768-cuda.png")
    assert_images_same(image, expected_image, "test2.2c")


## refiner model
def test_2_4__sdxl__loads_refiner_model():
    context.model_paths["stable-diffusion"] = "models/stable-diffusion/official/sd_xl_refiner_1.0.safetensors"
    load_model(context, "stable-diffusion")


### quick tests (only supports img2img)
def test_2_4a__sdxl_refiner_img2img_works():
    init_img = Image.open(f"{TEST_DATA_FOLDER}/input_images/dog-512x512.png")
    images = generate_images(
        context, "Horse", seed=42, width=512, height=512, num_inference_steps=25, init_image=init_img
    )

    assert images[0] is not None
    assert images[0].getbbox(), f"Image is black!"
