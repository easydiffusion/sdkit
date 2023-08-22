from PIL import Image

import torch
import os

from sdkit import Context
from sdkit.generate import generate_images
from sdkit.models import load_model
from sdkit.utils import diffusers_latent_samples_to_images, img_to_buffer, download_file


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
def test_1_0a__stable_diffusion_1_4_txt2img_works__64x64():
    image = generate_images(context, "Horse", seed=42, width=64, height=64, num_inference_steps=1)[0]

    assert image is not None
    assert image.getbbox(), f"Image is black!"


def test_1_0b__stable_diffusion_1_4_img2img_works__64x64():
    init_img = Image.open(f"{TEST_DATA_FOLDER}/input_images/dog-512x512.png")
    image = generate_images(context, "Horse", seed=42, width=64, height=64, num_inference_steps=3, init_image=init_img)
    image = image[0]

    assert image is not None
    assert image.getbbox(), f"Image is black!"


def test_1_0c__stable_diffusion_1_4_inpainting_works__64x64():
    init_img = Image.open(f"{TEST_DATA_FOLDER}/input_images/dog-512x512.png")
    mask = Image.open(f"{TEST_DATA_FOLDER}/input_images/dog_mask-512x512.png")
    image = generate_images(
        context, "Horse", seed=42, width=64, height=64, num_inference_steps=3, init_image=init_img, init_image_mask=mask
    )[0]

    assert image is not None
    assert image.getbbox(), f"Image is black!"


def test_1_0d__stable_diffusion_1_4_works_on_multiple_devices_and_vram_levels():
    for vram_usage_level in ("low", "balanced", "high"):

        def task(context: Context):
            context.test_diffusers = True
            context.vram_usage_level = vram_usage_level
            context.model_paths["stable-diffusion"] = f"models/stable-diffusion/sd-v1-4.ckpt"

            load_model(context, "stable-diffusion")

            image = generate_images(context, "Horse", seed=42, width=64, height=64, num_inference_steps=1)[0]

            assert image is not None, f"{vram_usage_level} {context.device} - Image is None"
            assert image.getbbox(), f"{vram_usage_level} {context.device} - Image is black!"

        # emulate multiple GPUs by running one thread on the CPU, and one on the GPU
        run_test_on_multiple_devices(task, ["cuda:0", "cpu"])


def make_live_preview_callback(width, height):
    def on_step(samples, i, *args):
        images = diffusers_latent_samples_to_images(context, (samples, args[0]))
        assert images is not None
        for img in images:
            w, h = img.size
            assert w == width
            assert h == height

            assert img.getbbox(), "Live preview image is black!"

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


# SD XL
def test_2_0__sdxl__loads_base_model():
    context.model_paths["stable-diffusion"] = "models/stable-diffusion/official/sd_xl_base_1.0.safetensors"
    load_model(context, "stable-diffusion")


# quick tests (64x64)
def test_2_1a__sdxl__txt2img_works():
    images = generate_images(context, "Horse", seed=42, width=64, height=64, num_inference_steps=1)

    assert images[0] is not None
    assert images[0].getbbox(), f"Image is black!"


def test_2_1b__sdxl_img2img_works():
    init_img = Image.open(f"{TEST_DATA_FOLDER}/input_images/dog-512x512.png")
    images = generate_images(context, "Horse", seed=42, width=64, height=64, num_inference_steps=3, init_image=init_img)

    assert images[0] is not None
    assert images[0].getbbox(), f"Image is black!"


def test_2_1c__sdxl_inpainting_works():
    init_img = Image.open(f"{TEST_DATA_FOLDER}/input_images/dog-512x512.png")
    mask = Image.open(f"{TEST_DATA_FOLDER}/input_images/dog_mask-512x512.png")
    images = generate_images(
        context, "Horse", seed=42, width=64, height=64, num_inference_steps=3, init_image=init_img, init_image_mask=mask
    )

    assert images[0] is not None
    assert images[0].getbbox(), f"Image is black!"


def test_2_1d__live_preview__sdxl_txt2img_works__64x64():
    image = generate_images(
        context,
        "Photograph of an astronaut riding a horse",
        seed=42,
        width=64,
        height=64,
        callback=make_live_preview_callback(64, 64),
    )[0]

    assert image is not None
    assert image.getbbox(), f"Image is black!"


## refiner model
def test_2_4__sdxl__loads_refiner_model():
    context.model_paths["stable-diffusion"] = "models/stable-diffusion/official/sd_xl_refiner_1.0.safetensors"
    load_model(context, "stable-diffusion")


### quick tests (only supports img2img)
def test_2_4a__sdxl_refiner_img2img_works():
    init_img = Image.open(f"{TEST_DATA_FOLDER}/input_images/dog-512x512.png")
    images = generate_images(context, "Horse", seed=42, width=64, height=64, num_inference_steps=3, init_image=init_img)

    assert images[0] is not None
    assert images[0].getbbox(), f"Image is black!"


## misc tests
def test_2_5a__sdxl__base_txt2img_works_on_low():
    context.vram_usage_level = "low"

    context.model_paths["stable-diffusion"] = "models/stable-diffusion/official/sd_xl_base_1.0.safetensors"
    load_model(context, "stable-diffusion")

    images = generate_images(context, "Horse", seed=42, width=64, height=64, num_inference_steps=1)

    assert images[0] is not None
    assert images[0].getbbox(), f"Image is black!"


def test_2_5b__sdxl__refiner_img2img_works_on_low():
    context.vram_usage_level = "low"

    context.model_paths["stable-diffusion"] = "models/stable-diffusion/official/sd_xl_refiner_1.0.safetensors"
    load_model(context, "stable-diffusion")

    init_img = Image.open(f"{TEST_DATA_FOLDER}/input_images/dog-512x512.png")
    images = generate_images(context, "Horse", seed=42, width=64, height=64, num_inference_steps=3, init_image=init_img)

    assert images[0] is not None
    assert images[0].getbbox(), f"Image is black!"


def test_2_6__sdxl__runs_on_multiple_devices_in_parallel():
    def task(context: Context):
        context.test_diffusers = True
        context.model_paths["stable-diffusion"] = "models/stable-diffusion/official/sd_xl_base_1.0.safetensors"

        load_model(context, "stable-diffusion")

        images = generate_images(context, "Horse", seed=42, width=64, height=64, num_inference_steps=1)

        assert images[0] is not None
        assert images[0].getbbox(), f"Image is black!"

    run_test_on_multiple_devices(task, ["cuda:0", "cpu"])


# inpainting models
## official 1.5 inpainting
def test_3_0__1_5_inpainting__loads_model():
    context.model_paths["stable-diffusion"] = "models/stable-diffusion/official/1.5/sd-v1-5-inpainting.ckpt"
    load_model(context, "stable-diffusion")


def test_3_1__stable_diffusion_1_5_inpainting_works__64x64():
    init_img = Image.open(f"{TEST_DATA_FOLDER}/input_images/dog-512x512.png")
    mask = Image.open(f"{TEST_DATA_FOLDER}/input_images/dog_mask-512x512.png")
    image = generate_images(
        context, "Horse", seed=42, width=64, height=64, num_inference_steps=3, init_image=init_img, init_image_mask=mask
    )[0]

    assert image is not None
    assert image.getbbox(), f"Image is black!"


## official 2.0 inpainting
def test_3_2__2_0_inpainting__loads_model():
    context.model_paths["stable-diffusion"] = "models/stable-diffusion/official/2.0/512-inpainting-ema.ckpt"
    load_model(context, "stable-diffusion")


def test_3_3__stable_diffusion_2_0_inpainting_works__64x64():
    init_img = Image.open(f"{TEST_DATA_FOLDER}/input_images/dog-512x512.png")
    mask = Image.open(f"{TEST_DATA_FOLDER}/input_images/dog_mask-512x512.png")
    image = generate_images(
        context, "Horse", seed=42, width=64, height=64, num_inference_steps=3, init_image=init_img, init_image_mask=mask
    )[0]

    assert image is not None
    assert image.getbbox(), f"Image is black!"


## custom inpainting 1.5
def test_3_4__custom_inpainting_1_5__loads_model():
    model_path = "models/stable-diffusion/custom/rpgInpainting_v4-inpainting.safetensors"
    model_url = "https://civitai.com/api/download/models/96255"

    if not os.path.exists(model_path):
        download_file(model_url, model_path)

    context.model_paths["stable-diffusion"] = model_path
    load_model(context, "stable-diffusion")


def test_3_5__stable_diffusion_custom_inpainting_1_5_works__64x64():
    init_img = Image.open(f"{TEST_DATA_FOLDER}/input_images/dog-512x512.png")
    mask = Image.open(f"{TEST_DATA_FOLDER}/input_images/dog_mask-512x512.png")
    image = generate_images(
        context, "Horse", seed=42, width=64, height=64, num_inference_steps=3, init_image=init_img, init_image_mask=mask
    )[0]

    assert image is not None
    assert image.getbbox(), f"Image is black!"


## custom inpainting 2.1
def test_3_6__custom_inpainting_2_1__loads_model():
    model_path = "models/stable-diffusion/custom/aZovyaRPGArtistTools_sd21768V1Inpainting.safetensors"
    model_url = "https://civitai.com/api/download/models/57615"

    if not os.path.exists(model_path):
        download_file(model_url, model_path)

    context.model_paths["stable-diffusion"] = model_path
    load_model(context, "stable-diffusion")


def test_3_7__stable_diffusion_custom_inpainting_2_0_works__64x64():
    init_img = Image.open(f"{TEST_DATA_FOLDER}/input_images/dog-512x512.png")
    mask = Image.open(f"{TEST_DATA_FOLDER}/input_images/dog_mask-512x512.png")
    image = generate_images(
        context, "Horse", seed=42, width=64, height=64, num_inference_steps=3, init_image=init_img, init_image_mask=mask
    )[0]

    assert image is not None
    assert image.getbbox(), f"Image is black!"
