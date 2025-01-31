from PIL import Image

from sdkit import Context
from sdkit.filter import apply_filters
from sdkit.models import load_model

from common import (
    TEST_DATA_FOLDER,
    GPU_DEVICE_NAME,
    USE_DIFFUSERS,
    assert_images_same,
    run_test_on_multiple_devices,
)

EXPECTED_DIR = f"{TEST_DATA_FOLDER}/expected_images"

context = None


def setup_module():
    global context

    context = Context()
    context.test_diffusers = USE_DIFFUSERS


def test_gfpgan_model_loads():
    context.model_paths["gfpgan"] = "models/gfpgan/GFPGANv1.3.pth"
    load_model(context, "gfpgan")


def test_realesrgan_model_loads():
    context.model_paths["realesrgan"] = "models/realesrgan/RealESRGAN_x4plus.pth"
    load_model(context, "realesrgan")


def test_gfpgan_applies():
    img = Image.open(f"{TEST_DATA_FOLDER}/input_images/man-512x512.png")

    filtered_img = apply_filters(context, "gfpgan", img)
    assert filtered_img is not None
    assert len(filtered_img) == 1

    expected_image = Image.open(f"{EXPECTED_DIR}/filters/gfpgan.png")
    assert_images_same(filtered_img[0], expected_image, "test_gfpgan")


def test_gfpgan_works_on_multiple_devices():
    def task(context):
        context.model_paths["gfpgan"] = "models/gfpgan/GFPGANv1.3.pth"
        load_model(context, "gfpgan")

        img = Image.open(f"{TEST_DATA_FOLDER}/input_images/man-512x512.png")

        filtered_img = apply_filters(context, "gfpgan", img)
        assert filtered_img is not None
        assert len(filtered_img) == 1

    run_test_on_multiple_devices(task, devices=["cpu", GPU_DEVICE_NAME])


def test_realesrgan_applies():
    img = Image.open(f"{TEST_DATA_FOLDER}/input_images/man-512x512.png")

    filtered_img = apply_filters(context, "realesrgan", img)
    assert filtered_img is not None
    assert len(filtered_img) == 1

    expected_image = Image.open(f"{EXPECTED_DIR}/filters/realesrgan.png")
    assert_images_same(filtered_img[0], expected_image, "test_realesrgan")


def test_realesrgan_works_on_multiple_devices():
    def task(context):
        context.model_paths["realesrgan"] = "models/realesrgan/RealESRGAN_x4plus.pth"
        load_model(context, "realesrgan")

        img = Image.open(f"{TEST_DATA_FOLDER}/input_images/man-512x512.png")
        img = img.resize((64, 64))

        filtered_img = apply_filters(context, "realesrgan", img)
        assert filtered_img is not None
        assert len(filtered_img) == 1

    run_test_on_multiple_devices(task, devices=["cpu", GPU_DEVICE_NAME])


def test_gfpgan_and_realesrgan_work_together_on_multiple_devices():
    def task(context):
        context.model_paths["gfpgan"] = "models/gfpgan/GFPGANv1.3.pth"
        load_model(context, "gfpgan")

        context.model_paths["realesrgan"] = "models/realesrgan/RealESRGAN_x4plus.pth"
        load_model(context, "realesrgan")

        img = Image.open(f"{TEST_DATA_FOLDER}/input_images/man-512x512.png")

        filtered_img = apply_filters(context, "gfpgan", img)
        assert filtered_img is not None
        assert len(filtered_img) == 1

        filtered_img = filtered_img[0].resize((64, 64))

        filtered_img = apply_filters(context, "realesrgan", filtered_img)
        assert filtered_img is not None
        assert len(filtered_img) == 1

    run_test_on_multiple_devices(task, devices=["cpu", GPU_DEVICE_NAME])


def test_stable_diffusion_and_gfpgan_load_together_on_multiple_devices():
    def task(context):
        context.model_paths["stable-diffusion"] = "models/stable-diffusion/1.x/sd-v1-4.ckpt"
        load_model(context, "stable-diffusion")

        context.model_paths["gfpgan"] = "models/gfpgan/GFPGANv1.3.pth"
        load_model(context, "gfpgan")

    run_test_on_multiple_devices(task, devices=[GPU_DEVICE_NAME, GPU_DEVICE_NAME])
