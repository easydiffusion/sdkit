from sdkit import Context
from sdkit.generate import generate_images
from sdkit.models import load_model, unload_model
from sdkit.utils import get_vram_usage_slow, log

from .common import USE_DIFFUSERS

context = None

usage_timeseries = []


def setup_module():
    global context

    context = Context()
    context.test_diffusers = USE_DIFFUSERS


def setup_function():
    usage_timeseries.clear()


def teardown_function():
    print("----")
    print(f"Usage timeseries:")
    for v in usage_timeseries:
        print(f"{v/1024**3:.1f} GiB")
    print("----")


def get_vram_usage():
    x = get_vram_usage_slow()
    print(f"VRAM usage: {int(x) / 1024**3:.1f} Gb")
    usage_timeseries.append(x)
    return x


def test_1__sd_1__model_unload_frees_vram():
    usage_start = get_vram_usage()

    context.model_paths["stable-diffusion"] = "models/stable-diffusion/1.x/sd-v1-4.ckpt"
    load_model(context, "stable-diffusion")

    usage_model_load = get_vram_usage()

    unload_model(context, "stable-diffusion")

    usage_model_unload = get_vram_usage()

    print("")
    log.info(
        f"VRAM trend: {usage_start/1024**3:.1f} GiB (start) to {usage_model_load/1024**3:.1f} GiB (after load) to {usage_model_unload/1024**3:.1f} GiB (after unload)"
    )
    print("")

    max_expected_vram = usage_start + 0.3
    assert (
        usage_model_unload < max_expected_vram
    ), f"Test failed! VRAM after unload was expected to be below {max_expected_vram/1024**3:.1f} GiB, but was {usage_model_unload/1024**3:.1f} GiB!"


def test_2__sd_1__model_changes():
    usage_start = get_vram_usage()

    context.model_paths["stable-diffusion"] = "models/stable-diffusion/1.x/sd-v1-4.ckpt"
    load_model(context, "stable-diffusion")

    usage_model_load1 = get_vram_usage()

    context.model_paths["stable-diffusion"] = "models/stable-diffusion/1.x/sd-v1-5-inpainting.ckpt"
    load_model(context, "stable-diffusion")

    usage_model_load2 = get_vram_usage()

    context.model_paths["stable-diffusion"] = "models/stable-diffusion/1.x/realisticVisionV51_v51VAE.safetensors"
    load_model(context, "stable-diffusion")

    usage_model_load3 = get_vram_usage()

    print("")
    log.info(
        f"VRAM trend: {usage_start/1024**3:.1f} GiB (start) to {usage_model_load1/1024**3:.1f} GiB (after load1) to {usage_model_load2/1024**3:.1f} GiB (after load2) to {usage_model_load3/1024**3:.1f} GiB (after load3)"
    )
    print("")

    max_expected_vram = usage_model_load1 + 0.3
    assert (
        usage_model_load3 < max_expected_vram
    ), f"Test failed! VRAM after load3 was expected to be below {max_expected_vram/1024**3:.1f} GiB, but was {usage_model_load3/1024**3:.1f} GiB!"


def test_3__sd_1_x__vram_frees_after_image():
    usage_start = get_vram_usage()

    context.model_paths["stable-diffusion"] = "models/stable-diffusion/1.x/sd-v1-4.ckpt"
    load_model(context, "stable-diffusion")

    log.info("Loaded the model..")
    usage_model_load = get_vram_usage()

    try:
        images = generate_images(context, prompt="Photograph of an astronaut riding a horse")
    except Exception as e:
        log.exception(e)

    log.info("Generated the image..")
    usage_after_render = get_vram_usage()

    print("")
    log.info(
        f"VRAM trend: {usage_start/1024**3:.1f} GiB (start) to {usage_model_load/1024**3:.1f} GiB (before render) to {usage_after_render/1024**3:.1f} GiB (after render)"
    )
    print("")

    max_expected_vram = usage_model_load + 0.3
    assert (
        usage_after_render < max_expected_vram
    ), f"Test failed! VRAM after render was expected to be below {max_expected_vram/1024**3:.1f} GiB, but was {usage_after_render/1024**3:.1f} GiB!"


def test_4__sd_1__model_change_and_render__check_for_compounding_leaks():
    usage_start = get_vram_usage()

    for i in range(6):
        context.model_paths["stable-diffusion"] = "models/stable-diffusion/1.x/sd-v1-4.ckpt"
        load_model(context, "stable-diffusion")

        usage_model_load1 = get_vram_usage()

        try:
            images = generate_images(context, prompt="Photograph of an astronaut riding a horse", num_inference_steps=2)
        except Exception as e:
            log.exception(e)

        log.info("Generated the image..")
        usage_after_render1 = get_vram_usage()

        context.model_paths["stable-diffusion"] = "models/stable-diffusion/1.x/realisticVisionV51_v51VAE.safetensors"
        load_model(context, "stable-diffusion")

        usage_model_load2 = get_vram_usage()

        try:
            images = generate_images(
                context, prompt="Photograph of an astronaut riding a horse", height=512, num_inference_steps=2
            )
        except Exception as e:
            log.exception(e)

        log.info("Generated the image..")
        usage_after_render2 = get_vram_usage()
