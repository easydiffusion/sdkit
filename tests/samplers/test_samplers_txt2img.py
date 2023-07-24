from PIL import Image

from sdkit import Context
from sdkit.generate import generate_images
from sdkit.generate.sampler import default_samplers, k_samplers, unipc_samplers
from sdkit.models import load_model


from common import (
    TEST_DATA_FOLDER,
    assert_images_same,
)

EXPECTED_DIR = f"{TEST_DATA_FOLDER}/expected_images/stable-diffusion/samplers"

all_samplers = (
    set(default_samplers.samplers.keys()) | set(k_samplers.samplers.keys()) | set(unipc_samplers.samplers.keys())
)

context = None


def setup_module():
    global context

    context = Context()
    context.test_diffusers = True
    context.model_paths["stable-diffusion"] = "models/stable-diffusion/sd-v1-4.ckpt"

    load_model(context, "stable-diffusion")


def make_sampler_test(sampler_name):
    def sampler_test():
        image = generate_images(
            context,
            "Photograph of an astronaut riding a horse",
            seed=42,
            width=512,
            height=512,
            sampler_name=sampler_name,
            num_inference_steps=25,
        )[0]

        expected_image = Image.open(f"{EXPECTED_DIR}/1.4-txt-{sampler_name}-42-512x512-50-cuda.png")
        assert_images_same(image, expected_image, "test" + sampler_name)

    return sampler_test


for s in all_samplers:
    globals()["test_" + s] = make_sampler_test(s)
