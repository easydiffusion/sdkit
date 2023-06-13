from PIL import Image

from sdkit import Context
from sdkit.filter import apply_filters
from sdkit.models import load_model


from common import TEST_DATA_FOLDER, get_image_for_device, assert_images_same, run_test_on_multiple_devices

EXPECTED_DIR = f"{TEST_DATA_FOLDER}/expected_images/codeformer"

context = None


def setup_module():
    global context

    context = Context()
    context.model_paths["codeformer"] = "models/codeformer/codeformer.pth"
    context.enable_codeformer = True

    load_model(context, "codeformer")


def test_codeformer_is_applied():
    image = Image.open(f"{TEST_DATA_FOLDER}/input_images/man-512x512.png")
    image_face_fixed = apply_filters(context, "codeformer", image, codeformer_fidelity=0.5)[0]
    expected_image = Image.open(f"{EXPECTED_DIR}/man-512x512-no_upscale-cuda.png")

    assert_images_same(image_face_fixed, expected_image, "codeformer_test1")


def test_codeformer_works_on_multiple_devices():
    def task(context):
        context.model_paths["codeformer"] = "models/codeformer/codeformer.pth"
        context.enable_codeformer = True

        load_model(context, "codeformer")

        # apply the filter
        image = Image.open(f"{TEST_DATA_FOLDER}/input_images/man-512x512.png")
        image_face_fixed = apply_filters(context, "codeformer", image)[0]

        expected_image = get_image_for_device(f"{EXPECTED_DIR}/man-512x512-no_upscale", context.device)
        assert_images_same(image_face_fixed, expected_image, "codeformer_test2")

    # emulate multiple GPUs by running one thread on the CPU, and one on the GPU
    run_test_on_multiple_devices(task, ["cuda:0", "cpu"])
