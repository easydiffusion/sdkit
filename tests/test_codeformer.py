from PIL import Image

from sdkit import Context
from sdkit.filter import apply_filters
from sdkit.models import load_model


from common import TEST_DATA_FOLDER, assert_images_same, run_test_on_multiple_devices

context = None


def setup_module():
    global context

    context = Context()
    context.model_paths["codeformer"] = "models/codeformer/codeformer.pth"
    context.enable_codeformer = True

    load_model(context, "codeformer")


def test_codeformer_is_applied():
    image = Image.open(f"{TEST_DATA_FOLDER}/input_images/man_512x512.png")
    image_face_fixed = apply_filters(context, "codeformer", image, codeformer_fidelity=0.5)[0]
    expected_image = Image.open(f"{TEST_DATA_FOLDER}/expected_images/codeformer/man_512x512_no-upscale_cuda.png")

    assert_images_same(image_face_fixed, expected_image, "tmp/codeformer_test1")


def test_codeformer_works_on_multiple_devices():
    def task(context):
        context.model_paths["codeformer"] = "models/codeformer/codeformer.pth"
        context.enable_codeformer = True

        load_model(context, "codeformer")

        # apply the filter
        image = Image.open(f"{TEST_DATA_FOLDER}/input_images/man_512x512.png")
        image_face_fixed = apply_filters(context, "codeformer", image)[0]

        expected_image = f"{TEST_DATA_FOLDER}/expected_images/codeformer/man_512x512_no-upscale_"
        if context.device.startswith("cuda"):
            expected_image += "cuda.png"
        elif context.device == "cpu":
            expected_image += "cpu.png"

        expected_image = Image.open(expected_image)
        assert_images_same(image_face_fixed, expected_image, "tmp/codeformer_test2")

    # emulate multiple GPUs by running one thread on the CPU, and one on the GPU
    run_test_on_multiple_devices(task, ["cuda:0", "cpu"])
