from PIL import Image

import sdkit
from sdkit.filter import apply_filters
from sdkit.models import load_model

from common import TEST_DATA_FOLDER, assert_images_same

context = None


def setup_module():
    global context

    context = sdkit.Context()
    context.model_paths["codeformer"] = "models/codeformer/codeformer.pth"
    context.enable_codeformer = True

    load_model(context, "codeformer")


def test_1_0__codeformer_is_applied():
    image = Image.open(f"{TEST_DATA_FOLDER}/input_images/man_512x512.png")
    image_face_fixed = apply_filters(context, "codeformer", image)[0]
    expected_image = Image.open(f"{TEST_DATA_FOLDER}/expected_images/codeformer/man_512x512_no-upscale.png")

    image_face_fixed.save("tmp/codeformer_man_512x512_1_0.png")

    assert_images_same(image_face_fixed, expected_image)
