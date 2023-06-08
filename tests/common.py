import os
from PIL import Image, ImageChops
import numpy as np

TEST_DATA_REPO = "https://github.com/easydiffusion/sdkit-test-data.git"
TEST_DATA_FOLDER = "sdkit-test-data"


def fetch_test_data():
    "Fetches the test data from the git repository, by issuing a git pull"

    print("Fetching the latest test data..")

    if os.path.exists(TEST_DATA_FOLDER):
        os.system(f"git -C {TEST_DATA_FOLDER} pull")
    else:
        os.system(f"git clone {TEST_DATA_REPO} {TEST_DATA_FOLDER}")


def get_image_diff_histogram(img_a: Image, img_b: Image):
    diff = ImageChops.difference(img_a, img_b)
    hist = diff.convert("L").histogram()
    return np.array(hist)


def assert_images_same(img_a: Image, img_b: Image):
    hist = get_image_diff_histogram(img_a, img_b)
    assert not np.any(hist[2:]), f"Histogram: {list(hist.data)}"


fetch_test_data()
