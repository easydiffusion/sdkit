import os
from PIL import Image, ImageChops
import numpy as np

from concurrent import futures
from typing import Callable

from sdkit import Context
from sdkit.utils import log

TEST_DATA_REPO = "https://github.com/easydiffusion/sdkit-test-data.git"
TEST_DATA_FOLDER = "sdkit-test-data"


def fetch_test_data():
    "Fetches the test data from the git repository, by issuing a git pull"

    print("Fetching the latest test data..")

    if os.path.exists(TEST_DATA_FOLDER):
        os.system(f"git -C {TEST_DATA_FOLDER} pull")
    else:
        os.system(f"git clone {TEST_DATA_REPO} {TEST_DATA_FOLDER}")


def get_image_diff(img_a: Image, img_b: Image):
    diff = ImageChops.difference(img_a, img_b)
    hist = diff.convert("L").histogram()
    return diff, np.array(hist)


def assert_images_same(actual: Image, expected: Image, failed_image_save_path: str = None):
    diff, hist = get_image_diff(actual, expected)
    same = not np.any(hist[2:])

    if not same and failed_image_save_path:
        actual.save(failed_image_save_path + "_actual.png")
        expected.save(failed_image_save_path + "_expected.png")
        diff.save(failed_image_save_path + "_diff.png")
        print(f">> Saved actual/expected/diff images to {failed_image_save_path}")

    assert same, f"Histogram: {list(hist.data)}"


def run_test_on_multiple_devices(task: Callable, devices: list):
    def task_thread(device):
        log.info(f"starting on device {device}")
        context = Context()
        context.device = device

        task(context)

    with futures.ThreadPoolExecutor(max_workers=len(devices)) as executor:
        threads = [executor.submit(task_thread, device) for device in devices]
        for f in futures.as_completed(threads):
            if f.exception():
                raise f.exception()


fetch_test_data()
