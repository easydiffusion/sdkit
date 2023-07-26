import os
from PIL import Image, ImageChops
import numpy as np

import torch
from concurrent import futures
from typing import Callable
import threading

from sdkit import Context
from sdkit.utils import log

TEST_DATA_REPO = "https://github.com/easydiffusion/sdkit-test-data.git"
TEST_DATA_FOLDER = "sdkit-test-data"
OUTPUT_FOLDER = "tmp/tests"


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


def get_image_for_device(file_path: str, device: str) -> Image:
    file_path += f"-{torch.device(device).type}.png"
    return Image.open(file_path)


def get_tensor_for_device(file_path: str, device: str) -> torch.Tensor:
    file_path += f"-{torch.device(device).type}.pt"
    return torch.load(file_path).to(device)


def assert_images_same(actual: Image, expected: Image, test_name: str = None):
    diff, hist = get_image_diff(actual, expected)
    same = not np.any(hist[2:])

    if not same and test_name:
        image_basepath = OUTPUT_FOLDER + "/" + test_name

        actual.save(image_basepath + "_1_actual.png")
        expected.save(image_basepath + "_2_expected.png")
        diff.save(image_basepath + "_3_diff.png")
        print(f">> Saved actual/expected/diff images to {image_basepath}")

    assert same, f"Histogram: {list(hist.data)}"


def run_test_on_multiple_devices(task: Callable, devices: list):
    def task_thread(device):
        threading.current_thread().name = f"sd-{device}"
        log.info(f"starting on device {device}")
        context = Context()
        context.device = device

        try:
            task(context)
        except KeyboardInterrupt:
            exit()
        except Exception as e:
            log.error(f"Error running the task!")
            raise e

    with futures.ThreadPoolExecutor(max_workers=len(devices)) as executor:
        threads = [executor.submit(task_thread, device) for device in devices]
        for f in futures.as_completed(threads):
            if f.exception():
                raise f.exception()
