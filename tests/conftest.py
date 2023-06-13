import shutil
import os

from common import fetch_test_data, OUTPUT_FOLDER


def pytest_sessionstart(session):
    shutil.rmtree(OUTPUT_FOLDER, ignore_errors=True)
    os.makedirs(OUTPUT_FOLDER)

    fetch_test_data()
