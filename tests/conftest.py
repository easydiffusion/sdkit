import shutil
import os

from common import fetch_test_data, OUTPUT_FOLDER


def pytest_sessionstart(session):
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    for filename in os.listdir(OUTPUT_FOLDER):
        filepath = os.path.join(OUTPUT_FOLDER, filename)
        try:
            shutil.rmtree(filepath)
        except:
            os.remove(filepath)

    fetch_test_data()
