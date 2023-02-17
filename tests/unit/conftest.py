import shutil
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture()
def list_of_int_fixture(faker):
    return [faker.random_digit_not_null() for _ in range(faker.random_digit_not_null())]


@pytest.fixture
def dir_fixture(tmp_path, faker):
    # create temporary directory for images
    dir_path = Path(tmp_path, faker.word())
    dir_path.mkdir()
    yield dir_path

    # cleanup
    shutil.rmtree(dir_path)


@pytest.fixture
def image_fixture(faker):
    color = tuple(int(c) for c in faker.rgb_color().split(","))
    return Image.new("RGB", (100, 100), color=color)


@pytest.fixture
def image_files_fixture(dir_fixture, image_fixture):
    # generate some sample images
    file_names = []

    for i in range(5):
        file_name = f"test_image_{i}.jpg"
        file_path = Path(dir_fixture, file_name)
        image_fixture.save(str(file_path))
        file_names.append(file_name)

    return file_names


@pytest.fixture
def png_path_fixture(faker, dir_fixture, image_fixture):
    # Create a temporary file path
    file_path = Path(dir_fixture, f"test_image_{faker.word()}")
    # Create a new image file
    image_fixture.save(f"{file_path}.png", "PNG")

    return str(file_path)


@pytest.fixture
def jpeg_path_fixture(faker, dir_fixture, image_fixture):
    # Create a temporary file path
    file_path = Path(dir_fixture, f"test_image_{faker.word()}")
    # Create a new image file
    image_fixture.save(f"{file_path}.jpg", "JPEG")

    return str(file_path)


@pytest.fixture
def metadata_fixture(faker):
    metadata = {}
    for _ in range(faker.random_digit_not_null()):
        key = faker.word()
        value = faker.random_element(elements=(faker.word(), faker.pyint()))
        metadata[key] = value

    return metadata
