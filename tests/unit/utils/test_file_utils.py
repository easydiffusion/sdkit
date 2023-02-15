import os
import shutil
from pathlib import Path

import pytest
import safetensors.torch
import torch
from PIL import Image

from sdkit.utils.file_utils import load_tensor_file, save_images, save_tensor_file


@pytest.fixture()
def list_fixture(faker):
    return [faker.random_digit_not_null() for _ in range(faker.random_digit_not_null())]


@pytest.mark.repeat(5)
def test_load_regular_torch_file(tmp_path, list_fixture):
    # Arrange
    file_path = Path(tmp_path, "test_model.pth")
    expected_data = torch.tensor(list_fixture)

    torch.save(expected_data, str(file_path))

    # Act
    loaded_data = load_tensor_file(str(file_path))

    # Assert
    assert torch.equal(expected_data, loaded_data)


@pytest.mark.repeat(5)
def test_load_safetensors_file(tmp_path, list_fixture):
    # Arrange
    file_path = Path(tmp_path, "test_model.safetensors")
    expected_data = {"tensor": torch.tensor(list_fixture)}

    safetensors.torch.save_file(expected_data, str(file_path))

    # Act
    loaded_data = load_tensor_file(str(file_path))

    # Assert
    assert torch.equal(expected_data["tensor"], loaded_data["tensor"])


@pytest.mark.repeat(5)
def test_save_regular_torch_file(tmp_path, list_fixture):
    # Arrange
    file_path = Path(tmp_path, "test_model.pth")
    tensor = torch.tensor(list_fixture)

    # Act
    save_tensor_file(tensor, str(file_path))

    # Assert
    assert file_path.exists()


@pytest.mark.repeat(5)
def test_save_safetensors_file(tmp_path, list_fixture):
    # Arrange
    file_path = Path(tmp_path, "test_model.safetensors")
    tensor = {"tensor": torch.tensor(list_fixture)}

    # Act
    save_tensor_file(tensor, str(file_path))

    # Assert
    assert file_path.exists()


@pytest.fixture
def image_dir_fixture(tmp_path):
    # create temporary directory for images
    dir_path = Path(tmp_path, "images")
    dir_path.mkdir()
    yield dir_path
    # cleanup
    shutil.rmtree(dir_path)


@pytest.fixture
def image_files_fixture(faker, image_dir_fixture):
    # generate some sample images
    file_names = []

    for i in range(5):
        file_name = f"test_image_{i}.jpg"
        file_path = Path(image_dir_fixture, file_name)
        color = tuple(int(c) for c in faker.rgb_color().split(","))
        img = Image.new("RGB", (100, 100), color=color)
        img.save(str(file_path))
        file_names.append(file_name)

    return file_names


def test_save_images_with_string_file_name(image_dir_fixture, image_files_fixture):
    # Arrange
    images = [Image.open(image_dir_fixture / file_name) for file_name in image_files_fixture]
    expected_file_names = [f"test_image_{i}.jpg" for i in range(len(images))]
    expected_files = [Path(image_dir_fixture, file_name) for file_name in expected_file_names]

    # Act
    save_images(images, dir_path=str(image_dir_fixture), file_name="test_image", output_format="JPEG")

    # Assert
    for file in expected_files:
        assert file.exists()


def test_save_images_with_function_file_name(image_dir_fixture, image_files_fixture):
    # Arrange
    images = [Image.open(image_dir_fixture / file_name) for file_name in image_files_fixture]
    expected_file_names = [f"test_image_{i}.jpg" for i in range(len(images))]
    expected_files = [Path(image_dir_fixture, file_name) for file_name in expected_file_names]

    # define the file name generating function
    def file_name_fn(index):
        return f"test_image_{index}.jpg"

    # Act
    save_images(images, dir_path=str(image_dir_fixture), file_name=file_name_fn, output_format="JPEG")

    # Assert
    for file in expected_files:
        print(file, type(file), file.exists())
        assert file.exists()


def test_save_images_png(image_dir_fixture, image_files_fixture):
    # Arrange
    images = [Image.open(image_dir_fixture / file_name) for file_name in image_files_fixture]
    expected_file_names = [f"test_image_{i}.png" for i in range(len(images))]
    expected_files = [Path(image_dir_fixture, file_name) for file_name in expected_file_names]

    # Act
    save_images(images, dir_path=str(image_dir_fixture), file_name="test_image", output_format="PNG")

    # Assert
    for file in expected_files:
        assert file.exists()


def test_save_images_quality(image_dir_fixture, image_files_fixture):
    # Arrange
    images = [Image.open(image_dir_fixture / file_name) for file_name in image_files_fixture]
    expected_file_names = [f"test_image_{i}.jpg" for i in range(len(images))]
    expected_files = [Path(image_dir_fixture, file_name) for file_name in expected_file_names]

    # Act
    save_images(
        images, dir_path=str(image_dir_fixture), file_name="test_image", output_format="JPEG", output_quality=50
    )

    # Assert
    for file in expected_files:
        assert file.exists()
