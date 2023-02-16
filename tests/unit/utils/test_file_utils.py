import json
import random
from pathlib import Path

import piexif
import pytest
import safetensors.torch
import torch
from PIL import ExifTags, Image

from sdkit.utils.file_utils import (
    load_tensor_file,
    save_dicts,
    save_images,
    save_jpeg_exif,
    save_png_metadata,
    save_tensor_file,
    save_text_metadata,
)


@pytest.mark.repeat(5)
def test_load_regular_torch_file(tmp_path, list_of_int_fixture):
    # Arrange
    file_path = Path(tmp_path, "test_model.pth")
    expected_data = torch.tensor(list_of_int_fixture)

    torch.save(expected_data, str(file_path))

    # Act
    loaded_data = load_tensor_file(str(file_path))

    # Assert
    assert torch.equal(expected_data, loaded_data)


@pytest.mark.repeat(5)
def test_load_safetensors_file(tmp_path, list_of_int_fixture):
    # Arrange
    file_path = Path(tmp_path, "test_model.safetensors")
    expected_data = {"tensor": torch.tensor(list_of_int_fixture)}

    safetensors.torch.save_file(expected_data, str(file_path))

    # Act
    loaded_data = load_tensor_file(str(file_path))

    # Assert
    assert torch.equal(expected_data["tensor"], loaded_data["tensor"])


@pytest.mark.repeat(5)
def test_save_regular_torch_file(tmp_path, list_of_int_fixture):
    # Arrange
    file_path = Path(tmp_path, "test_model.pth")
    tensor = torch.tensor(list_of_int_fixture)

    # Act
    save_tensor_file(tensor, str(file_path))

    # Assert
    assert file_path.exists()


@pytest.mark.repeat(5)
def test_save_safetensors_file(tmp_path, list_of_int_fixture):
    # Arrange
    file_path = Path(tmp_path, "test_model.safetensors")
    tensor = {"tensor": torch.tensor(list_of_int_fixture)}

    # Act
    save_tensor_file(tensor, str(file_path))

    # Assert
    assert file_path.exists()


def test_save_images_with_string_file_name(dir_fixture, image_files_fixture):
    # Arrange
    images = [Image.open(dir_fixture / file_name) for file_name in image_files_fixture]
    expected_file_names = [f"test_image_{i}.jpg" for i in range(len(images))]
    expected_files = [Path(dir_fixture, file_name) for file_name in expected_file_names]

    # Act
    save_images(images, dir_path=str(dir_fixture), file_name="test_image", output_format="JPEG")

    # Assert
    for file in expected_files:
        assert file.exists()


def test_save_images_with_function_file_name(dir_fixture, image_files_fixture):
    # Arrange
    images = [Image.open(dir_fixture / file_name) for file_name in image_files_fixture]
    expected_file_names = [f"test_image_{i}.jpg" for i in range(len(images))]
    expected_files = [Path(dir_fixture, file_name) for file_name in expected_file_names]

    # define the file name generating function
    def file_name_fn(index):
        return f"test_image_{index}.jpg"

    # Act
    save_images(images, dir_path=str(dir_fixture), file_name=file_name_fn, output_format="JPEG")

    # Assert
    for file in expected_files:
        print(file, type(file), file.exists())
        assert file.exists()


def test_save_images_png(dir_fixture, image_files_fixture):
    # Arrange
    images = [Image.open(dir_fixture / file_name) for file_name in image_files_fixture]
    expected_file_names = [f"test_image_{i}.png" for i in range(len(images))]
    expected_files = [Path(dir_fixture, file_name) for file_name in expected_file_names]

    # Act
    save_images(images, dir_path=str(dir_fixture), file_name="test_image", output_format="PNG")

    # Assert
    for file in expected_files:
        assert file.exists()


def test_save_images_quality(dir_fixture, image_files_fixture):
    # Arrange
    images = [Image.open(dir_fixture / file_name) for file_name in image_files_fixture]
    expected_file_names = [f"test_image_{i}.jpg" for i in range(len(images))]
    expected_files = [Path(dir_fixture, file_name) for file_name in expected_file_names]

    # Act
    save_images(images, dir_path=str(dir_fixture), file_name="test_image", output_format="JPEG", output_quality=50)

    # Assert
    for file in expected_files:
        assert file.exists()


def test_save_dicts_no_directory_path():
    with pytest.raises(ValueError, match="No directory specified"):
        save_dicts(entries=[], dir_path=None)


@pytest.mark.parametrize(
    "output_format, file_format, method",
    [
        ("embed", "png", "sdkit.utils.file_utils.save_png_metadata"),
        ("embed", "jpeg", "sdkit.utils.file_utils.save_jpeg_exif"),
    ],
)
def test_save_dicts_embed(dir_fixture, metadata_fixture, faker, mocker, output_format, file_format, method):
    entries = [metadata_fixture]
    file_name = faker.word()

    def randomise_capitalisation(word: str):
        return "".join(random.choice([letter.upper(), letter.lower()]) for letter in word)

    output_format = randomise_capitalisation(output_format)
    file_format = randomise_capitalisation(file_format)

    mocked_method = mocker.patch(method)

    save_dicts(entries, dir_fixture, file_name=file_name, output_format=output_format, file_format=file_format)

    mocked_method.assert_called()


def test_save_dicts_embed_error(dir_fixture, metadata_fixture, faker):
    entries = [metadata_fixture]
    file_name = faker.word()
    file_format = faker.word()

    with pytest.raises(ValueError, match=f"Unknown format image type: {file_format}"):
        save_dicts(entries, dir_fixture, file_name=file_name, output_format="embed", file_format=file_format)


@pytest.mark.parametrize(
    "file_format",
    [
        ("txt"),
        ("json"),
    ],
)
def test_save_dicts_text(dir_fixture, metadata_fixture, faker, mocker, file_format):
    entries = [metadata_fixture]
    file_name = faker.word()

    mocked_method = mocker.patch("sdkit.utils.file_utils.save_text_metadata")

    save_dicts(entries, dir_fixture, file_name=file_name, output_format="txt", file_format=file_format)

    mocked_method.assert_called()


def test_save_text_metadata_txt(tmp_path, metadata_fixture):
    # Arrange
    output_format = "txt"
    output_file_path = Path(tmp_path, "metadata")

    # Act
    save_text_metadata(output_format, metadata_fixture, str(output_file_path))

    # Assert
    with open(f"{output_file_path}.{output_format}", "r", encoding="utf-8") as f:
        metadata_contents = f.read()
        for key, value in metadata_fixture.items():
            assert f"{key}: {value}" in metadata_contents


def test_save_text_metadata_json(tmp_path, metadata_fixture):
    # Arrange
    output_format = "json"
    output_file_path = Path(tmp_path, "metadata")

    # Act
    save_text_metadata(output_format, metadata_fixture, str(output_file_path))

    # Assert
    with open(f"{output_file_path}.{output_format}", "r", encoding="utf-8") as f:
        metadata_contents = json.load(f)
        assert metadata_contents == metadata_fixture


def test_save_text_metadata_unknown_format(tmp_path, metadata_fixture):
    # Arrange
    output_format = "xml"
    output_file_path = Path(tmp_path, "metadata")

    # Act & Assert
    with pytest.raises(ValueError, match=f"Unknown format file type {output_format}"):
        save_text_metadata(output_format, metadata_fixture, str(output_file_path))


def test_save_png_metadata(png_path_fixture, metadata_fixture):
    # Arrange
    file_format = "png"

    # Arrange
    image_file_path = str(png_path_fixture)
    expected_metadata = metadata_fixture

    # Act
    save_png_metadata(file_format, metadata_fixture, image_file_path)

    # Assert
    target_image = Image.open(f"{image_file_path}.{file_format}")
    embedded_metadata = target_image.info

    for key, val in expected_metadata.items():
        assert key in embedded_metadata
        assert embedded_metadata[key] == str(val)


def test_save_jpeg_exif(jpeg_path_fixture, metadata_fixture):
    # Arrange
    file_format = "jpg"

    # Arrange
    image_file_path = str(jpeg_path_fixture)
    expected_metadata = metadata_fixture

    # Act
    save_jpeg_exif(file_format, metadata_fixture, image_file_path)

    # Assert
    target_image = Image.open(f"{image_file_path}.{file_format}")
    exif_data = piexif.load(target_image.info["exif"])
    exif_user_comment = piexif.helper.UserComment.load(exif_data["Exif"][piexif.ExifIFD.UserComment])

    assert json.loads(exif_user_comment) == expected_metadata
