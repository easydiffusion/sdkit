import random
from pathlib import Path

import pytest
import safetensors.torch
import torch

from sdkit.utils.file_utils import load_tensor_file, save_tensor_file


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
