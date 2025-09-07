import os
import shutil
import pytest
from GPT_SoVITS.BigVGAN.env import build_env

# Create a temporary directory for testing
@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path

def test_build_env_copy_file(temp_dir):
    # Create a dummy config file
    config_content = "test_config_content"
    config_file_path = os.path.join(temp_dir, "test_config.json")
    with open(config_file_path, "w") as f:
        f.write(config_content)

    # Define target path and config name
    target_path = os.path.join(temp_dir, "output_dir")
    config_name = "copied_config.json"

    # Call build_env
    build_env(config_file_path, config_name, target_path)

    # Assert that the file was copied
    assert os.path.exists(os.path.join(target_path, config_name))
    with open(os.path.join(target_path, config_name), "r") as f:
        assert f.read() == config_content

def test_build_env_no_copy(temp_dir):
    # Define target path and config name
    target_path = os.path.join(temp_dir, "output_dir")
    config_name = "test_config.json"
    t_path = os.path.join(target_path, config_name)

    # Create the target directory and file directly
    os.makedirs(target_path, exist_ok=True)
    with open(t_path, "w") as f:
        f.write("existing_content")

    # Call build_env with config == t_path
    build_env(t_path, config_name, target_path)

    # Assert that the file was NOT copied (content remains the same)
    with open(t_path, "r") as f:
        assert f.read() == "existing_content"
