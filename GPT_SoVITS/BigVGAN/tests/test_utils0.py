import os
import pytest
import torch
import torch.nn as nn
from GPT_SoVITS.BigVGAN.utils0 import (
    plot_spectrogram,
    plot_spectrogram_clipped,
    init_weights,
    apply_weight_norm,
    get_padding,
    load_checkpoint,
    save_checkpoint,
    scan_checkpoint,
    save_audio,
)
from scipy.io.wavfile import read as wav_read

# Fixture for temporary directory
@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path

def test_plot_spectrogram():
    spectrogram = torch.randn(10, 20)
    fig = plot_spectrogram(spectrogram)
    assert fig is not None

def test_plot_spectrogram_clipped():
    spectrogram = torch.randn(10, 20) * 5 # Some values above clip_max
    fig = plot_spectrogram_clipped(spectrogram, clip_max=2.0)
    assert fig is not None

def test_init_weights():
    conv = nn.Conv1d(1, 1, 3)
    init_weights(conv)
    # Hard to assert exact weight values, but ensure it runs without error

def test_apply_weight_norm():
    conv = nn.Conv1d(1, 1, 3)
    apply_weight_norm(conv)
    # Hard to assert exact weight norm application, but ensure it runs without error

def test_get_padding():
    assert get_padding(3) == 1
    assert get_padding(5, dilation=2) == 4

def test_load_checkpoint(temp_dir):
    # Create a dummy checkpoint file
    dummy_data = {"key": "value"}
    checkpoint_path = os.path.join(temp_dir, "dummy_checkpoint.pt")
    torch.save(dummy_data, checkpoint_path)

    loaded_data = load_checkpoint(checkpoint_path, "cpu")
    assert loaded_data == dummy_data

def test_save_checkpoint(temp_dir):
    dummy_data = {"another_key": "another_value"}
    checkpoint_path = os.path.join(temp_dir, "saved_checkpoint.pt")
    save_checkpoint(checkpoint_path, dummy_data)
    assert os.path.exists(checkpoint_path)
    loaded_data = torch.load(checkpoint_path, map_location="cpu")
    assert loaded_data == dummy_data

def test_scan_checkpoint_pattern(temp_dir):
    # Create dummy checkpoint files
    os.makedirs(os.path.join(temp_dir, "checkpoints"), exist_ok=True)
    with open(os.path.join(temp_dir, "checkpoints", "g_00000001"), "w") as f: f.write("1")
    with open(os.path.join(temp_dir, "checkpoints", "g_00000002"), "w") as f: f.write("2")
    with open(os.path.join(temp_dir, "checkpoints", "g_00000003"), "w") as f: f.write("3")

    latest_checkpoint = scan_checkpoint(os.path.join(temp_dir, "checkpoints"), "g_")
    assert latest_checkpoint == os.path.join(temp_dir, "checkpoints", "g_00000003")

def test_scan_checkpoint_renamed_file(temp_dir):
    os.makedirs(os.path.join(temp_dir, "checkpoints"), exist_ok=True)
    renamed_path = os.path.join(temp_dir, "checkpoints", "renamed_model.pt")
    with open(renamed_path, "w") as f: f.write("renamed")
    
    latest_checkpoint = scan_checkpoint(os.path.join(temp_dir, "checkpoints"), "g_", renamed_file="renamed_model.pt")
    assert latest_checkpoint == renamed_path

def test_scan_checkpoint_none_found(temp_dir):
    os.makedirs(os.path.join(temp_dir, "checkpoints"), exist_ok=True)
    latest_checkpoint = scan_checkpoint(os.path.join(temp_dir, "checkpoints"), "g_")
    assert latest_checkpoint is None

def test_save_audio(temp_dir):
    audio_data = torch.randn(16000) # 1 second of audio at 16kHz
    output_path = os.path.join(temp_dir, "test_audio.wav")
    sample_rate = 16000
    save_audio(audio_data, output_path, sample_rate)
    assert os.path.exists(output_path)
    # Optionally, read back and check properties
    sr, data = wav_read(output_path)
    assert sr == sample_rate
    assert len(data) == len(audio_data)