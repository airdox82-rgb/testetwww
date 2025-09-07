import os
import pytest
import torch
import numpy as np
from GPT_SoVITS.BigVGAN.meldataset import (
    dynamic_range_compression,
    dynamic_range_decompression,
    dynamic_range_compression_torch,
    dynamic_range_decompression_torch,
    spectral_normalize_torch,
    spectral_de_normalize_torch,
    mel_spectrogram,
    get_mel_spectrogram,
    get_dataset_filelist,
    MAX_WAV_VALUE,
    MelDataset,
)
from GPT_SoVITS.BigVGAN.env import AttrDict
from unittest.mock import patch, MagicMock # Import for mocking

# Fixture for temporary directory
@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path

def test_dynamic_range_compression():
    x = np.array([1.0, 0.1, 0.00001, 0.000001])
    compressed = dynamic_range_compression(x)
    assert isinstance(compressed, np.ndarray)
    assert compressed[0] == np.log(1.0)
    assert compressed[2] == np.log(0.00001) # Should not be clipped

def test_dynamic_range_decompression():
    x = np.array([0.0, -2.302585]) # log(1.0), log(0.1)
    decompressed = dynamic_range_decompression(x)
    assert isinstance(decompressed, np.ndarray)
    assert np.isclose(decompressed[0], 1.0)
    assert np.isclose(decompressed[1], 0.1)

def test_dynamic_range_compression_torch():
    x = torch.tensor([1.0, 0.1, 0.00001, 0.000001])
    compressed = dynamic_range_compression_torch(x)
    assert isinstance(compressed, torch.Tensor)
    assert compressed[0] == torch.log(torch.tensor(1.0))
    assert compressed[2] == torch.log(torch.tensor(0.00001))

def test_dynamic_range_decompression_torch():
    x = torch.tensor([0.0, -2.302585])
    decompressed = dynamic_range_decompression_torch(x)
    assert isinstance(decompressed, torch.Tensor)
    assert torch.isclose(decompressed[0], torch.tensor(1.0))
    assert torch.isclose(decompressed[1], torch.tensor(0.1))

def test_spectral_normalize_torch():
    magnitudes = torch.tensor([1.0, 0.5, 0.01])
    normalized = spectral_normalize_torch(magnitudes)
    assert isinstance(normalized, torch.Tensor)
    assert torch.isclose(normalized[0], torch.log(torch.tensor(1.0)))

def test_spectral_de_normalize_torch():
    magnitudes = torch.tensor([0.0, -0.693147, -4.60517]) # log(1.0), log(0.5), log(0.01)
    denormalized = spectral_de_normalize_torch(magnitudes)
    assert isinstance(denormalized, torch.Tensor)
    assert torch.isclose(denormalized[0], torch.tensor(1.0))

def test_mel_spectrogram():
    # Minimal hparams for mel_spectrogram
    h_mel = AttrDict({
        "n_fft": 1024,
        "num_mels": 80,
        "sampling_rate": 22050,
        "hop_size": 256,
        "win_size": 1024,
        "fmin": 0,
        "fmax": 8000,
    })
    # Create a dummy audio waveform
    y = torch.randn(1, h_mel.sampling_rate * 2) # 2 seconds of audio
    mel_spec = mel_spectrogram(
        y,
        h_mel.n_fft,
        h_mel.num_mels,
        h_mel.sampling_rate,
        h_mel.hop_size,
        h_mel.win_size,
        h_mel.fmin,
        h_mel.fmax,
    )
    assert isinstance(mel_spec, torch.Tensor)
    assert mel_spec.shape[0] == 1
    assert mel_spec.shape[1] == h_mel.num_mels

def test_get_mel_spectrogram():
    h_mel = AttrDict({
        "n_fft": 1024,
        "num_mels": 80,
        "sampling_rate": 22050,
        "hop_size": 256,
        "win_size": 1024,
        "fmin": 0,
        "fmax": 8000,
    })
    y = torch.randn(1, h_mel.sampling_rate * 2)
    mel_spec = get_mel_spectrogram(y, h_mel)
    assert isinstance(mel_spec, torch.Tensor)
    assert mel_spec.shape[0] == 1
    assert mel_spec.shape[1] == h_mel.num_mels

def test_get_dataset_filelist(temp_dir):
    # Create dummy file lists and wavs directory
    wavs_dir = os.path.join(temp_dir, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)
    for i in range(5):
        with open(os.path.join(wavs_dir, f"audio_{i}.wav"), "w") as f: f.write("dummy")

    training_file_content = "audio1|text1\naudio2|text2\n"
    validation_file_content = "audio3|text3\n"
    unseen_file_content = "audio1|text1\n"

    training_file_path = os.path.join(temp_dir, "training.txt")
    validation_file_path = os.path.join(temp_dir, "validation.txt")
    unseen_file_path = os.path.join(temp_dir, "unseen.txt")

    with open(training_file_path, "w") as f: f.write(training_file_content)
    with open(validation_file_path, "w") as f: f.write(validation_file_content)
    with open(unseen_file_path, "w") as f: f.write(unseen_file_content)

    a = AttrDict({
        "input_training_file": training_file_path,
        "input_validation_file": validation_file_path,
        "input_wavs_dir": wavs_dir,
        "list_input_unseen_validation_file": [unseen_file_path],
        "list_input_unseen_wavs_dir": [wavs_dir],
    })

    training_files, validation_files, list_unseen_validation_files = get_dataset_filelist(a)

    assert len(training_files) == 2
    assert os.path.basename(training_files[0]) == "audio1.wav"
    assert len(validation_files) == 1
    assert os.path.basename(validation_files[0]) == "audio3.wav"
    assert len(list_unseen_validation_files) == 1
    assert len(list_unseen_validation_files[0]) == 1
    assert os.path.basename(list_unseen_validation_files[0][0]) == "audio1.wav"

# Test MelDataset class (init and len)
def test_meldataset_init_and_len(temp_dir):
    # Create dummy audio files
    audio_files_dir = os.path.join(temp_dir, "audio_files")
    os.makedirs(audio_files_dir, exist_ok=True)
    for i in range(5):
        with open(os.path.join(audio_files_dir, f"audio_{i}.wav"), "w") as f: f.write("dummy")

    training_files = [os.path.join(audio_files_dir, f"audio_{i}.wav") for i in range(5)]

    hparams = AttrDict({
        "n_fft": 1024,
        "num_mels": 80,
        "sampling_rate": 22050,
        "hop_size": 256,
        "win_size": 1024,
        "fmin": 0,
        "fmax": 8000,
    })

    dataset = MelDataset(
        training_files=training_files,
        hparams=hparams,
        segment_size=16000,
        n_fft=1024,
        num_mels=80,
        hop_size=256,
        win_size=1024,
        sampling_rate=22050,
        fmin=0,
        fmax=8000,
        split=True,
        shuffle=True,
        device="cpu",
        fmax_loss=None,
        fine_tuning=False,
        base_mels_path=None,
        is_seen=True,
    )
    assert len(dataset) == 5
    assert dataset.segment_size == 16000
    assert dataset.sampling_rate == 22050
    assert dataset.is_seen == True

    # Test with is_seen=False
    dataset_unseen = MelDataset(
        training_files=training_files,
        hparams=hparams,
        segment_size=16000,
        n_fft=1024,
        num_mels=80,
        hop_size=256,
        win_size=1024,
        sampling_rate=22050,
        fmin=0,
        fmax=8000,
        split=True,
        shuffle=True,
        device="cpu",
        fmax_loss=None,
        fine_tuning=False,
        base_mels_path=None,
        is_seen=False,
    )
    assert len(dataset_unseen) == 5

# Test MelDataset __getitem__ (mocking librosa)
@patch('librosa.load')
@patch('librosa.resample')
@patch('numpy.load') # For fine_tuning=True
def test_meldataset_getitem_training_split(mock_np_load, mock_librosa_resample, mock_librosa_load, temp_dir):
    # Setup dummy audio file
    audio_file_path = os.path.join(temp_dir, "audio.wav")
    with open(audio_file_path, "w") as f: f.write("dummy") # Content doesn't matter for mock

    # Mock librosa.load to return a dummy audio array and sampling rate
    mock_librosa_load.return_value = (np.random.rand(44100 * 5), 44100) # 5 seconds of audio at 44.1kHz

    # Mock librosa.resample if it's called
    mock_librosa_resample.return_value = np.random.rand(16000) # Resampled to target segment size

    training_files = [audio_file_path]

    hparams = AttrDict({
        "n_fft": 1024,
        "num_mels": 80,
        "sampling_rate": 22050,
        "hop_size": 256,
        "win_size": 1024,
        "fmin": 0,
        "fmax": 8000,
    })

    dataset = MelDataset(
        training_files=training_files,
        hparams=hparams,
        segment_size=16000,
        n_fft=1024,
        num_mels=80,
        hop_size=256,
        win_size=1024,
        sampling_rate=22050,
        fmin=0,
        fmax=8000,
        split=True, # Training split
        shuffle=False, # For deterministic testing
        device="cpu",
        fmax_loss=None,
        fine_tuning=False,
        base_mels_path=None,
        is_seen=True,
    )

    mel, audio, filename, mel_loss = dataset[0]

    assert isinstance(mel, torch.Tensor)
    assert isinstance(audio, torch.Tensor)
    assert isinstance(filename, str)
    assert isinstance(mel_loss, torch.Tensor)
    mock_librosa_load.assert_called_once_with(audio_file_path, sr=None, mono=True)

@patch('librosa.load')
@patch('librosa.resample')
@patch('numpy.load')
def test_meldataset_getitem_validation(mock_np_load, mock_librosa_resample, mock_librosa_load, temp_dir):
    audio_file_path = os.path.join(temp_dir, "audio.wav")
    with open(audio_file_path, "w") as f: f.write("dummy")
    mock_librosa_load.return_value = (np.random.rand(44100 * 5), 44100)

    training_files = [audio_file_path]
    hparams = AttrDict({
        "n_fft": 1024, "num_mels": 80, "sampling_rate": 22050, "hop_size": 256,
        "win_size": 1024, "fmin": 0, "fmax": 8000,
    })

    dataset = MelDataset(
        training_files=training_files, hparams=hparams, segment_size=16000,
        n_fft=1024, num_mels=80, hop_size=256, win_size=1024, sampling_rate=22050,
        fmin=0, fmax=8000, split=False, # Validation
        shuffle=False, device="cpu", fmax_loss=None, fine_tuning=False,
        base_mels_path=None, is_seen=True,
    )
    mel, audio, filename, mel_loss = dataset[0]
    assert isinstance(mel, torch.Tensor)

@patch('librosa.load')
@patch('librosa.resample')
@patch('numpy.load')
def test_meldataset_getitem_fine_tuning(mock_np_load, mock_librosa_resample, mock_librosa_load, temp_dir):
    audio_file_path = os.path.join(temp_dir, "audio.wav")
    with open(audio_file_path, "w") as f: f.write("dummy")
    
    # Create a dummy .npy mel file
    base_mels_path = os.path.join(temp_dir, "mels")
    os.makedirs(base_mels_path, exist_ok=True)
    dummy_mel_path = os.path.join(base_mels_path, "audio.npy")
    np.save(dummy_mel_path, np.random.rand(80, 100)) # Dummy mel spectrogram

    mock_librosa_load.return_value = (np.random.rand(22050 * 2), 22050) # Match sampling_rate
    mock_np_load.return_value = np.random.rand(80, 100) # Mock numpy load for mel

    training_files = [audio_file_path]
    hparams = AttrDict({
        "n_fft": 1024, "num_mels": 80, "sampling_rate": 22050, "hop_size": 256,
        "win_size": 1024, "fmin": 0, "fmax": 8000,
    })

    dataset = MelDataset(
        training_files=training_files, hparams=hparams, segment_size=16000,
        n_fft=1024, num_mels=80, hop_size=256, win_size=1024, sampling_rate=22050,
        fmin=0, fmax=8000, split=True, shuffle=False, device="cpu", fmax_loss=None,
        fine_tuning=True, # Fine-tuning
        base_mels_path=base_mels_path,
        is_seen=True,
    )
    mel, audio, filename, mel_loss = dataset[0]
    assert isinstance(mel, torch.Tensor)
    mock_np_load.assert_called_once()

@patch('librosa.load')
@patch('librosa.resample')
@patch('numpy.load')
def test_meldataset_getitem_error_handling(mock_np_load, mock_librosa_resample, mock_librosa_load, temp_dir):
    audio_file_path = os.path.join(temp_dir, "audio.wav")
    with open(audio_file_path, "w") as f: f.write("dummy")
    
    # Make librosa.load raise an exception
    mock_librosa_load.side_effect = Exception("Mock librosa load error")

    training_files = [audio_file_path, os.path.join(temp_dir, "audio2.wav")] # Need another file for random pick
    with open(os.path.join(temp_dir, "audio2.wav"), "w") as f: f.write("dummy")

    hparams = AttrDict({
        "n_fft": 1024, "num_mels": 80, "sampling_rate": 22050, "hop_size": 256,
        "win_size": 1024, "fmin": 0, "fmax": 8000,
    })

    dataset = MelDataset(
        training_files=training_files, hparams=hparams, segment_size=16000,
        n_fft=1024, num_mels=80, hop_size=256, win_size=1024, sampling_rate=22050,
        fmin=0, fmax=8000, split=True, shuffle=False, device="cpu", fmax_loss=None,
        fine_tuning=False, # Not fine-tuning, so it should handle error
        base_mels_path=None, is_seen=True,
    )
    # Expect it to not raise an error, but return a random sample
    mel, audio, filename, mel_loss = dataset[0]
    assert isinstance(mel, torch.Tensor)
    # Assert that librosa.load was called more than once (due to retry)
    assert mock_librosa_load.call_count > 1

@patch('librosa.load')
@patch('librosa.resample')
@patch('numpy.load')
def test_meldataset_getitem_fine_tuning_error_handling(mock_np_load, mock_librosa_resample, mock_librosa_load, temp_dir):
    audio_file_path = os.path.join(temp_dir, "audio.wav")
    with open(audio_file_path, "w") as f: f.write("dummy")
    
    mock_librosa_load.side_effect = Exception("Mock librosa load error")

    training_files = [audio_file_path]
    hparams = AttrDict({
        "n_fft": 1024, "num_mels": 80, "sampling_rate": 22050, "hop_size": 256,
        "win_size": 1024, "fmin": 0, "fmax": 8000,
    })

    dataset = MelDataset(
        training_files=training_files, hparams=hparams, segment_size=16000,
        n_fft=1024, num_mels=80, hop_size=256, win_size=1024, sampling_rate=22050,
        fmin=0, fmax=8000, split=True, shuffle=False, device="cpu", fmax_loss=None,
        fine_tuning=True, # Fine-tuning, so it should raise error
        base_mels_path=None, is_seen=True,
    )
    with pytest.raises(Exception, match="Mock librosa load error"):
        dataset[0]