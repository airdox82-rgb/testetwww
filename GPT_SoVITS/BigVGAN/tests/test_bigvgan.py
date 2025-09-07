import torch
from GPT_SoVITS.BigVGAN.bigvgan import BigVGAN, AMPBlock1, AMPBlock2
from GPT_SoVITS.BigVGAN.env import AttrDict
from GPT_SoVITS.BigVGAN.activations import Snake, SnakeBeta

# Minimal hyperparameters for testing
h_config = {
    "num_mels": 80,
    "upsample_initial_channel": 512,
    "upsample_rates": [8, 8, 2, 2],
    "upsample_kernel_sizes": [16, 16, 4, 4],
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "activation": "snake",
    "snake_logscale": True,
    "resblock": "1",
    "use_bias_at_final": True,
    "use_tanh_at_final": True,
}
h = AttrDict(h_config)

# hparams for snakebeta activation
h_config_snakebeta = h_config.copy()
h_config_snakebeta["activation"] = "snakebeta"
h_snakebeta = AttrDict(h_config_snakebeta)

# hparams for AMPBlock2
h_config_amp2 = h_config.copy()
h_config_amp2["resblock"] = "2"
h_amp2 = AttrDict(h_config_amp2)


def test_bigvgan_instantiation():
    model = BigVGAN(h)
    assert isinstance(model, BigVGAN)

def test_bigvgan_forward():
    model = BigVGAN(h)
    # Create a dummy input tensor (batch_size, num_mels, sequence_length)
    x = torch.randn(1, h.num_mels, 100)
    output = model(x)
    assert output.shape[0] == x.shape[0]
    assert output.shape[1] == 1 # Output channel should be 1 (audio waveform)

def test_bigvgan_remove_weight_norm():
    model = BigVGAN(h)
    # Apply weight norm (it's applied during init)
    # Then remove it
    model.remove_weight_norm()
    # Check if weight norm is removed (this is hard to assert directly without inspecting internal modules)
    # For now, just ensure the method runs without error.

def test_bigvgan_remove_weight_norm_already_removed():
    model = BigVGAN(h)
    model.remove_weight_norm() # First removal
    model.remove_weight_norm() # Second removal to hit the except block
    # Ensure it runs without error

def test_ampblock1_instantiation():
    block = AMPBlock1(h, channels=h.upsample_initial_channel // (2**len(h.upsample_rates)), activation="snake")
    assert isinstance(block, AMPBlock1)

def test_ampblock2_instantiation():
    block = AMPBlock2(h_amp2, channels=h_amp2.upsample_initial_channel // (2**len(h_amp2.upsample_rates)), activation="snake")
    assert isinstance(block, AMPBlock2)

def test_ampblock1_forward():
    block = AMPBlock1(h, channels=h.upsample_initial_channel // (2**len(h.upsample_rates)), activation="snake")
    x = torch.randn(1, h.upsample_initial_channel // (2**len(h.upsample_rates)), 100)
    output = block(x)
    assert output.shape == x.shape

def test_ampblock2_forward():
    block = AMPBlock2(h_amp2, channels=h_amp2.upsample_initial_channel // (2**len(h_amp2.upsample_rates)), activation="snake")
    x = torch.randn(1, h_amp2.upsample_initial_channel // (2**len(h_amp2.upsample_rates)), 100)
    output = block(x)
    assert output.shape == x.shape

def test_ampblock1_remove_weight_norm():
    block = AMPBlock1(h, channels=h.upsample_initial_channel // (2**len(h.upsample_rates)), activation="snake")
    block.remove_weight_norm()
    # Ensure it runs without error

def test_ampblock2_remove_weight_norm():
    block = AMPBlock2(h_amp2, channels=h_amp2.upsample_initial_channel // (2**len(h_amp2.upsample_rates)), activation="snake")
    block.remove_weight_norm()
    # Ensure it runs without error

def test_bigvgan_snakebeta_activation():
    model = BigVGAN(h_snakebeta)
    assert isinstance(model, BigVGAN)
    x = torch.randn(1, h_snakebeta.num_mels, 100)
    output = model(x)
    assert output.shape[0] == x.shape[0]
    assert output.shape[1] == 1

def test_ampblock1_snakebeta_activation():
    block = AMPBlock1(h_snakebeta, channels=h_snakebeta.upsample_initial_channel // (2**len(h_snakebeta.upsample_rates)), activation="snakebeta")
    assert isinstance(block, AMPBlock1)
    x = torch.randn(1, h_snakebeta.upsample_initial_channel // (2**len(h_snakebeta.upsample_rates)), 100)
    output = block(x)
    assert output.shape == x.shape

def test_ampblock2_snakebeta_activation():
    block = AMPBlock2(h_snakebeta, channels=h_snakebeta.upsample_initial_channel // (2**len(h_snakebeta.upsample_rates)), activation="snakebeta")
    assert isinstance(block, AMPBlock2)
    x = torch.randn(1, h_snakebeta.upsample_initial_channel // (2**len(h_snakebeta.upsample_rates)), 100)
    output = block(x)
    assert output.shape == x.shape
