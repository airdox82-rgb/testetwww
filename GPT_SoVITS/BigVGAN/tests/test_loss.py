import pytest
import torch
import torch.nn as nn
from GPT_SoVITS.BigVGAN.loss import (
    MultiScaleMelSpectrogramLoss,
    feature_loss,
    discriminator_loss,
    generator_loss,
)
from GPT_SoVITS.BigVGAN.env import AttrDict # For AttrDict if needed by MultiScaleMelSpectrogramLoss

# Test MultiScaleMelSpectrogramLoss
def test_multiscale_mel_spectrogram_loss_init():
    loss_fn = MultiScaleMelSpectrogramLoss(sampling_rate=22050)
    assert isinstance(loss_fn, nn.Module)

def test_multiscale_mel_spectrogram_loss_forward():
    sampling_rate = 22050
    loss_fn = MultiScaleMelSpectrogramLoss(sampling_rate=sampling_rate)

    # Dummy input tensors (batch_size, channels, sequence_length)
    # Waveforms should be in [-1, 1] range
    x = torch.randn(1, 1, sampling_rate * 2) # 2 seconds of audio
    y = torch.randn(1, 1, sampling_rate * 2) # 2 seconds of audio

    loss = loss_fn(x, y)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0 # Loss should be non-negative

def test_feature_loss():
    # Dummy feature maps (List[List[torch.Tensor]])
    # Each inner list represents a layer, each tensor is a feature map
    fmap_r = [
        [torch.randn(1, 64, 100), torch.randn(1, 128, 50)],
        [torch.randn(1, 256, 25)],
    ]
    fmap_g = [
        [torch.randn(1, 64, 100), torch.randn(1, 128, 50)],
        [torch.randn(1, 256, 25)],
    ]

    loss = feature_loss(fmap_r, fmap_g)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0

def test_discriminator_loss():
    # Dummy discriminator outputs
    disc_real_outputs = [torch.randn(1, 1), torch.randn(1, 1)]
    disc_generated_outputs = [torch.randn(1, 1), torch.randn(1, 1)]

    loss, r_losses, g_losses = discriminator_loss(disc_real_outputs, disc_generated_outputs)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0
    assert isinstance(r_losses, list)
    assert isinstance(g_losses, list)
    assert len(r_losses) == len(disc_real_outputs)
    assert len(g_losses) == len(disc_generated_outputs)

def test_generator_loss():
    # Dummy discriminator outputs for generator
    disc_outputs = [torch.randn(1, 1), torch.randn(1, 1)]

    loss, gen_losses = generator_loss(disc_outputs)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0
    assert isinstance(gen_losses, list)
    assert len(gen_losses) == len(disc_outputs)
