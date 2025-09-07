import pytest
import torch
import torch.nn as nn
from GPT_SoVITS.BigVGAN.discriminators import (
    DiscriminatorP,
    MultiPeriodDiscriminator,
    DiscriminatorR,
    MultiResolutionDiscriminator,
    DiscriminatorB,
    MultiBandDiscriminator,
    DiscriminatorCQT,
    MultiScaleSubbandCQTDiscriminator,
    CombinedDiscriminator,
)
from GPT_SoVITS.BigVGAN.env import AttrDict
from GPT_SoVITS.BigVGAN.utils0 import get_padding # DiscriminatorP uses this

# Dummy hparams for discriminators
h_discriminator = AttrDict({
    "discriminator_channel_mult": 1,
    "use_spectral_norm": False,
    "mpd_reshapes": [2, 3, 5], # Example periods
    "resolutions": [[1024, 256, 1024], [512, 128, 512], [256, 64, 256]], # Example resolutions for MRD
    "mbd_fft_sizes": [2048, 1024, 512], # Example fft sizes for MBD
    "cqtd_filters": 32,
    "cqtd_max_filters": 1024,
    "cqtd_filters_scale": 1,
    "cqtd_dilations": [1, 2, 4],
    "cqtd_in_channels": 1,
    "cqtd_out_channels": 1,
    "sampling_rate": 22050,
    "cqtd_hop_lengths": [512, 256, 256],
    "cqtd_n_octaves": [9, 9, 9],
    "cqtd_bins_per_octaves": [24, 36, 48],
    "cqtd_normalize_volume": False,
})

# Test DiscriminatorP
def test_discriminator_p_instantiation():
    disc = DiscriminatorP(h_discriminator, period=2)
    assert isinstance(disc, DiscriminatorP)

def test_discriminator_p_forward():
    disc = DiscriminatorP(h_discriminator, period=2)
    x = torch.randn(1, 1, 16384) # Dummy audio input
    output, fmap = disc(x)
    assert isinstance(output, torch.Tensor)
    assert isinstance(fmap, list)
    assert len(fmap) > 0

# Test MultiPeriodDiscriminator
def test_multi_period_discriminator_instantiation():
    disc = MultiPeriodDiscriminator(h_discriminator)
    assert isinstance(disc, MultiPeriodDiscriminator)

def test_multi_period_discriminator_forward():
    disc = MultiPeriodDiscriminator(h_discriminator)
    y = torch.randn(1, 1, 16384)
    y_hat = torch.randn(1, 1, 16384)
    y_d_rs, y_d_gs, fmap_rs, fmap_gs = disc(y, y_hat)
    assert isinstance(y_d_rs, list)
    assert isinstance(y_d_gs, list)
    assert isinstance(fmap_rs, list)
    assert isinstance(fmap_gs, list)
    assert len(y_d_rs) > 0

# Test DiscriminatorR
def test_discriminator_r_instantiation():
    disc = DiscriminatorR(h_discriminator, resolution=[1024, 256, 1024])
    assert isinstance(disc, DiscriminatorR)

def test_discriminator_r_forward():
    disc = DiscriminatorR(h_discriminator, resolution=[1024, 256, 1024])
    x = torch.randn(1, 1, 16384)
    output, fmap = disc(x)
    assert isinstance(output, torch.Tensor)
    assert isinstance(fmap, list)
    assert len(fmap) > 0

# Test MultiResolutionDiscriminator
def test_multi_resolution_discriminator_instantiation():
    disc = MultiResolutionDiscriminator(h_discriminator)
    assert isinstance(disc, MultiResolutionDiscriminator)

def test_multi_resolution_discriminator_forward():
    disc = MultiResolutionDiscriminator(h_discriminator)
    y = torch.randn(1, 1, 16384)
    y_hat = torch.randn(1, 1, 16384)
    y_d_rs, y_d_gs, fmap_rs, fmap_gs = disc(y, y_hat)
    assert isinstance(y_d_rs, list)
    assert isinstance(y_d_gs, list)
    assert isinstance(fmap_rs, list)
    assert isinstance(fmap_gs, list)
    assert len(y_d_rs) > 0

# Test DiscriminatorB
def test_discriminator_b_instantiation():
    disc = DiscriminatorB(window_length=1024)
    assert isinstance(disc, DiscriminatorB)

def test_discriminator_b_forward():
    disc = DiscriminatorB(window_length=1024)
    x = torch.randn(1, 1, 16384)
    output, fmap = disc(x)
    assert isinstance(output, torch.Tensor)
    assert isinstance(fmap, list)
    assert len(fmap) > 0

# Test MultiBandDiscriminator
def test_multi_band_discriminator_instantiation():
    disc = MultiBandDiscriminator(h_discriminator)
    assert isinstance(disc, MultiBandDiscriminator)

def test_multi_band_discriminator_forward():
    disc = MultiBandDiscriminator(h_discriminator)
    y = torch.randn(1, 1, 16384)
    y_hat = torch.randn(1, 1, 16384)
    y_d_rs, y_d_gs, fmap_rs, fmap_gs = disc(y, y_hat)
    assert isinstance(y_d_rs, list)
    assert isinstance(y_d_gs, list)
    assert isinstance(fmap_rs, list)
    assert isinstance(fmap_gs, list)
    assert len(y_d_rs) > 0

# Test DiscriminatorCQT
def test_discriminator_cqt_instantiation():
    # Mock nnAudio.features.cqt.CQT2010v2 as it's a dependency
    # For now, just ensure it can be instantiated with dummy cfg
    # This might fail if nnAudio is not installed or if the CQT2010v2 constructor is complex
    try:
        disc = DiscriminatorCQT(h_discriminator, hop_length=512, n_octaves=9, bins_per_octave=24)
        assert isinstance(disc, DiscriminatorCQT)
    except ImportError:
        pytest.skip("nnAudio not installed, skipping DiscriminatorCQT test")

def test_discriminator_cqt_forward():
    try:
        disc = DiscriminatorCQT(h_discriminator, hop_length=512, n_octaves=9, bins_per_octave=24)
        x = torch.randn(1, 1, 16384)
        output, fmap = disc(x)
        assert isinstance(output, torch.Tensor)
        assert isinstance(fmap, list)
        assert len(fmap) > 0
    except ImportError:
        pytest.skip("nnAudio not installed, skipping DiscriminatorCQT test")

# Test MultiScaleSubbandCQTDiscriminator
def test_multi_scale_subband_cqt_discriminator_instantiation():
    try:
        disc = MultiScaleSubbandCQTDiscriminator(h_discriminator)
        assert isinstance(disc, MultiScaleSubbandCQTDiscriminator)
    except ImportError:
        pytest.skip("nnAudio not installed, skipping MultiScaleSubbandCQTDiscriminator test")

def test_multi_scale_subband_cqt_discriminator_forward():
    try:
        disc = MultiScaleSubbandCQTDiscriminator(h_discriminator)
        y = torch.randn(1, 1, 16384)
        y_hat = torch.randn(1, 1, 16384)
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = disc(y, y_hat)
        assert isinstance(y_d_rs, list)
        assert isinstance(y_d_gs, list)
        assert isinstance(fmap_rs, list)
        assert isinstance(fmap_gs, list)
        assert len(y_d_rs) > 0
    except ImportError:
        pytest.skip("nnAudio not installed, skipping MultiScaleSubbandCQTDiscriminator test")

# Test CombinedDiscriminator
def test_combined_discriminator_instantiation():
    # Create dummy discriminators to combine
    mpd = MultiPeriodDiscriminator(h_discriminator)
    mrd = MultiResolutionDiscriminator(h_discriminator)
    
    combined_disc = CombinedDiscriminator(list_discriminator=[mpd, mrd])
    assert isinstance(combined_disc, CombinedDiscriminator)

def test_combined_discriminator_forward():
    mpd = MultiPeriodDiscriminator(h_discriminator)
    mrd = MultiResolutionDiscriminator(h_discriminator)
    combined_disc = CombinedDiscriminator(list_discriminator=[mpd, mrd])

    y = torch.randn(1, 1, 16384)
    y_hat = torch.randn(1, 1, 16384)

    y_d_rs, y_d_gs, fmap_rs, fmap_gs = combined_disc(y, y_hat)
    assert isinstance(y_d_rs, list)
    assert isinstance(y_d_gs, list)
    assert isinstance(fmap_rs, list)
    assert isinstance(fmap_gs, list)
    assert len(y_d_rs) > 0
