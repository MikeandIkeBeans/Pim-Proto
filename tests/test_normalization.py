"""Tests for normalization and augmentation functions."""

import pytest
import numpy as np
from retry_lstm import (
    default_normalize, spatial_augment, 
    _pair_dist, NUM_LANDMARKS, L_SHOULDER, R_SHOULDER
)

class TestNormalization:
    """Test suite for normalization functions."""
    
    def test_pair_dist_basic(self):
        """Test basic distance calculation."""
        frame = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
        frame[L_SHOULDER] = [1.0, 0.0, 0.0]
        frame[R_SHOULDER] = [0.0, 0.0, 0.0001]  # Non-zero to avoid None return
        
        dist = _pair_dist(frame, L_SHOULDER, R_SHOULDER)
        assert dist is not None
        assert abs(dist - 1.0) < 0.01  # Distance should be ~1.0 (with small epsilon added)
    
    def test_normalize_preserves_shape(self):
        """Test shape preservation."""
        seq = np.random.randn(50, NUM_LANDMARKS, 3).astype(np.float32)
        norm = default_normalize(seq)
        assert norm.shape == seq.shape
    
    @pytest.mark.parametrize("scale", [1e-6, 1e-3, 1.0, 100.0])
    def test_normalize_various_scales(self, scale):
        """Test normalization with various input scales."""
        seq = np.random.randn(50, NUM_LANDMARKS, 3).astype(np.float32) * scale
        norm = default_normalize(seq)
        assert np.all(np.isfinite(norm))


class TestAugmentation:
    """Test suite for augmentation functions."""
    
    @pytest.mark.parametrize("rot_deg", [0, 15, 45, 90])
    def test_rotation_angles(self, rot_deg):
        """Test different rotation angles."""
        seq = np.ones((30, NUM_LANDMARKS, 3), dtype=np.float32)
        aug = spatial_augment(seq, rot_deg_max=rot_deg, p_rotate=1.0,
                            p_scale=0.0, p_noise=0.0)
        assert np.all(np.isfinite(aug))
    
    def test_augment_deterministic_with_seed(self):
        """Test reproducibility with seed."""
        seq = np.random.randn(30, NUM_LANDMARKS, 3).astype(np.float32)
        
        np.random.seed(42)
        aug1 = spatial_augment(seq)
        
        np.random.seed(42)
        aug2 = spatial_augment(seq)
        
        assert np.allclose(aug1, aug2)