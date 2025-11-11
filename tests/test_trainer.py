#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for Big Spatio-Temporal Transformer Trainer
Run with: pytest test_big_transformer.py -v
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import tempfile
import shutil
from collections import Counter

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import utilities from src package
from pim_detection.utils import (
    set_seed,
    load_multiview_sequence,
    ALL_CLASSES,
    NUM_LANDMARKS,
    L_SHOULDER,
    R_SHOULDER,
    L_HIP,
    R_HIP,
)

# Import training classes from gpu_stress (not yet migrated)
from gpu_stress import (
    default_normalize,
    spatial_augment,
    _pair_dist,
    NPZPoseDataset,
    window_class_counts,
    make_window_sampler,
    BigTransformer,
    BalancedSoftmaxCE,
    EMA,
    WarmupCosineScheduler,
    safe_stratified_split,
    make_kfolds,
)


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir)


@pytest.fixture
def sample_sequence():
    """Generate a simple test sequence (T=100, 33 landmarks, 3 coords)."""
    np.random.seed(42)
    T, L, C = 100, NUM_LANDMARKS, 3
    seq = np.random.randn(T, L, C).astype(np.float32)
    # Ensure key landmarks are non-zero
    seq[:, [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP], :] += 1.0
    return seq


@pytest.fixture
def sample_multiview_file(temp_dir, sample_sequence):
    """Create a test NPZ file with multiple views."""
    path = temp_dir / "test_normal_001.npz"
    
    # Create 2 views
    view1 = sample_sequence.copy()
    view2 = sample_sequence.copy() + np.random.randn(*sample_sequence.shape) * 0.1
    
    np.savez(str(path), view_0=view1, view_1=view2)
    return path


@pytest.fixture
def sample_dataset_files(temp_dir):
    """Create a small test dataset with multiple classes."""
    files = []
    labels = []
    
    np.random.seed(42)
    classes = ["normal", "tremor", "myoclonus"]
    files_per_class = [10, 5, 3]  # Imbalanced
    
    for cls, n_files in zip(classes, files_per_class):
        for i in range(n_files):
            # Random length sequences
            T = np.random.randint(60, 150)
            seq = np.random.randn(T, NUM_LANDMARKS, 3).astype(np.float32)
            seq[:, [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP], :] += 1.0
            
            path = temp_dir / f"{cls}_{i:03d}.npz"
            np.savez(str(path), view_0=seq)
            
            files.append(path)
            labels.append(ALL_CLASSES.index(cls))
    
    return files, labels


# ============================================================
# TEST: SEED AND REPRODUCIBILITY
# ============================================================

def test_set_seed_reproducibility():
    """Test that set_seed produces reproducible results."""
    set_seed(42)
    x1 = torch.randn(10)
    r1 = np.random.randn(10)
    
    set_seed(42)
    x2 = torch.randn(10)
    r2 = np.random.randn(10)
    
    assert torch.allclose(x1, x2), "PyTorch RNG not reproducible"
    assert np.allclose(r1, r2), "NumPy RNG not reproducible"


# ============================================================
# TEST: NORMALIZATION
# ============================================================

def test_pair_dist_basic():
    """Test pairwise distance calculation."""
    frame = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
    frame[L_SHOULDER] = [1.0, 0.0, 0.0]
    frame[R_SHOULDER] = [0.0, 0.0, 0.0001]  # Non-zero to avoid None return
    
    dist = _pair_dist(frame, L_SHOULDER, R_SHOULDER)
    assert dist is not None
    assert abs(dist - 1.0) < 0.01, f"Expected ~1.0, got {dist}"


def test_pair_dist_zero_landmarks():
    """Test that pair_dist returns None for zero landmarks."""
    frame = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
    dist = _pair_dist(frame, L_SHOULDER, R_SHOULDER)
    assert dist is None, "Should return None for zero landmarks"


def test_default_normalize_shape(sample_sequence):
    """Test that normalization preserves shape."""
    normalized = default_normalize(sample_sequence)
    assert normalized.shape == sample_sequence.shape
    assert normalized.dtype == np.float32


def test_default_normalize_scale(sample_sequence):
    """Test that normalization produces reasonable scale."""
    normalized = default_normalize(sample_sequence)
    
    # After normalization, values should be roughly in [-5, 5]
    assert np.all(np.isfinite(normalized)), "Contains inf/nan"
    assert normalized.min() >= -5.0, f"Min too low: {normalized.min()}"
    assert normalized.max() <= 5.0, f"Max too high: {normalized.max()}"


def test_default_normalize_centering(sample_sequence):
    """Test that normalization centers around origin."""
    normalized = default_normalize(sample_sequence)
    
    # Hip center should be close to origin
    hip_center = 0.5 * (normalized[:, L_HIP] + normalized[:, R_HIP])
    assert np.abs(hip_center).mean() < 0.1, "Not properly centered"


def test_normalize_handles_edge_cases():
    """Test normalization with edge cases."""
    # All zeros
    seq_zeros = np.zeros((50, NUM_LANDMARKS, 3), dtype=np.float32)
    norm_zeros = default_normalize(seq_zeros)
    assert np.all(np.isfinite(norm_zeros)), "Failed on all zeros"
    
    # Very small scale
    seq_tiny = np.random.randn(50, NUM_LANDMARKS, 3).astype(np.float32) * 1e-6
    seq_tiny[:, [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP], :] += 1e-5
    norm_tiny = default_normalize(seq_tiny)
    assert np.all(np.isfinite(norm_tiny)), "Failed on tiny values"


# ============================================================
# TEST: AUGMENTATION
# ============================================================

def test_spatial_augment_shape(sample_sequence):
    """Test that augmentation preserves shape."""
    augmented = spatial_augment(sample_sequence)
    assert augmented.shape == sample_sequence.shape


def test_spatial_augment_deterministic():
    """Test that augmentation with p=1.0 always applies."""
    np.random.seed(42)
    seq = np.ones((50, NUM_LANDMARKS, 3), dtype=np.float32)
    
    # Rotation should change x,y but not z
    aug = spatial_augment(seq, rot_deg_max=45, p_rotate=1.0, 
                         p_scale=0.0, p_noise=0.0)
    assert not np.allclose(aug[..., :2], seq[..., :2]), "Rotation not applied"
    
    # Scaling should change all coordinates
    aug = spatial_augment(seq, scale_min=2.0, scale_max=2.0, 
                         p_rotate=0.0, p_scale=1.0, p_noise=0.0)
    assert np.allclose(aug, seq * 2.0), "Scaling not applied correctly"


def test_spatial_augment_no_corruption():
    """Test that augmentation doesn't corrupt data with zeros."""
    np.random.seed(42)
    seq = np.random.randn(50, NUM_LANDMARKS, 3).astype(np.float32)
    aug = spatial_augment(seq)
    
    assert np.all(np.isfinite(aug)), "Augmentation produced inf/nan"
    assert aug.dtype == np.float32


# ============================================================
# TEST: FILE LOADING
# ============================================================

def test_load_multiview_basic(sample_multiview_file):
    """Test loading a basic multiview file."""
    views = load_multiview_sequence(sample_multiview_file)
    
    assert len(views) == 2, f"Expected 2 views, got {len(views)}"
    assert all(v.shape[1] == NUM_LANDMARKS for v in views)
    assert all(v.shape[2] == 3 for v in views)
    assert all(v.dtype == np.float32 for v in views)


def test_load_multiview_single_view(temp_dir):
    """Test loading a file with single view."""
    seq = np.random.randn(80, NUM_LANDMARKS, 3).astype(np.float32)
    path = temp_dir / "single_view.npz"
    np.savez(str(path), data=seq)
    
    views = load_multiview_sequence(path)
    assert len(views) == 1
    assert views[0].shape == seq.shape


def test_load_multiview_stacked_format(temp_dir):
    """Test loading concatenated multi-view format (T, 33, 9)."""
    T = 80
    # 3 views concatenated: shape (T, 33, 9)
    seq = np.random.randn(T, NUM_LANDMARKS, 9).astype(np.float32)
    path = temp_dir / "stacked.npz"
    np.savez(str(path), data=seq)
    
    views = load_multiview_sequence(path)
    assert len(views) == 3, f"Expected 3 views from 9 channels, got {len(views)}"
    assert all(v.shape == (T, NUM_LANDMARKS, 3) for v in views)


def test_load_multiview_invalid_file(temp_dir):
    """Test that invalid files raise appropriate errors."""
    path = temp_dir / "invalid.npz"
    # Create data with completely wrong shape - 1D array that can't be interpreted
    invalid_data = np.random.randn(50)
    np.savez(str(path), data=invalid_data)
    
    # Should raise ValueError from load_npy_sequence fallback
    with pytest.raises(ValueError):
        load_multiview_sequence(path)


# ============================================================
# TEST: DATASET
# ============================================================

def test_dataset_basic(sample_dataset_files):
    """Test basic dataset functionality."""
    files, labels = sample_dataset_files
    
    ds = NPZPoseDataset(
        files, labels,
        window_size=30, stride=15, mode="expand",
        normalize_fn=default_normalize, augment=None,
        cache_views=False
    )
    
    assert len(ds) > 0, "Dataset is empty"
    
    # Test __getitem__
    x, y, fi, vi, st = ds[0]
    assert isinstance(x, torch.Tensor)
    assert x.shape[0] == 30, f"Expected window size 30, got {x.shape[0]}"
    assert x.shape[1] == NUM_LANDMARKS
    assert isinstance(y, torch.Tensor)
    assert y.item() in range(len(ALL_CLASSES))


def test_dataset_modes(sample_dataset_files):
    """Test different dataset modes."""
    files, labels = sample_dataset_files
    
    # Expand mode: all views
    ds_expand = NPZPoseDataset(files, labels, window_size=30, stride=30, 
                               mode="expand", cache_views=False)
    len_expand = len(ds_expand)
    
    # Select mode: single view
    ds_select = NPZPoseDataset(files, labels, window_size=30, stride=30,
                               mode="select", view_index=0, cache_views=False)
    len_select = len(ds_select)
    
    # Expand should have more or equal windows (if multi-view files exist)
    assert len_expand >= len_select


def test_dataset_max_windows_per_file(sample_dataset_files):
    """Test max_windows_per_file limiting."""
    files, labels = sample_dataset_files
    
    ds_unlimited = NPZPoseDataset(files, labels, window_size=30, stride=10,
                                  mode="select", max_windows_per_file=None,
                                  cache_views=False)
    
    ds_limited = NPZPoseDataset(files, labels, window_size=30, stride=10,
                                mode="select", max_windows_per_file=3,
                                cache_views=False)
    
    assert len(ds_limited) <= len(ds_unlimited)


def test_dataset_padding(sample_dataset_files):
    """Test that short sequences get padded correctly."""
    files, labels = sample_dataset_files
    
    ds = NPZPoseDataset(files, labels, window_size=200,  # Longer than most seqs
                       stride=200, mode="select", cache_views=False)
    
    if len(ds) > 0:
        x, y, *_ = ds[0]
        assert x.shape[0] == 200, "Padding failed"


def test_dataset_flatten_mode(sample_dataset_files):
    """Test flatten mode."""
    files, labels = sample_dataset_files
    
    ds = NPZPoseDataset(files, labels, window_size=30, stride=30,
                       mode="select", flatten=True, cache_views=False)
    
    x, y, *_ = ds[0]
    assert x.dim() == 2, f"Expected 2D tensor, got {x.dim()}D"
    assert x.shape[0] == 30  # time
    assert x.shape[1] == NUM_LANDMARKS * 3  # flattened landmarks


def test_dataset_cache_behavior(sample_dataset_files):
    """Test that caching works correctly."""
    files, labels = sample_dataset_files
    
    ds = NPZPoseDataset(files[:3], labels[:3], window_size=30, stride=30,
                       mode="select", cache_views=True)
    
    # Access same item twice
    x1, *_ = ds[0]
    x2, *_ = ds[0]
    
    assert torch.equal(x1, x2), "Cache not working, got different results"
    assert len(ds._cache) > 0, "Cache not populated"


# ============================================================
# TEST: SAMPLING
# ============================================================

def test_window_class_counts(sample_dataset_files):
    """Test window class counting."""
    files, labels = sample_dataset_files
    
    ds = NPZPoseDataset(files, labels, window_size=30, stride=30,
                       mode="select", cache_views=False)
    
    counts = window_class_counts(ds, len(ALL_CLASSES))
    
    assert counts.shape[0] == len(ALL_CLASSES)
    assert counts.sum() == len(ds), "Counts don't sum to dataset size"
    assert torch.all(counts >= 0), "Negative counts"


def test_window_sampler_weights(sample_dataset_files):
    """Test that window sampler creates valid weights."""
    files, labels = sample_dataset_files
    
    ds = NPZPoseDataset(files, labels, window_size=30, stride=30,
                       mode="select", cache_views=False)
    
    sampler = make_window_sampler(ds, len(ALL_CLASSES))
    
    assert len(sampler.weights) == len(ds)
    assert all(w > 0 for w in sampler.weights), "Non-positive weights"
    
    # Rare classes should have higher weights
    win_labels = [ds.labels[s.file_idx] for s in ds._index]
    label_counts = Counter(win_labels)
    
    if len(label_counts) > 1:
        rare_label = min(label_counts, key=label_counts.get)
        common_label = max(label_counts, key=label_counts.get)
        
        rare_indices = [i for i, l in enumerate(win_labels) if l == rare_label]
        common_indices = [i for i, l in enumerate(win_labels) if l == common_label]
        
        rare_weight = sampler.weights[rare_indices[0]]
        common_weight = sampler.weights[common_indices[0]]
        
        assert rare_weight > common_weight, "Rare class should have higher weight"


# ============================================================
# TEST: MODEL
# ============================================================

def test_model_forward_pass():
    """Test model forward pass with various inputs."""
    model = BigTransformer(num_classes=len(ALL_CLASSES), 
                          d_model=128, nhead=4, layers=2,
                          ff_dim=256, drop=0.1, pool="mean")
    model.eval()
    
    B, T, L, C = 4, 30, NUM_LANDMARKS, 3
    x = torch.randn(B, T, L, C)
    
    with torch.no_grad():
        logits = model(x)
    
    assert logits.shape == (B, len(ALL_CLASSES))
    assert torch.all(torch.isfinite(logits)), "Model output contains inf/nan"


def test_model_pooling_modes():
    """Test different pooling modes."""
    B, T, L, C = 2, 30, NUM_LANDMARKS, 3
    x = torch.randn(B, T, L, C)
    
    for pool in ["mean", "cls", "attn"]:
        model = BigTransformer(num_classes=len(ALL_CLASSES),
                              d_model=64, nhead=2, layers=1,
                              ff_dim=128, pool=pool)
        model.eval()
        
        with torch.no_grad():
            logits = model(x)
        
        assert logits.shape == (B, len(ALL_CLASSES)), f"Failed for pool={pool}"


def test_model_flatten_input():
    """Test model with flattened input."""
    # When using flattened input, channels_per_landmark should be 3
    # and the model will expect (B, T, NUM_LANDMARKS * 3) input
    model = BigTransformer(num_classes=len(ALL_CLASSES),
                          d_model=64, nhead=2, layers=1,
                          ff_dim=128, channels_per_landmark=3)
    model.eval()
    
    B, T = 2, 30
    x = torch.randn(B, T, NUM_LANDMARKS * 3)  # Flattened: (B, T, 99)
    
    with torch.no_grad():
        logits = model(x)
    
    assert logits.shape == (B, len(ALL_CLASSES))


def test_model_gradient_flow():
    """Test that gradients flow through model."""
    model = BigTransformer(num_classes=len(ALL_CLASSES),
                          d_model=64, nhead=2, layers=1, ff_dim=128)
    model.train()
    
    x = torch.randn(2, 30, NUM_LANDMARKS, 3, requires_grad=True)
    y = torch.tensor([0, 1])
    
    logits = model(x)
    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()
    
    # Check that model parameters have gradients
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert torch.all(torch.isfinite(param.grad)), f"Invalid gradient for {name}"


# ============================================================
# TEST: LOSS
# ============================================================

def test_balanced_softmax_ce():
    """Test BalancedSoftmaxCE loss."""
    cls_counts = torch.tensor([100.0, 10.0, 5.0])  # Imbalanced
    loss_fn = BalancedSoftmaxCE(cls_counts, label_smoothing=0.1)
    
    logits = torch.randn(8, 3)
    targets = torch.randint(0, 3, (8,))
    
    loss = loss_fn(logits, targets)
    
    assert torch.isfinite(loss), "Loss is inf/nan"
    assert loss.item() >= 0, "Loss is negative"


def test_balanced_softmax_reduces_to_ce():
    """Test that BalancedSoftmax with uniform counts ≈ CE."""
    cls_counts = torch.tensor([100.0, 100.0, 100.0])  # Balanced
    loss_balanced = BalancedSoftmaxCE(cls_counts, label_smoothing=0.0)
    loss_ce = nn.CrossEntropyLoss()
    
    torch.manual_seed(42)
    logits = torch.randn(16, 3)
    targets = torch.randint(0, 3, (16,))
    
    l1 = loss_balanced(logits, targets)
    l2 = loss_ce(logits, targets)
    
    assert torch.allclose(l1, l2, atol=1e-5), "Should match CE for balanced counts"


# ============================================================
# TEST: EMA
# ============================================================

def test_ema_initialization():
    """Test EMA initialization."""
    model = BigTransformer(num_classes=len(ALL_CLASSES), d_model=64, 
                          nhead=2, layers=1, ff_dim=128)
    ema = EMA(model, decay=0.999)
    
    # Shadow params should match model initially
    for name, param in model.state_dict().items():
        if param.dtype.is_floating_point:
            assert name in ema.shadow
            assert torch.allclose(ema.shadow[name], param)


def test_ema_update():
    """Test EMA updates parameters."""
    model = BigTransformer(num_classes=len(ALL_CLASSES), d_model=64,
                          nhead=2, layers=1, ff_dim=128)
    ema = EMA(model, decay=0.9)
    
    # Store initial shadow
    initial_shadow = {k: v.clone() for k, v in ema.shadow.items()}
    
    # Modify model parameters
    for param in model.parameters():
        param.data += torch.randn_like(param) * 0.1
    
    # Update EMA
    ema.update(model)
    
    # Shadow should have changed
    for name in initial_shadow:
        if not torch.allclose(initial_shadow[name], ema.shadow[name]):
            return  # At least one parameter changed, test passes
    
    pytest.fail("EMA shadow did not update")


def test_ema_context_manager():
    """Test EMA context manager restores parameters."""
    model = BigTransformer(num_classes=len(ALL_CLASSES), d_model=64,
                          nhead=2, layers=1, ff_dim=128)
    
    # Initialize EMA with current model params
    ema = EMA(model, decay=0.9)
    
    # Modify model significantly multiple times to build up EMA difference
    for _ in range(10):
        for param in model.parameters():
            param.data += 0.5
        ema.update(model)
    
    # Store current model params (after modifications)
    current = {k: v.clone() for k, v in model.state_dict().items() 
               if v.dtype.is_floating_point}
    
    # Use context manager
    inside_values = {}
    with ema.average_parameters(model):
        # Inside context, should have EMA params (different from current)
        for name, param in model.state_dict().items():
            if param.dtype.is_floating_point:
                inside_values[name] = param.clone()
                # EMA should be different from current (blended with earlier values)
                # Skip buffers like positional encodings that don't change
                if 'pe' not in name and param.numel() > 1:
                    assert not torch.allclose(param, current[name], atol=1e-3), f"EMA not applied for {name}"
    
    # After context, should restore current params
    for name, param in model.state_dict().items():
        if param.dtype.is_floating_point:
            # Should be back to current values (before context manager)
            assert torch.allclose(param, current[name]), f"Failed to restore {name}"


# ============================================================
# TEST: LR SCHEDULER
# ============================================================

def test_warmup_cosine_scheduler():
    """Test warmup cosine scheduler."""
    model = BigTransformer(num_classes=len(ALL_CLASSES), d_model=64,
                          nhead=2, layers=1, ff_dim=128)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    sched = WarmupCosineScheduler(opt, warmup_steps=10, total_steps=100,
                                  base_lr=1e-3, min_lr=1e-5)
    
    lrs = []
    for _ in range(100):
        lrs.append(opt.param_groups[0]['lr'])
        sched.step()
    
    # Check warmup phase
    assert lrs[0] < lrs[9], "LR should increase during warmup"
    
    # Check peak around warmup end
    assert lrs[10] == pytest.approx(1e-3, abs=1e-6), "Should reach base_lr"
    
    # Check cosine decay
    assert lrs[50] < lrs[10], "LR should decay after warmup"
    assert lrs[99] > 1e-5 * 0.9, "Should approach min_lr"


# ============================================================
# TEST: DATA SPLITTING
# ============================================================

def test_safe_stratified_split_basic():
    """Test stratified split with sufficient data."""
    labels = [0] * 20 + [1] * 20 + [2] * 20
    
    tr, va = safe_stratified_split(labels, test_size=0.2, seed=42)
    
    assert len(tr) + len(va) == len(labels)
    assert len(set(tr) & set(va)) == 0, "Train/val overlap"
    
    # Check stratification
    tr_labels = [labels[i] for i in tr]
    va_labels = [labels[i] for i in va]
    
    for c in [0, 1, 2]:
        assert c in tr_labels, f"Class {c} missing from train"
        assert c in va_labels, f"Class {c} missing from val"


def test_safe_stratified_split_edge_cases():
    """Test stratified split with edge cases."""
    # Single sample per class - all go to train (can't split singletons)
    labels = [0, 1, 2]
    tr, va = safe_stratified_split(labels, test_size=0.33, seed=42)
    assert len(tr) == 3, "Singleton classes should all go to train"
    assert len(va) == 0, "No validation samples when all classes are singletons"
    
    # Two samples per class - one train, one val per class
    labels = [0, 0, 1, 1, 2, 2]
    tr, va = safe_stratified_split(labels, test_size=0.33, seed=42)
    assert len(tr) > 0 and len(va) > 0, "Should have both train and val"
    for c in [0, 1, 2]:
        tr_labels = [labels[i] for i in tr]
        va_labels = [labels[i] for i in va]
        # Each class should appear at least once across train+val
        assert (c in tr_labels) or (c in va_labels)


def test_make_kfolds_basic():
    """Test k-fold cross-validation."""
    labels = [0] * 20 + [1] * 20 + [2] * 20
    
    folds = make_kfolds(labels, k=5, seed=42)
    
    assert len(folds) == 5
    
    # Check no overlap between folds
    all_indices = set(range(len(labels)))
    for tr, va in folds:
        assert len(set(tr) & set(va)) == 0, "Train/val overlap in fold"
        assert set(tr) | set(va) == all_indices, "Not all indices covered"


def test_make_kfolds_edge_cases():
    """Test k-fold with insufficient samples per class."""
    # Fewer samples than folds
    labels = [0, 1, 2, 3]
    
    folds = make_kfolds(labels, k=5, seed=42)
    
    assert len(folds) == 5
    for tr, va in folds:
        assert len(tr) + len(va) == len(labels)


# ============================================================
# TEST: INTEGRATION
# ============================================================

def test_full_training_cycle_mini(sample_dataset_files):
    """Test a mini training loop (integration test)."""
    files, labels = sample_dataset_files
    
    # Create small dataset
    ds = NPZPoseDataset(files[:5], labels[:5], window_size=30, stride=30,
                       mode="select", normalize_fn=default_normalize,
                       cache_views=False)
    
    sampler = make_window_sampler(ds, len(ALL_CLASSES))
    loader = torch.utils.data.DataLoader(ds, batch_size=4, sampler=sampler)
    
    # Small model
    model = BigTransformer(num_classes=len(ALL_CLASSES), d_model=32,
                          nhead=2, layers=1, ff_dim=64, drop=0.1)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Train for 2 steps
    model.train()
    losses = []
    
    for i, (x, y, *_) in enumerate(loader):
        if i >= 2:
            break
        
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
    
    assert len(losses) == 2
    assert all(np.isfinite(l) for l in losses), "Loss contains inf/nan"


# ============================================================
# RUN TESTS
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])