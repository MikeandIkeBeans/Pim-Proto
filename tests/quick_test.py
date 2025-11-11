#!/usr/bin/env python3
"""
Quick test runner - No pytest required!
Run this to quickly validate your training code.

Usage: python quick_test.py
"""

import sys
import traceback
import numpy as np
import torch
from pathlib import Path
import tempfile
import shutil

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

# Import from new package structure
try:
    # Import utilities from the new package
    from pim_detection.utils import (
        set_seed, 
        load_multiview_sequence,
        NUM_LANDMARKS, 
        L_SHOULDER, 
        R_SHOULDER, 
        L_HIP, 
        R_HIP, 
        ALL_CLASSES
    )
    print(f"{GREEN}✓{RESET} Successfully imported from pim_detection.utils")
except ImportError as e:
    print(f"{RED}✗{RESET} Failed to import from pim_detection.utils: {e}")
    sys.exit(1)

# Import from gpu_stress (still in root - will be migrated later)
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from gpu_stress import (
        default_normalize, spatial_augment, _pair_dist,
        NPZPoseDataset, BigTransformer,
        BalancedSoftmaxCE, EMA, WarmupCosineScheduler
    )
    print(f"{GREEN}✓{RESET} Successfully imported from gpu_stress")
except ImportError as e:
    print(f"{RED}✗{RESET} Failed to import from gpu_stress: {e}")
    print(f"{BLUE}ℹ{RESET} Note: Some components not yet migrated to package structure")
    sys.exit(1)


class TestRunner:
    """Simple test runner without pytest."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []
    
    def test(self, name):
        """Decorator for test functions."""
        def decorator(func):
            self.tests.append((name, func))
            return func
        return decorator
    
    def run(self):
        """Run all registered tests."""
        print(f"\n{BOLD}Running {len(self.tests)} tests...{RESET}\n")
        
        for name, func in self.tests:
            try:
                func()
                print(f"{GREEN}✓{RESET} {name}")
                self.passed += 1
            except AssertionError as e:
                print(f"{RED}✗{RESET} {name}")
                print(f"  {RED}AssertionError: {e}{RESET}")
                self.failed += 1
            except Exception as e:
                print(f"{RED}✗{RESET} {name}")
                print(f"  {RED}Exception: {e}{RESET}")
                traceback.print_exc()
                self.failed += 1
        
        print(f"\n{BOLD}Results:{RESET}")
        print(f"  {GREEN}Passed: {self.passed}{RESET}")
        print(f"  {RED}Failed: {self.failed}{RESET}")
        
        return self.failed == 0


# Initialize test runner
runner = TestRunner()


# ============================================================
# TESTS
# ============================================================

@runner.test("Seed reproducibility")
def test_seed_reproducibility():
    set_seed(42)
    x1 = torch.randn(10)
    set_seed(42)
    x2 = torch.randn(10)
    assert torch.allclose(x1, x2), "PyTorch random seed not working"


@runner.test("Pair distance calculation")
def test_pair_dist():
    frame = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
    frame[L_SHOULDER] = [1.0, 0.0, 0.0]
    frame[R_SHOULDER] = [0.0, 0.0, 0.0]
    dist = _pair_dist(frame, L_SHOULDER, R_SHOULDER)
    assert dist is not None, "Distance is None"
    assert abs(dist - 1.0) < 1e-5, f"Expected ~1.0, got {dist}"


@runner.test("Normalization preserves shape")
def test_normalize_shape():
    seq = np.random.randn(50, NUM_LANDMARKS, 3).astype(np.float32)
    seq[:, [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP], :] += 1.0
    norm = default_normalize(seq)
    assert norm.shape == seq.shape, f"Shape mismatch: {norm.shape} vs {seq.shape}"
    assert norm.dtype == np.float32, f"Wrong dtype: {norm.dtype}"


@runner.test("Normalization produces finite values")
def test_normalize_finite():
    seq = np.random.randn(50, NUM_LANDMARKS, 3).astype(np.float32)
    seq[:, [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP], :] += 1.0
    norm = default_normalize(seq)
    assert np.all(np.isfinite(norm)), "Contains inf/nan"
    assert norm.min() >= -5.0 and norm.max() <= 5.0, "Values out of expected range"


@runner.test("Normalization handles edge cases")
def test_normalize_edge_cases():
    # All zeros
    seq = np.zeros((30, NUM_LANDMARKS, 3), dtype=np.float32)
    norm = default_normalize(seq)
    assert np.all(np.isfinite(norm)), "Failed on all zeros"
    
    # Very small values
    seq = np.random.randn(30, NUM_LANDMARKS, 3).astype(np.float32) * 1e-6
    seq[:, [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP], :] += 1e-5
    norm = default_normalize(seq)
    assert np.all(np.isfinite(norm)), "Failed on tiny values"


@runner.test("Augmentation preserves shape")
def test_augment_shape():
    seq = np.random.randn(30, NUM_LANDMARKS, 3).astype(np.float32)
    aug = spatial_augment(seq)
    assert aug.shape == seq.shape, "Augmentation changed shape"
    assert np.all(np.isfinite(aug)), "Augmentation produced inf/nan"


@runner.test("Augmentation is deterministic with seed")
def test_augment_deterministic():
    seq = np.random.randn(30, NUM_LANDMARKS, 3).astype(np.float32)
    
    np.random.seed(42)
    aug1 = spatial_augment(seq)
    
    np.random.seed(42)
    aug2 = spatial_augment(seq)
    
    assert np.allclose(aug1, aug2), "Augmentation not reproducible"


@runner.test("File loading - create and load single view")
def test_file_loading():
    tmpdir = tempfile.mkdtemp()
    try:
        seq = np.random.randn(80, NUM_LANDMARKS, 3).astype(np.float32)
        path = Path(tmpdir) / "test.npz"
        np.savez(str(path), view_0=seq)
        
        views = load_multiview_sequence(path)
        assert len(views) == 1, f"Expected 1 view, got {len(views)}"
        assert views[0].shape == seq.shape, "Shape mismatch after loading"
    finally:
        shutil.rmtree(tmpdir)


@runner.test("File loading - multiple views")
def test_file_loading_multiview():
    tmpdir = tempfile.mkdtemp()
    try:
        view1 = np.random.randn(80, NUM_LANDMARKS, 3).astype(np.float32)
        view2 = np.random.randn(80, NUM_LANDMARKS, 3).astype(np.float32)
        path = Path(tmpdir) / "test_mv.npz"
        np.savez(str(path), view_0=view1, view_1=view2)
        
        views = load_multiview_sequence(path)
        assert len(views) == 2, f"Expected 2 views, got {len(views)}"
        assert all(v.shape[1] == NUM_LANDMARKS for v in views), "Wrong landmark count"
    finally:
        shutil.rmtree(tmpdir)


@runner.test("Dataset basic functionality")
def test_dataset_basic():
    tmpdir = tempfile.mkdtemp()
    try:
        # Create test files
        files, labels = [], []
        for i in range(3):
            seq = np.random.randn(100, NUM_LANDMARKS, 3).astype(np.float32)
            seq[:, [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP], :] += 1.0
            path = Path(tmpdir) / f"normal_{i:03d}.npz"
            np.savez(str(path), view_0=seq)
            files.append(str(path))
            labels.append(0)
        
        ds = NPZPoseDataset(files, labels, window_size=30, stride=15,
                           mode="select", normalize_fn=default_normalize,
                           augment=None, cache_views=False)
        
        assert len(ds) > 0, "Dataset is empty"
        
        x, y, fi, vi, st = ds[0]
        assert isinstance(x, torch.Tensor), "Output is not a tensor"
        assert x.shape[0] == 30, f"Expected window size 30, got {x.shape[0]}"
        assert x.shape[1] == NUM_LANDMARKS, "Wrong number of landmarks"
        assert isinstance(y, torch.Tensor), "Label is not a tensor"
    finally:
        shutil.rmtree(tmpdir)


@runner.test("Dataset window padding")
def test_dataset_padding():
    tmpdir = tempfile.mkdtemp()
    try:
        # Create short sequence
        seq = np.random.randn(40, NUM_LANDMARKS, 3).astype(np.float32)
        seq[:, [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP], :] += 1.0
        path = Path(tmpdir) / "short.npz"
        np.savez(str(path), view_0=seq)
        
        # Request longer window
        ds = NPZPoseDataset([str(path)], [0], window_size=60, stride=60,
                           mode="select", cache_views=False)
        
        if len(ds) > 0:
            x, *_ = ds[0]
            assert x.shape[0] == 60, f"Padding failed: got {x.shape[0]}"
    finally:
        shutil.rmtree(tmpdir)


@runner.test("Model forward pass - mean pooling")
def test_model_forward():
    model = BigTransformer(num_classes=len(ALL_CLASSES),
                          d_model=64, nhead=2, layers=1, ff_dim=128,
                          drop=0.1, pool="mean")
    model.eval()
    
    x = torch.randn(2, 30, NUM_LANDMARKS, 3)
    
    with torch.no_grad():
        logits = model(x)
    
    assert logits.shape == (2, len(ALL_CLASSES)), f"Wrong output shape: {logits.shape}"
    assert torch.all(torch.isfinite(logits)), "Model output contains inf/nan"


@runner.test("Model forward pass - CLS pooling")
def test_model_cls_pooling():
    model = BigTransformer(num_classes=len(ALL_CLASSES),
                          d_model=64, nhead=2, layers=1, ff_dim=128,
                          pool="cls")
    model.eval()
    
    x = torch.randn(2, 30, NUM_LANDMARKS, 3)
    
    with torch.no_grad():
        logits = model(x)
    
    assert logits.shape == (2, len(ALL_CLASSES)), "CLS pooling failed"


@runner.test("Model forward pass - attention pooling")
def test_model_attn_pooling():
    model = BigTransformer(num_classes=len(ALL_CLASSES),
                          d_model=64, nhead=2, layers=1, ff_dim=128,
                          pool="attn")
    model.eval()
    
    x = torch.randn(2, 30, NUM_LANDMARKS, 3)
    
    with torch.no_grad():
        logits = model(x)
    
    assert logits.shape == (2, len(ALL_CLASSES)), "Attention pooling failed"


@runner.test("Model gradient flow")
def test_model_gradients():
    model = BigTransformer(num_classes=len(ALL_CLASSES),
                          d_model=32, nhead=2, layers=1, ff_dim=64)
    model.train()
    
    x = torch.randn(2, 30, NUM_LANDMARKS, 3, requires_grad=True)
    y = torch.tensor([0, 1])
    
    logits = model(x)
    loss = torch.nn.CrossEntropyLoss()(logits, y)
    loss.backward()
    
    # Check gradients exist
    grad_count = 0
    for param in model.parameters():
        if param.grad is not None:
            assert torch.all(torch.isfinite(param.grad)), "Gradient contains inf/nan"
            grad_count += 1
    
    assert grad_count > 0, "No gradients computed"


@runner.test("BalancedSoftmaxCE loss")
def test_balanced_loss():
    cls_counts = torch.tensor([100.0, 20.0, 5.0])
    loss_fn = BalancedSoftmaxCE(cls_counts, label_smoothing=0.1)
    
    logits = torch.randn(8, 3)
    targets = torch.randint(0, 3, (8,))
    
    loss = loss_fn(logits, targets)
    
    assert torch.isfinite(loss), "Loss is inf/nan"
    assert loss.item() >= 0, "Loss is negative"


@runner.test("EMA initialization")
def test_ema_init():
    model = BigTransformer(num_classes=len(ALL_CLASSES),
                          d_model=32, nhead=2, layers=1, ff_dim=64)
    ema = EMA(model, decay=0.999)
    
    # Check shadow exists
    assert len(ema.shadow) > 0, "EMA shadow is empty"
    
    # Shadow should match initial model
    for name, param in model.state_dict().items():
        if param.dtype.is_floating_point:
            assert name in ema.shadow, f"Missing shadow for {name}"


@runner.test("EMA update")
def test_ema_update():
    model = BigTransformer(num_classes=len(ALL_CLASSES),
                          d_model=32, nhead=2, layers=1, ff_dim=64)
    ema = EMA(model, decay=0.9)
    
    # Get initial shadow value
    first_key = list(ema.shadow.keys())[0]
    initial_shadow = ema.shadow[first_key].clone()
    
    # Modify model
    for param in model.parameters():
        param.data += 0.1
    
    # Update EMA
    ema.update(model)
    
    # Shadow should have changed
    updated_shadow = ema.shadow[first_key]
    assert not torch.allclose(initial_shadow, updated_shadow), "EMA did not update"


@runner.test("EMA context manager")
def test_ema_context():
    model = BigTransformer(num_classes=len(ALL_CLASSES),
                          d_model=32, nhead=2, layers=1, ff_dim=64)
    ema = EMA(model, decay=0.999)
    
    # Get original param
    first_param_name = list(model.state_dict().keys())[0]
    original = model.state_dict()[first_param_name].clone()
    
    # Modify model
    for param in model.parameters():
        param.data += 1.0
    
    modified = model.state_dict()[first_param_name].clone()
    ema.update(model)
    
    # Use context manager
    with ema.average_parameters(model):
        inside_context = model.state_dict()[first_param_name].clone()
        # Should be different from modified
        assert not torch.allclose(inside_context, modified), "EMA not applied in context"
    
    # Should restore after context
    after_context = model.state_dict()[first_param_name]
    assert torch.allclose(after_context, modified), "Parameters not restored after context"


@runner.test("Warmup cosine scheduler")
def test_scheduler():
    model = torch.nn.Linear(10, 10)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    sched = WarmupCosineScheduler(opt, warmup_steps=10, total_steps=100,
                                  base_lr=1e-3, min_lr=1e-5)
    
    lrs = []
    for _ in range(100):
        lrs.append(opt.param_groups[0]['lr'])
        sched.step()
    
    # Check warmup increases
    assert lrs[0] < lrs[9], "LR should increase during warmup"
    
    # Check reaches base_lr
    assert abs(lrs[10] - 1e-3) < 1e-6, f"Should reach base_lr, got {lrs[10]}"
    
    # Check decay after warmup
    assert lrs[50] < lrs[10], "LR should decay after warmup"
    assert lrs[99] >= 1e-5 * 0.9, f"Should approach min_lr, got {lrs[99]}"


@runner.test("Mini training loop integration")
def test_mini_training():
    tmpdir = tempfile.mkdtemp()
    try:
        # Create tiny dataset
        files, labels = [], []
        for cls_idx in [0, 1]:
            for i in range(2):
                seq = np.random.randn(80, NUM_LANDMARKS, 3).astype(np.float32)
                seq[:, [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP], :] += 1.0
                path = Path(tmpdir) / f"class{cls_idx}_{i:03d}.npz"
                np.savez(str(path), view_0=seq)
                files.append(str(path))
                labels.append(cls_idx)
        
        # Create dataset
        ds = NPZPoseDataset(files, labels, window_size=30, stride=30,
                           mode="select", normalize_fn=default_normalize,
                           cache_views=False)
        
        loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True)
        
        # Small model
        model = BigTransformer(num_classes=len(ALL_CLASSES),
                              d_model=32, nhead=2, layers=1, ff_dim=64)
        
        criterion = torch.nn.CrossEntropyLoss()
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
        
        assert len(losses) == 2, "Training loop failed"
        assert all(np.isfinite(l) for l in losses), "Training produced inf/nan loss"
        
    finally:
        shutil.rmtree(tmpdir)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print(f"\n{BOLD}{BLUE}╔═══════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}{BLUE}║  Quick Test Suite for Big Transformer   ║{RESET}")
    print(f"{BOLD}{BLUE}╚═══════════════════════════════════════════╝{RESET}")
    
    success = runner.run()
    
    if success:
        print(f"\n{BOLD}{GREEN}All tests passed! 🎉{RESET}\n")
        sys.exit(0)
    else:
        print(f"\n{BOLD}{RED}Some tests failed. Please review the errors above.{RESET}\n")
        sys.exit(1)