"""Comprehensive tests for dataset loading, processing, and data quality validation.

Tests are designed to work with real NPZ pose data files in the npz_output directory.
"""

import pytest
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))  # Add parent dir to path

from gpu_stress import (
    NPZPoseDataset, load_multiview_sequence, 
    window_class_counts, make_window_sampler,
    NUM_LANDMARKS, ALL_CLASSES
)


@pytest.fixture
def real_data_dir():
    """Path to real dataset directory."""
    return Path("npz_output")


@pytest.fixture
def real_dataset_files(real_data_dir):
    """Load real dataset files if available."""
    if not real_data_dir.exists():
        pytest.skip(f"Real data directory not found: {real_data_dir}")
    
    files = list(real_data_dir.glob("*.npz"))
    if not files:
        pytest.skip(f"No .npz files found in {real_data_dir}")
    
    labels = []
    for f in files:
        # Handle multi-word class names
        parts = f.name.split('_')
        label_str = parts[0].lower()
        
        # Map special cases
        if label_str == "fencer":
            label_str = "fencer posture"
        elif label_str == "versive":
            label_str = "versive head"
        
        if label_str in ALL_CLASSES:
            labels.append(ALL_CLASSES.index(label_str))
        else:
            labels.append(-1)  # Unknown class
    
    return files, labels


class TestDatasetLoading:
    """Test dataset loading functionality with real data."""
    
    def test_load_real_files(self, real_dataset_files):
        """Test loading real dataset files."""
        files, labels = real_dataset_files
        assert len(files) > 0, "No files found to test"
        
        # Test loading a few files
        for f in files[:5]:
            views = load_multiview_sequence(f)
            assert len(views) > 0, f"No views loaded from {f.name}"
            assert views[0].shape[1] == NUM_LANDMARKS, f"Wrong number of landmarks in {f.name}"
            assert views[0].shape[2] in [3, 4], f"Wrong coordinate dimensions in {f.name}"


class TestNPZPoseDataset:
    """Test NPZPoseDataset class with real data."""
    
    def test_dataset_length(self, real_dataset_files):
        """Test dataset returns correct length with real data."""
        files, labels = real_dataset_files
        ds = NPZPoseDataset(files[:10], labels[:10], window_size=30, stride=15, 
                           mode="select", cache_views=False)
        assert len(ds) > 0
    
    @pytest.mark.parametrize("mode", ["expand", "select", "stack"])
    def test_dataset_modes(self, real_dataset_files, mode):
        """Test different dataset modes with real data."""
        files, labels = real_dataset_files
        ds = NPZPoseDataset(files[:5], labels[:5], window_size=30, stride=30,
                           mode=mode, cache_views=False)
        x, y, *_ = ds[0]
        assert x.dim() in [3, 4], f"Unexpected tensor dimensions for mode {mode}"
        assert y.item() >= 0, "Invalid label"
        assert y.item() < len(ALL_CLASSES), f"Label {y.item()} out of range"


class TestSampling:
    """Test sampling utilities with real data."""
    
    def test_window_counts_sum(self, real_dataset_files):
        """Test that window counts sum correctly."""
        files, labels = real_dataset_files
        ds = NPZPoseDataset(files[:10], labels[:10], window_size=30, stride=30,
                           mode="select", cache_views=False)
        counts = window_class_counts(ds, len(ALL_CLASSES))
        assert counts.sum() == len(ds)
    
    def test_sampler_balances_classes(self, real_dataset_files):
        """Test that sampler creates balanced batches."""
        files, labels = real_dataset_files
        ds = NPZPoseDataset(files[:20], labels[:20], window_size=30, stride=30,
                           mode="select", cache_views=False)
        sampler = make_window_sampler(ds, len(ALL_CLASSES))
        
        # Sample many indices
        sampled_labels = []
        for idx in list(sampler):
            label = ds.labels[ds._index[idx].file_idx]
            sampled_labels.append(label)
            if len(sampled_labels) >= 1000:
                break
        
        counts = Counter(sampled_labels)
        # Classes should be relatively balanced
        if len(counts) > 1:
            max_count = max(counts.values())
            min_count = min(counts.values())
            ratio = max_count / min_count
            assert ratio < 3.0, f"Classes not balanced: {counts}"


class TestDataQuality:
    """Test data quality and integrity of actual dataset files."""
    
    def test_all_files_loadable(self, real_dataset_files):
        """Test that all files can be loaded without errors."""
        files, labels = real_dataset_files
        errors = []
        
        for f in files:
            try:
                with np.load(f, allow_pickle=False) as z:
                    _ = list(z.keys())
            except Exception as e:
                errors.append(f"ERROR loading {f.name}: {e}")
        
        assert len(errors) == 0, f"Failed to load {len(errors)} files:\n" + "\n".join(errors)
    
    def test_no_nan_values(self, real_dataset_files):
        """Test that no files contain NaN or infinite values."""
        files, labels = real_dataset_files
        problematic_files = []
        
        for f in files:
            try:
                with np.load(f, allow_pickle=False) as z:
                    for key in z.keys():
                        if key.startswith('view_') or key == 'sequences':
                            arr = z[key]
                            if not np.all(np.isfinite(arr)):
                                problematic_files.append(f.name)
                                break
            except Exception:
                continue
        
        assert len(problematic_files) == 0, \
            f"Files contain NaN/inf values: {problematic_files}"
    
    def test_variance_threshold(self, real_dataset_files):
        """Test that files have reasonable variance (not all zeros/constants)."""
        files, labels = real_dataset_files
        low_variance_files = []
        variance_threshold = 0.001
        
        for f in files:
            try:
                with np.load(f, allow_pickle=False) as z:
                    for key in z.keys():
                        if key.startswith('view_') or key == 'sequences':
                            arr = z[key]
                            xyz = arr[:, :, :3] if arr.ndim >= 3 else arr
                            variance = float(np.var(xyz))
                            if variance < variance_threshold:
                                low_variance_files.append((f.name, variance))
                                break
            except Exception:
                continue
        
        assert len(low_variance_files) == 0, \
            f"Files with suspiciously low variance: {low_variance_files}"
    
    def test_zero_frame_percentage(self, real_dataset_files):
        """Test that files don't have excessive zero frames."""
        files, labels = real_dataset_files
        problematic_files = []
        max_zero_percentage = 0.5  # 50%
        
        for f in files:
            try:
                with np.load(f, allow_pickle=False) as z:
                    for key in z.keys():
                        if key.startswith('view_') or key == 'sequences':
                            arr = z[key]
                            if arr.ndim >= 3:
                                T = arr.shape[0]
                                xyz = arr[:, :, :3]
                                zero_frames = np.all(xyz == 0, axis=(1, 2)).sum()
                                if zero_frames > T * max_zero_percentage:
                                    problematic_files.append((f.name, zero_frames, T))
                                    break
            except Exception:
                continue
        
        assert len(problematic_files) == 0, \
            f"Files with >50% zero frames: {problematic_files}"
    
    def test_class_distribution(self, real_dataset_files):
        """Test that dataset has reasonable class distribution."""
        files, labels = real_dataset_files
        label_counts = Counter(labels)
        
        # Remove unknown classes (-1) from count
        label_counts.pop(-1, None)
        
        assert len(label_counts) > 0, "No valid class labels found"
        
        # Check that no class is completely missing (allow some missing classes)
        present_classes = [ALL_CLASSES[idx] for idx in label_counts.keys()]
        assert len(present_classes) >= 3, \
            f"Too few classes present: {present_classes}"
        
        # Check that no class dominates too much (>80% of data)
        total = sum(label_counts.values())
        max_count = max(label_counts.values())
        assert max_count / total < 0.8, \
            f"One class dominates dataset: {label_counts}"
    
    def test_class_imbalance_ratio(self, real_dataset_files):
        """Test that class imbalance is not extreme."""
        files, labels = real_dataset_files
        label_counts = Counter(labels)
        label_counts.pop(-1, None)
        
        if len(label_counts) > 1:
            max_count = max(label_counts.values())
            min_count = min(label_counts.values())
            imbalance_ratio = max_count / max(min_count, 1)
            
            # Warn if imbalance is high, but don't fail
            if imbalance_ratio > 5:
                pytest.skip(f"High class imbalance ({imbalance_ratio:.1f}:1) - consider class weights")
    
    def test_visibility_values(self, real_dataset_files):
        """Test that visibility values are in valid range [0, 1]."""
        files, labels = real_dataset_files
        invalid_files = []
        
        for f in files[:20]:  # Sample first 20 files
            try:
                with np.load(f, allow_pickle=False) as z:
                    for key in z.keys():
                        if key.startswith('view_') or key == 'sequences':
                            arr = z[key]
                            if arr.shape[-1] > 3:  # Has visibility channel
                                vis = arr[:, :, 3]
                                if np.any((vis < 0) | (vis > 1)):
                                    invalid_files.append(f.name)
                                    break
            except Exception:
                continue
        
        assert len(invalid_files) == 0, \
            f"Files with invalid visibility values: {invalid_files}"
    
    @pytest.mark.parametrize("expected_shape_dims", [3, 4])
    def test_array_shapes_consistent(self, real_dataset_files, expected_shape_dims):
        """Test that arrays have consistent dimensionality."""
        files, labels = real_dataset_files
        
        for f in files[:20]:  # Sample first 20 files
            try:
                with np.load(f, allow_pickle=False) as z:
                    for key in z.keys():
                        if key.startswith('view_') or key == 'sequences':
                            arr = z[key]
                            assert arr.ndim == 3, \
                                f"Unexpected ndim in {f.name}[{key}]: {arr.ndim}"
                            assert arr.shape[-1] in [3, 4], \
                                f"Unexpected last dim in {f.name}[{key}]: {arr.shape[-1]}"
            except AssertionError:
                raise
            except Exception:
                continue


class TestDataStatistics:
    """Generate and validate statistical summaries of the dataset."""
    
    def test_print_dataset_statistics(self, real_dataset_files, capsys):
        """Print comprehensive dataset statistics."""
        files, labels = real_dataset_files
        stats = defaultdict(list)
        
        for f in files:
            # Parse class name with multi-word support
            parts = f.name.split('_')
            label = parts[0].lower()
            if label == "fencer":
                label = "fencer posture"
            elif label == "versive":
                label = "versive head"
            
            try:
                with np.load(f, allow_pickle=False) as z:
                    for key in z.keys():
                        if key.startswith('view_') or key == 'sequences':
                            arr = z[key]
                            if arr.ndim >= 3:
                                T = arr.shape[0]
                                xyz = arr[:, :, :3]
                                vis = arr[:, :, 3] if arr.shape[-1] > 3 else np.ones((T, arr.shape[1]))
                                
                                stats[label].append({
                                    'file': f.name,
                                    'frames': T,
                                    'has_zeros': np.all(xyz == 0, axis=(1, 2)).sum(),
                                    'has_nan': np.any(~np.isfinite(xyz)),
                                    'variance': float(np.var(xyz)),
                                    'visibility': float(np.mean(vis)),
                                    'motion_mag': float(np.mean(np.abs(np.diff(xyz, axis=0)))) if T > 1 else 0.0,
                                    'x_range': float(np.ptp(xyz[:, :, 0])),
                                    'y_range': float(np.ptp(xyz[:, :, 1])),
                                    'z_range': float(np.ptp(xyz[:, :, 2]))
                                })
                            break
            except Exception as e:
                print(f"ERROR loading {f.name}: {e}")
        
        # Print statistics
        print("\n" + "="*80)
        print("DATASET STATISTICS")
        print("="*80)
        
        total_files = sum(len(v) for v in stats.values())
        total_frames = sum(d['frames'] for files_data in stats.values() for d in files_data)
        
        print(f"\nTotal files: {total_files}")
        print(f"Total frames: {total_frames}")
        
        for label, files_data in sorted(stats.items()):
            frames = [d['frames'] for d in files_data]
            zeros = [d['has_zeros'] for d in files_data]
            variances = [d['variance'] for d in files_data]
            visibility = [d['visibility'] for d in files_data]
            motion = [d['motion_mag'] for d in files_data]
            
            print(f"\n{label.upper()}:")
            print(f"  Files: {len(files_data)} ({100*len(files_data)/total_files:.1f}%)")
            print(f"  Frames: mean={np.mean(frames):.1f}, std={np.std(frames):.1f}, "
                  f"min={np.min(frames)}, max={np.max(frames)}")
            print(f"  Zero frames: mean={np.mean(zeros):.1f}, max={np.max(zeros)}")
            print(f"  Variance: mean={np.mean(variances):.3f}, std={np.std(variances):.3f}")
            print(f"  Visibility: mean={np.mean(visibility):.3f}")
            print(f"  Motion magnitude: mean={np.mean(motion):.6f} ± {np.std(motion):.6f}")
            
            # Flag suspicious files
            warnings = []
            for d in files_data:
                if d['has_nan']:
                    warnings.append(f"    ⚠️  {d['file']}: contains NaN!")
                if d['variance'] < 0.001:
                    warnings.append(f"    ⚠️  {d['file']}: suspiciously low variance ({d['variance']:.6f})")
                if d['has_zeros'] > d['frames'] * 0.5:
                    warnings.append(f"    ⚠️  {d['file']}: >50% zero frames ({d['has_zeros']}/{d['frames']})")
                if d['visibility'] < 0.5:
                    warnings.append(f"    ⚠️  {d['file']}: low visibility ({d['visibility']:.3f})")
            
            if warnings:
                print("\n".join(warnings))
        
        # Class separability analysis
        print(f"\n🎯 CLASS SEPARABILITY (motion-based):")
        print("-" * 80)
        motion_by_class = {label: [d['motion_mag'] for d in files_data] 
                          for label, files_data in stats.items()}
        
        all_motion = []
        for motions in motion_by_class.values():
            all_motion.extend(motions)
        
        overall_mean = np.mean(all_motion)
        between_var = sum(len(motions) * (np.mean(motions) - overall_mean) ** 2 
                         for motions in motion_by_class.values())
        within_var = sum(np.sum((np.array(motions) - np.mean(motions)) ** 2) 
                        for motions in motion_by_class.values())
        
        if within_var > 0:
            f_ratio = between_var / within_var
            print(f"  Between-class variance: {between_var:.6f}")
            print(f"  Within-class variance: {within_var:.6f}")
            print(f"  F-ratio: {f_ratio:.4f}")
            
            if f_ratio < 0.1:
                print(f"  ⚠️  VERY LOW separability - classes may be too similar!")
            elif f_ratio < 0.5:
                print(f"  ⚠️  LOW separability - challenging classification task")
            else:
                print(f"  ✅ Reasonable separability")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        print("-" * 80)
        
        avg_x_range = np.mean([d['x_range'] for files_data in stats.values() for d in files_data])
        if avg_x_range > 2.0:
            print("  • Data appears un-normalized (range > 2). Consider normalization.")
        
        avg_vis = np.mean([d['visibility'] for files_data in stats.values() for d in files_data])
        if avg_vis < 0.8:
            print(f"  • Low average visibility ({avg_vis:.3f}). Consider filtering low-quality frames.")
        
        if within_var > 0 and f_ratio < 0.5:
            print("  • Low class separability. Consider:")
            print("    - Using longer sequence windows")
            print("    - Adding data augmentation (temporal jitter, rotations)")
            print("    - Using attention mechanisms or temporal modeling")
        
        class_counts = Counter(labels)
        class_counts.pop(-1, None)
        if len(class_counts) > 1:
            max_count = max(class_counts.values())
            min_count = min(class_counts.values())
            if max_count / max(min_count, 1) > 2:
                print(f"  • Class imbalance detected ({max_count}:{min_count} ratio)")
                print("    - Consider class weights or oversampling")
        
        print("\n" + "="*80)
        
        # Always pass - this test is for information only
        assert True
    
    def test_frame_length_distribution(self, real_dataset_files):
        """Test that frame lengths fall within reasonable ranges."""
        files, labels = real_dataset_files
        frame_lengths = []
        
        for f in files:
            try:
                with np.load(f, allow_pickle=False) as z:
                    for key in z.keys():
                        if key.startswith('view_') or key == 'sequences':
                            arr = z[key]
                            T = arr.shape[0]
                            frame_lengths.append(T)
                            break
            except Exception:
                continue
        
        assert len(frame_lengths) > 0, "No valid sequences found"
        
        mean_length = np.mean(frame_lengths)
        std_length = np.std(frame_lengths)
        
        # Check for reasonable distribution
        assert mean_length > 20, f"Average sequence too short: {mean_length}"
        # Allow longer sequences for video data (up to 5000 frames ~= 2-3 minutes at 30fps)
        assert mean_length < 5000, f"Average sequence suspiciously long: {mean_length}"
        assert std_length < mean_length * 2, \
            f"Very high variance in sequence lengths: mean={mean_length}, std={std_length}"
    
    def test_motion_magnitude_reasonable(self, real_dataset_files):
        """Test that motion magnitudes are in reasonable ranges."""
        files, labels = real_dataset_files
        motion_mags = []
        
        for f in files[:50]:  # Sample first 50 files
            try:
                with np.load(f, allow_pickle=False) as z:
                    for key in z.keys():
                        if key.startswith('view_') or key == 'sequences':
                            arr = z[key]
                            xyz = arr[:, :, :3]
                            if xyz.shape[0] > 1:
                                motion = np.mean(np.abs(np.diff(xyz, axis=0)))
                                motion_mags.append(motion)
                            break
            except Exception:
                continue
        
        if motion_mags:
            mean_motion = np.mean(motion_mags)
            # Motion should be detectable but not extreme
            assert mean_motion > 1e-6, f"Almost no motion detected: {mean_motion}"
            assert mean_motion < 10.0, f"Suspiciously high motion: {mean_motion}"
    
    def test_coordinate_ranges(self, real_dataset_files):
        """Test that coordinate ranges are reasonable."""
        files, labels = real_dataset_files
        x_ranges, y_ranges, z_ranges = [], [], []
        
        for f in files[:30]:  # Sample first 30 files
            try:
                with np.load(f, allow_pickle=False) as z:
                    for key in z.keys():
                        if key.startswith('view_') or key == 'sequences':
                            arr = z[key]
                            xyz = arr[:, :, :3]
                            x_ranges.append(np.ptp(xyz[:, :, 0]))
                            y_ranges.append(np.ptp(xyz[:, :, 1]))
                            z_ranges.append(np.ptp(xyz[:, :, 2]))
                            break
            except Exception:
                continue
        
        # All coordinates should have some variation
        assert np.mean(x_ranges) > 0.01, "X coordinates have almost no variation"
        assert np.mean(y_ranges) > 0.01, "Y coordinates have almost no variation"
        assert np.mean(z_ranges) > 0.01, "Z coordinates have almost no variation"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])