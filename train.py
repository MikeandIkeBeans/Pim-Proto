#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PoseTCN Trainer — FAST + FEATURE-RICH (Hybrid Fusion, Class-Balancing, Temporal Tools)

Goal:
- Keep the **training speed** characteristics of your "more efficient" script
  (fast dataloading, multi-worker, pinned memory, persistent workers, eager load).
- Bring back the **tuning knobs and data balancing** you liked:
  * Hybrid fusion modes: early / late / hybrid (weighted)
  * Class-balancing: train-subset sampler, smoothed+clipped class weights, head-bias init
  * Temporal artifact tools: purity analysis, soft labels (frame-level), filtering
  * Post-hoc temperature scaling
  * Rich augmentations with **normal-class-aware** scaling
  * BF16-safe evaluation
- Fix correctness pitfalls: file-index vs meta-index alignment (uses meta indices everywhere)

Notes:
- Default is **eager load** (fast) with multi-workers + pinned memory for speed.
- You can toggle `--lazy_load` to reduce RAM (slower; especially on .npz). Eager is recommended for speed.
- Works on Windows and Linux; start method set to spawn/forkserver accordingly.
"""

import os
import glob
import random
import time
import json
import platform
import re
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report

# --------- Constants ---------
NUM_LANDMARKS = 33
INPUT_LANDMARK_DIM = 33 * 3
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 11, 12, 23, 24

# --------- Reproducibility & CUDA ---------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
if torch.cuda.is_available():
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass

# --------- Normalization ---------
def _pair_dist(fr: np.ndarray, a: int, b: int) -> Optional[float]:
    va, vb = fr[a], fr[b]
    if not (np.any(va != 0) and np.any(vb != 0)):
        return None
    return float(np.linalg.norm(va - vb) + 1e-8)

def normalize_single_view(seq: np.ndarray) -> np.ndarray:
    """Center by hips (fallback shoulders), scale by median body-size cue."""
    T = seq.shape[0]
    hips_ok = (np.any(seq[:, L_HIP] != 0, axis=1) & np.any(seq[:, R_HIP] != 0, axis=1))
    hip_ctr = 0.5 * (seq[:, L_HIP] + seq[:, R_HIP])
    sh_ctr = 0.5 * (seq[:, L_SHOULDER] + seq[:, R_SHOULDER])
    ctr = hip_ctr.copy(); ctr[~hips_ok] = sh_ctr[~hips_ok]
    seq = seq - ctr[:, None, :]

    dists = []
    for t in range(T):
        fr = seq[t]
        cues = [d for d in (
            _pair_dist(fr, L_SHOULDER, R_SHOULDER),
            _pair_dist(fr, L_HIP, R_HIP),
            _pair_dist(fr, L_SHOULDER, L_HIP)) if d is not None]
        dists.append(np.median(cues) if cues else np.nan)
    vals = np.asarray(dists, dtype=np.float32)
    scale = np.nanmedian(vals) if np.isfinite(vals).any() else 1.0
    if not np.isfinite(scale) or scale < 1e-3:
        scale = 1.0
    seq = np.clip(seq / scale, -10.0, 10.0)
    return np.nan_to_num(seq, nan=0.0).astype(np.float32)

# --------- Focal Loss ---------
class FocalLoss(nn.Module):
    """Focal Loss with optional class weights and label smoothing."""
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha  # (C,) tensor on device
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        log_probs = nn.functional.log_softmax(inputs, dim=-1)
        num_classes = inputs.size(-1)

        if self.label_smoothing > 0.0:
            smooth = torch.zeros_like(inputs).scatter_(1, targets.unsqueeze(1), 1.0)
            smooth = smooth * (1 - self.label_smoothing) + self.label_smoothing / num_classes
            ce_loss = -(smooth * log_probs).sum(dim=-1)  # (B,)
            if self.alpha is not None:
                w = self.alpha[targets]
                ce_loss = ce_loss * w
        else:
            ce_loss = nn.functional.nll_loss(log_probs, targets, reduction='none', weight=self.alpha)

        pt = log_probs.exp().gather(1, targets.unsqueeze(1)).squeeze(1)
        focal = (1 - pt).pow(self.gamma)
        return (focal * ce_loss).mean()

# --------- Model (Backbone + Multi-View Fusion) ---------
class SE1d(nn.Module):
    def __init__(self, ch: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, ch // reduction)
        self.fc1 = nn.Conv1d(ch, hidden, kernel_size=1, bias=True)
        self.fc2 = nn.Conv1d(hidden, ch, kernel_size=1, bias=True)
        self.act = nn.SiLU(); self.gate = nn.Sigmoid()
    def forward(self, x):
        s = x.mean(dim=2, keepdim=True)
        s = self.act(self.fc1(s))
        s = self.gate(self.fc2(s))
        return x * s

class DSResBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1, drop: float = 0.1, stochastic_depth: float = 0.0):
        super().__init__()
        self.dw = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation,
                            dilation=dilation, groups=channels, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.pw = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(drop)
        self.se = SE1d(channels, reduction=8)
        self.sd = float(stochastic_depth)
    def forward(self, x):
        # Drop-path that is compiler-friendly (no .item())
        if self.training and self.sd > 0.0:
            keep_prob = 1.0 - self.sd
            shape = (x.size(0),) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            binary_mask = torch.floor(random_tensor)
        out = self.dw(x)
        out = self.bn1(out); out = self.act(out)
        out = self.pw(out);  out = self.bn2(out)
        out = self.se(out);  out = self.drop(out)
        if self.training and self.sd > 0.0:
            out = out / keep_prob * binary_mask
        out = self.act(out + x)
        return out

class Backbone1D(nn.Module):
    """Backbone that maps (B, T, F) → embedding (B, 2*width) using temporal attn + GAP."""
    def __init__(self, in_features: int, width: int, drop: float, stochastic_depth: float, dilations: List[int]):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_features, width, kernel_size=1, bias=False),
            nn.BatchNorm1d(width),
            nn.SiLU(),
        )
        sd_rates = [stochastic_depth * i / max(1, len(dilations) - 1) for i in range(len(dilations))]
        self.blocks = nn.ModuleList([DSResBlock(width, d, drop=drop, stochastic_depth=sd) for d, sd in zip(dilations, sd_rates)])
        self.attn = nn.Conv1d(width, 1, kernel_size=1)
        self.norm = nn.LayerNorm(2 * width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        x = x.transpose(1, 2)            # (B, F, T)
        h = self.stem(x)                  # (B, W, T)
        for blk in self.blocks:
            h = blk(h)
        w = torch.softmax(self.attn(h), dim=2)   # (B, 1, T)
        z_attn = (h * w).sum(dim=2)              # (B, W)
        z_gap = h.mean(dim=2)                    # (B, W)
        z = torch.cat([z_attn, z_gap], dim=1)    # (B, 2W)
        z = self.norm(z)
        return z

class PoseTCNMultiView(nn.Module):
    """
    Multi-view classifier with fusion options:
      - early: concatenate views → single backbone → head
      - late: per-view backbone (shared weights), logits averaged (or attention)
      - hybrid: weighted sum of early and late logits

    Args:
      in_features_per_view = 33 * 3
      num_views: number of views concatenated in the dataset
      fusion: 'early' | 'late' | 'hybrid'
      view_fusion: 'mean' | 'attn' (for late)
      hybrid_alpha: weight for early logits in [0,1] (late gets 1-alpha)
    """
    def __init__(self, num_classes: int, num_views: int, width: int = 384, drop: float = 0.1,
                 stochastic_depth: float = 0.05, dilations: Optional[List[int]] = None,
                 fusion: str = "early", view_fusion: str = "mean", hybrid_alpha: float = 0.5):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8, 16, 32]
        self.num_classes = num_classes
        self.num_views = num_views
        self.fusion = fusion
        self.view_fusion = view_fusion
        self.hybrid_alpha = float(hybrid_alpha)

        in_per_view = NUM_LANDMARKS * 3
        in_early = in_per_view * num_views

        # Early path backbone + head
        self.backbone_early = Backbone1D(in_features=in_early, width=width, drop=drop,
                                         stochastic_depth=stochastic_depth, dilations=dilations)
        self.head_early = nn.Linear(2 * width, num_classes)

        # Late path backbone (shared across views) + head
        self.backbone_late = Backbone1D(in_features=in_per_view, width=width, drop=drop,
                                        stochastic_depth=stochastic_depth, dilations=dilations)
        self.head_late = nn.Linear(2 * width, num_classes)

        # Optional attention to fuse per-view embeddings in late mode
        if self.view_fusion == "attn":
            self.view_attn = nn.Sequential(
                nn.LayerNorm(2 * width),
                nn.Linear(2 * width, 1)
            )
        else:
            self.view_attn = None
    def _backbone_late_chunked(self, xv: torch.Tensor, chunk: int = 0) -> torch.Tensor:
        # xv: (B*V, T, F_late)
        if chunk is None or chunk <= 0 or xv.size(0) <= chunk:
            return self.backbone_late(xv)
        outs = []
        for i in range(0, xv.size(0), chunk):
            outs.append(self.backbone_late(xv[i:i+chunk]))
        return torch.cat(outs, dim=0)

    def _split_views(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F=33*3*num_views) → (B, V, T, 33*3)
        B, T, F = x.shape
        per_view = NUM_LANDMARKS * 3
        assert F == per_view * self.num_views, f"Expected F={per_view*self.num_views}, got {F}"
        x = x.view(B, T, self.num_views, per_view)
        x = x.permute(0, 2, 1, 3).contiguous()   # (B, V, T, per_view)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, F) where F = per_view * num_views
        returns logits: (B, C)
        """
        logits_early = None
        logits_late = None

        # Early fusion path
        if self.fusion in ("early", "hybrid"):
            z_early = self.backbone_early(x)
            logits_early = self.head_early(z_early)

        # Late fusion path
        if self.fusion in ("late", "hybrid"):
            xv = self._split_views(x)                      # (B, V, T, Fv)
            B, V, T, Fv = xv.shape
            xv = xv.view(B * V, T, Fv)
            z = self.backbone_late(xv)                     # (B*V, 2W)
            z = z.view(B, V, -1)                           # (B, V, 2W)

            if self.view_fusion == "attn" and self.view_attn is not None:
                scores = self.view_attn(z).squeeze(-1)     # (B, V)
                w = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, V, 1)
                z_fused = (z * w).sum(dim=1)               # (B, 2W)
            else:
                z_fused = z.mean(dim=1)                    # (B, 2W)

            logits_late = self.head_late(z_fused)

        if self.fusion == "early":
            return logits_early
        elif self.fusion == "late":
            return logits_late
        else:
            # hybrid: weighted sum of logits
            a = self.hybrid_alpha
            return a * logits_early + (1.0 - a) * logits_late

# --------- Subject Inference ---------
def infer_subject_from_meta(npz_path: str, meta: dict) -> str:
    """
    Infer subject from filename or metadata.
    Handles:
    - Full names: FirstName_LastName (two consecutive capitalized words)
    - Partial names: single_lowercase_word (matches against known subjects)
    - Timestamps: YYYY-MM-DD HH-MM-SS (becomes Date_YYYYMMDD_HHMMSS)
    - Metadata video_filename field
    """
    base = Path(npz_path).name
    parts = base.split('_')

    known_subjects_map = {
        # Lastname-only partials
        'ella': 'Cummings_Ella',
        'garcia': 'Garcia_Anika',
        'mia': 'Andrilla_Mia',
        'christopher': 'Andrilla_Christopher',
        'lilian': 'Barrientos_Lilian',
        'jose': 'Montenegro_Jose',
        'shelley': 'Sinegal_Shelley',
        'carrie': 'Cummings_Carrie',
        'jesse': 'Diepenbrock_Jesse',
        'dekyi': 'Llahmo_Dekyi',
        'tenzin': 'Tswang_Tenzin',
        'mark': 'Borsody_Mark',
        'dylan': 'Goeppinger_Dylan',
        'lucy': 'Sinegal_Lucy',
        'denise': 'Castillo_Denise',
        'avneet': 'Sidhu_Avneet',
        'jacqui': 'Chua_Jacqui',
        'elena': 'Countouriotus_Elena',
        'jackson': 'Henn_Jackson',
        'pema': 'Namgyal_Pema',
        'tania': 'Perez_Tania',
        'naresh': 'Joshi_Naresh',
        'padilla': 'Ramirez_Padilla',
        'andrilla': 'Andrilla_Mia',
        'barrientos': 'Barrientos_Lilian',
        'montenegro': 'Montenegro_Jose',
        'sinegal': 'Sinegal_Shelley',
        'cummings': 'Cummings_Carrie',
        'diepenbrock': 'Diepenbrock_Jesse',
        'llahmo': 'Llahmo_Dekyi',
        'tswang': 'Tswang_Tenzin',
        'borsody': 'Borsody_Mark',
        'goeppinger': 'Goeppinger_Dylan',
        'castillo': 'Castillo_Denise',
        'sidhu': 'Sidhu_Avneet',
        'chua': 'Chua_Jacqui',
        'countouriotus': 'Countouriotus_Elena',
        'henn': 'Henn_Jackson',
        'namgyal': 'Namgyal_Pema',
        'perez': 'Perez_Tania',
        'joshi': 'Joshi_Naresh',
        'ramirez': 'Ramirez_Padilla',
        'lizeth': 'Ramirez_Padilla',
        'valencia': 'Samantha_Valencia',
    }

    # Triple-capital pattern (e.g., Ramirez_Padilla_Lizeth -> Ramirez_Padilla)
    for i in range(len(parts) - 2):
        a, b, c = parts[i], parts[i + 1], parts[i + 2]
        if (a and b and c and a[0].isupper() and b[0].isupper() and c[0].isupper()):
            return f"{a}_{b}"

    # Two-capital fallback
    for i in range(len(parts) - 1):
        a, b = parts[i], parts[i + 1]
        if a and b and a[0].isupper() and b[0].isupper():
            return f"{a}_{b}"

    # Partial name mapping
    for part in parts:
        part_lower = part.lower()
        if part_lower in known_subjects_map:
            return known_subjects_map[part_lower]

    # Timestamp pattern
    timestamp_pattern = r'(\d{4})-(\d{2})-(\d{2})\s*(?:(\d{2})-(\d{2})-(\d{2}))?'
    for part in parts:
        match = re.search(timestamp_pattern, part)
        if match:
            year, month, day = match.groups()[:3]
            time_part = match.groups()[3:6]
            if time_part[0]:
                hour, minute, second = time_part
                return f"Date_{year}{month}{day}_{hour}{minute}{second}"
            else:
                return f"Date_{year}{month}{day}"

    # Metadata video filename
    vf = meta.get('video_filename', '') or ''
    if vf:
        stem = Path(vf).stem
        toks = stem.split('_')
        for i in range(len(toks) - 1):
            a, b = toks[i], toks[i + 1]
            if a and b and a[0].isupper() and b[0].isupper():
                return f"{a}_{b}"

    return "UnknownSubject"

# --------- Subject Split Report ---------
def print_subject_split_report(full_ds, train_files_idx, val_files_idx):
    print("\n" + "="*80)
    print("SUBJECT SPLIT REPORT")
    print("="*80)

    train_subjects = [full_ds.meta_per_file[i]['subject'] for i in train_files_idx]
    val_subjects   = [full_ds.meta_per_file[i]['subject'] for i in val_files_idx]

    overlap = set(train_subjects) & set(val_subjects)
    print(f"Train subjects: {len(set(train_subjects))} | Val subjects: {len(set(val_subjects))}")
    if overlap:
        print(f"⚠️ Overlap detected in subjects: {overlap}")
    else:
        print("✅ No subject overlap between train and val")

    train_counts = Counter(train_subjects)
    val_counts   = Counter(val_subjects)

    print("\nTrain Subjects:")
    for subj in sorted(train_counts.keys()):
        print(f"  - {subj:<25} ({train_counts[subj]} files)")

    print("\nVal Subjects:")
    for subj in sorted(val_counts.keys()):
        print(f"  - {subj:<25} ({val_counts[subj]} files)")

    print("="*80 + "\n")

# --------- Dataset (Eager + Optional Lazy; Meta-Index Safe) ---------
class NPZWindowDataset(Dataset):
    def __init__(self, files: List[str], class_map: Dict[str, int],
                 T: int = 60, default_stride: int = 15,
                 target_stride_s: Optional[float] = None,
                 max_views: int = 3, use_visibility_mask: bool = False,
                 use_soft_labels: bool = False,
                 lazy_load: bool = False):
        """
        If lazy_load=False (default): eagerly loads all view arrays (fast training, higher RAM).
        If lazy_load=True: keeps only metadata and loads arrays on demand (lower RAM, slower).
        Uses **meta indices** everywhere to avoid file-index skew when some files are skipped.
        """
        self.files = files
        self.class_map = class_map
        self.T = T
        self.default_stride = default_stride
        self.target_stride_s = target_stride_s
        self.max_views = max_views
        self.use_visibility_mask = use_visibility_mask
        self.use_soft_labels = use_soft_labels
        self.lazy_load = lazy_load

        self.index: List[Tuple[int,int]] = []          # (meta_idx, start)
        self.meta_per_file: List[Dict] = []
        self.discovered_labels = set()

        # Storage depending on lazy mode
        if not self.lazy_load:
            self.views_per_file: List[List[np.ndarray]] = []
            self.frame_labels_per_file: List[Optional[np.ndarray]] = []
        else:
            self._lazy_cache = {}  # per-process cache (path -> np.load) will be opened on demand

        failed_files = []

        for fi, path in enumerate(self.files):
            try:
                with np.load(path, allow_pickle=False) as z:
                    view_keys = [k for k in z.files if k.startswith("view_")]
                    if not view_keys:
                        continue
                    view_keys = view_keys[: self.max_views]

                    # metadata
                    meta = {}
                    for k in z.files:
                        if k in view_keys:
                            continue
                        arr = z[k]
                        if getattr(arr, 'shape', ()) == ():
                            try:
                                v = arr.item()
                                if isinstance(v, (bytes, bytearray)):
                                    v = v.decode(errors="ignore")
                                meta[k] = v
                            except Exception:
                                meta[k] = str(arr)
                        else:
                            meta[k] = arr

                    # label
                    label_name = meta.get("movement_type", None)
                    if label_name is None or str(label_name).strip() == "":
                        continue
                    label_name = str(label_name).strip()
                    self.discovered_labels.add(label_name)

                    # fps & subject
                    fps = float(meta.get("fps", 60.0) or 60.0)
                    subject = infer_subject_from_meta(path, meta)

                    # views & frame labels
                    if not self.lazy_load:
                        view_arrays = []
                        for vk in view_keys:
                            arr = np.asarray(z[vk], dtype=np.float32)
                            if arr.ndim != 3 or arr.shape[-1] < 3:
                                continue
                            xyz = arr[..., :3]
                            if self.use_visibility_mask and arr.shape[-1] >= 4:
                                vis = arr[..., 3]
                                mask = (vis < 0.2).astype(np.float32)
                                xyz = xyz * (1.0 - mask[..., None])
                            view_arrays.append(xyz)
                        if not view_arrays:
                            continue
                        Tlen = min(a.shape[0] for a in view_arrays)
                        view_arrays = [a[:Tlen] for a in view_arrays]

                        # optional frame labels
                        frame_labels = None
                        if self.use_soft_labels:
                            if 'frame_labels' in z:
                                frame_labels = np.asarray(z['frame_labels'])
                            elif 'labels_per_frame' in z:
                                frame_labels = np.asarray(z['labels_per_frame'])
                    else:
                        # lazy: do not read arrays; just infer length quickly
                        Tlen = None
                        for vk in view_keys:
                            shape = z[vk].shape
                            if Tlen is None:
                                Tlen = shape[0]
                            else:
                                Tlen = min(Tlen, shape[0])
                        if Tlen is None:
                            continue

                    # too short
                    if Tlen < self.T:
                        continue

                    # windows
                    stride = (
                        max(1, int(round(fps * self.target_stride_s)))
                        if self.target_stride_s and fps > 0 else self.default_stride
                    )
                    starts = list(range(0, Tlen - self.T + 1, stride))
                    if not starts:
                        starts = [0]

                    # append meta
                    meta_idx = len(self.meta_per_file)
                    self.meta_per_file.append({
                        "path": path, "label": label_name, "fps": fps,
                        "frames": Tlen, "subject": subject, "views": len(view_keys),
                    })

                    # append arrays or placeholders
                    if not self.lazy_load:
                        self.views_per_file.append(view_arrays)
                        self.frame_labels_per_file.append(frame_labels)

                    # window indices store meta_idx
                    self.index.extend([(meta_idx, s) for s in starts])

            except Exception as e:
                failed_files.append((path, str(e)))
                continue

        if failed_files:
            print(f"⚠️  Failed to load {len(failed_files)} files:")
            for path, err in failed_files[:5]:
                print(f"  - {Path(path).name}: {err}")
            if len(failed_files) > 5:
                print(f"  ... and {len(failed_files) - 5} more")

        self.num_views = min(self.max_views, max((m.get("views", 1) for m in self.meta_per_file), default=1))
        self.input_dim = NUM_LANDMARKS * 3 * self.num_views
        self.discovered_labels = sorted(self.discovered_labels)

        if self.use_soft_labels and not self.lazy_load:
            has_frame_labels = sum(1 for fl in self.frame_labels_per_file if fl is not None)
            if has_frame_labels > 0:
                print(f"\n📋 Frame-level labels found: {has_frame_labels}/{len(self.meta_per_file)} files")

        if self.lazy_load:
            print("   🔄 Lazy loading enabled (lower memory, potentially slower)")

    def __len__(self):
        return len(self.index)

    def _load_views_lazy(self, meta_idx: int) -> List[np.ndarray]:
        """Load views on demand; avoid keeping too many open handles."""
        path = self.meta_per_file[meta_idx]['path']
        z = np.load(path, allow_pickle=False)
        try:
            view_keys = [k for k in z.files if k.startswith("view_")][: self.max_views]
            out = []
            for vk in view_keys:
                arr = np.asarray(z[vk], dtype=np.float32)
                if arr.ndim != 3 or arr.shape[-1] < 3:
                    continue
                xyz = arr[..., :3]
                if self.use_visibility_mask and arr.shape[-1] >= 4:
                    vis = arr[..., 3]
                    mask = (vis < 0.2).astype(np.float32)
                    xyz = xyz * (1.0 - mask[..., None])
                out.append(xyz)
            return out
        finally:
            try:
                z.close()
            except Exception:
                pass

    def _load_frame_labels_lazy(self, meta_idx: int) -> Optional[np.ndarray]:
        if not self.use_soft_labels:
            return None
        path = self.meta_per_file[meta_idx]['path']
        z = np.load(path, allow_pickle=False)
        try:
            if 'frame_labels' in z:
                return np.asarray(z['frame_labels'])
            elif 'labels_per_frame' in z:
                return np.asarray(z['labels_per_frame'])
            else:
                return None
        finally:
            try:
                z.close()
            except Exception:
                pass

    def _compute_soft_label(self, frame_labels: np.ndarray, start: int) -> np.ndarray:
        window_labels = frame_labels[start:start + self.T]
        counts = Counter([str(l).strip() for l in window_labels])
        soft = np.zeros(len(self.class_map), dtype=np.float32)
        for label_str, cnt in counts.items():
            if label_str in self.class_map:
                soft[self.class_map[label_str]] = cnt / len(window_labels)
        return soft

    def __getitem__(self, idx):
        meta_idx, s = self.index[idx]
        if self.lazy_load:
            views = self._load_views_lazy(meta_idx)
            frame_labels = self._load_frame_labels_lazy(meta_idx) if self.use_soft_labels else None
        else:
            views = self.views_per_file[meta_idx]
            frame_labels = self.frame_labels_per_file[meta_idx] if self.use_soft_labels else None

        seqs = []
        num_v = min(len(views), self.num_views)
        for v in range(num_v):
            seq = views[v][s: s + self.T]         # (T, 33, 3)
            seq = normalize_single_view(seq)      # per-window normalization
            seqs.append(seq)

        if num_v == 0:
            pad = np.zeros((self.T, NUM_LANDMARKS, 3), dtype=np.float32)
            seqs = [pad] * self.num_views
        elif num_v < self.num_views:
            base = seqs[0]
            seqs.extend([base.copy() for _ in range(self.num_views - num_v)])

        fused = np.concatenate(seqs, axis=2).astype(np.float32)   # (T, 33, 3*num_views)
        label = self.meta_per_file[meta_idx]["label"]
        subj  = self.meta_per_file[meta_idx]["subject"]

        if self.use_soft_labels and frame_labels is not None:
            y_soft = self._compute_soft_label(frame_labels, s)   # (C,)
            return fused, y_soft, subj, True

        y = self.class_map[label]
        return fused, y, subj, False

# --------- Temporal Artifact Analysis & Filtering ---------
def analyze_window_purity(dataset: NPZWindowDataset) -> Dict:
    """Analyze temporal segmentation artifacts in the dataset (requires eager + frame labels)."""
    if dataset.lazy_load:
        print("⚠️  Lazy load is enabled; skipping purity analysis (needs eager data).")
        return {}

    if not dataset.use_soft_labels:
        print("⚠️  Soft labels disabled; no frame-level labels available for analysis.")
        return {}

    if not any(fl is not None for fl in dataset.frame_labels_per_file):
        print("⚠️  No frame-level labels found.")
        return {}

    purities_by_class = defaultdict(list)
    all_purities = []

    for meta_idx, start in dataset.index:
        frame_labels = dataset.frame_labels_per_file[meta_idx]
        if frame_labels is None:
            continue
        window_labels = frame_labels[start:start + dataset.T]
        counts = Counter([str(l).strip() for l in window_labels])
        dominant_label, dominant_count = counts.most_common(1)[0]
        purity = dominant_count / len(window_labels)
        purities_by_class[dominant_label].append(purity)
        all_purities.append(purity)

    if not all_purities:
        print("⚠️  No analyzable windows for purity.")
        return {}

    print(f"\n{'='*80}")
    print("TEMPORAL SEGMENTATION ARTIFACT ANALYSIS")
    print(f"{'='*80}")
    print(f"Overall windows: {len(all_purities)}")
    print(f"  Mean purity:   {np.mean(all_purities):.1%}")
    print(f"  Median purity: {np.median(all_purities):.1%}")
    pct80 = 100 * sum(p >= 0.8 for p in all_purities) / len(all_purities)
    pct90 = 100 * sum(p >= 0.9 for p in all_purities) / len(all_purities)
    print(f"  ≥80% purity:   {pct80:.1f}% of windows")
    print(f"  ≥90% purity:   {pct90:.1f}% of windows")

    print("\nPer-Class Contamination (1 - purity):")
    print("-" * 80)
    avg_contam = {}
    for cls in sorted(purities_by_class.keys()):
        purs = purities_by_class[cls]
        contam = [1.0 - p for p in purs]
        avg_contam[cls] = np.mean(contam)
        print(f"  {cls:>20s}: mean={np.mean(contam):.1%}, median={np.median(contam):.1%}, "
              f"max={np.max(contam):.1%}, n={len(purs)}")

    worst_class = max(avg_contam, key=avg_contam.get)
    worst_contam = avg_contam[worst_class]

    print(f"\nRECOMMENDATIONS:")
    print("-" * 80)
    if worst_contam > 0.3:
        print(f"🚨 SEVERE: '{worst_class}' has {worst_contam:.1%} average contamination")
        print(f"   → Use --filter_window_purity 0.8 to exclude contaminated windows")
        print(f"   → Or reduce window size: --T 30 (current: {dataset.T})")
    elif worst_contam > 0.15:
        print(f"⚠️  MODERATE: '{worst_class}' has {worst_contam:.1%} average contamination")
        print(f"   → Consider --use_soft_labels for gradient accuracy")
        print(f"   → Or use --filter_window_purity 0.8")
    else:
        print(f"✓ LOW contamination (max {worst_contam:.1%} for '{worst_class}')")
    print(f"{'='*80}\n")

    return {
        'purities_by_class': purities_by_class,
        'all_purities': all_purities,
        'worst_class': worst_class,
        'worst_contamination': worst_contam
    }

def filter_windows_by_purity(dataset: NPZWindowDataset, train_indices: List[int], min_purity: float = 0.8) -> List[int]:
    """Filter training windows to only include pure examples (requires eager + frame labels)."""
    if dataset.lazy_load:
        print(f"⚠️  Lazy load enabled; cannot filter by purity.")
        return train_indices

    if not dataset.use_soft_labels:
        print(f"⚠️  Soft labels disabled; cannot filter by purity.")
        return train_indices

    if not any(fl is not None for fl in dataset.frame_labels_per_file):
        print(f"⚠️  No frame-level labels for purity filtering.")
        return train_indices

    pure_indices = []
    for idx in train_indices:
        meta_idx, start = dataset.index[idx]
        frame_labels = dataset.frame_labels_per_file[meta_idx]
        if frame_labels is None:
            pure_indices.append(idx)
            continue
        window_labels = frame_labels[start:start + dataset.T]
        counts = Counter([str(l).strip() for l in window_labels])
        _, dominant_count = counts.most_common(1)[0]
        purity = dominant_count / len(window_labels)
        if purity >= min_purity:
            pure_indices.append(idx)

    print(f"\n🎯 Window Purity Filtering:")
    print(f"   Threshold: {min_purity:.1%}")
    print(f"   Windows kept: {len(pure_indices)}/{len(train_indices)} "
          f"({100*len(pure_indices)/max(1,len(train_indices)):.1f}%)")
    print(f"   Windows filtered: {len(train_indices) - len(pure_indices)}")

    return pure_indices

# --------- Balancing: Samplers & Weights ---------
def make_balanced_sampler_for_subset(full_ds: NPZWindowDataset, subset_indices: List[int]) -> WeightedRandomSampler:
    """
    Weighted sampler balancing label and subject frequency **within the train subset only**.
    """
    file_to_nwin = defaultdict(int)
    for win_idx in subset_indices:
        meta_idx, _ = full_ds.index[win_idx]
        file_to_nwin[meta_idx] += 1

    label_counts = Counter()
    subject_counts = Counter()
    for meta_idx, n in file_to_nwin.items():
        lab = full_ds.meta_per_file[meta_idx]['label']
        sub = full_ds.meta_per_file[meta_idx]['subject']
        label_counts[lab] += n
        subject_counts[sub] += n

    label_w = {lab: 1.0 / max(c, 1) for lab, c in label_counts.items()}
    lw_mean = np.mean(list(label_w.values())) if label_w else 1.0
    label_w = {k: v / lw_mean for k, v in label_w.items()}

    subject_w = {sub: 1.0 / max(c, 1) for sub, c in subject_counts.items()}
    sw_mean = np.mean(list(subject_w.values())) if subject_w else 1.0
    subject_w = {k: v / sw_mean for k, v in subject_w.items()}

    weights = []
    for win_idx in subset_indices:
        meta_idx, _ = full_ds.index[win_idx]
        lab = full_ds.meta_per_file[meta_idx]['label']
        sub = full_ds.meta_per_file[meta_idx]['subject']
        weights.append(label_w.get(lab, 1.0) * subject_w.get(sub, 1.0))

    return WeightedRandomSampler(
        torch.tensor(weights, dtype=torch.float32),
        num_samples=len(weights),
        replacement=True
    )

def class_weights_from_train_subset(
    full_ds: NPZWindowDataset,
    train_subset: Subset,
    class_map: Dict[str, int],
    pow_smooth: float = 0.5,
    clip: Tuple[float,float] = (0.5, 8.0)
) -> torch.Tensor:
    """
    Smoothed and clipped class weights:
    - pow_smooth < 1.0 reduces extremes
    - clip caps weights into [clip_min, clip_max]
    """
    counts = np.zeros(len(class_map), dtype=np.int64)
    for win_idx in train_subset.indices:
        meta_idx, _ = full_ds.index[win_idx]
        lab = full_ds.meta_per_file[meta_idx]['label']
        counts[class_map[lab]] += 1

    inv = counts.sum() / np.clip(counts, 1, None)
    w = inv / inv.mean()
    w = np.power(w, pow_smooth)
    w = np.clip(w, clip[0], clip[1])
    return torch.tensor(w, dtype=torch.float32)

# --------- Collate with Class-Aware Augmentation ---------
class CollateWindows:
    def __init__(self, augment: bool = False,
                 time_mask_prob: float = 0.0, time_mask_max_frames: int = 8,
                 joint_dropout_prob: float = 0.0, joint_dropout_frac: float = 0.15,
                 noise_std: float = 0.0,
                 rotation_prob: float = 0.0, rotation_angle_deg: float = 15.0,
                 scale_prob: float = 0.0, scale_min: float = 0.9, scale_max: float = 1.1,
                 temporal_warp_prob: float = 0.0,
                 # Normal-class aware scaling
                 normal_class_name: str = "normal",
                 class_map: Optional[Dict[str,int]] = None,
                 aug_normal_scale: float = 0.5,
                 aug_normal_overrides: Optional[Dict[str,float]] = None):
        """
        If augment=True, applies geometric/temporal/noise augmentations.
        For 'normal' class, probabilities can be scaled down by aug_normal_scale to
        preserve calibration, or overridden via aug_normal_overrides dict with keys:
          ['time_mask_prob','joint_dropout_prob','noise_std','rotation_prob','scale_prob','temporal_warp_prob']
        """
        self.augment = augment
        self.time_mask_prob = time_mask_prob
        self.time_mask_max_frames = time_mask_max_frames
        self.joint_dropout_prob = joint_dropout_prob
        self.joint_dropout_frac = joint_dropout_frac
        self.noise_std = noise_std
        self.rotation_prob = rotation_prob
        self.rotation_angle_deg = rotation_angle_deg
        self.scale_prob = scale_prob
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.temporal_warp_prob = temporal_warp_prob

        self.normal_class_name = normal_class_name
        self.class_map = class_map or {}
        self.aug_normal_scale = float(aug_normal_scale)
        self.aug_normal_overrides = aug_normal_overrides or {}

        # Resolve normal class index if available
        self.normal_class_idx = None
        if self.normal_class_name in self.class_map:
            self.normal_class_idx = self.class_map[self.normal_class_name]

    def _rotate_y(self, pose_np, angle_rad):
        T, F = pose_np.shape
        if F % 3 != 0:
            return pose_np
        L = F // 3
        x = pose_np.reshape(T, L, 3)
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
        x = x @ R.T
        return x.reshape(T, F)

    def _temporal_warp(self, pose_np, rate_range=(0.8, 1.2)):
        T, F = pose_np.shape
        rate = float(np.random.uniform(*rate_range))
        new_T = int(max(4, round(T * rate)))
        old_idx = np.linspace(0, T - 1, T, dtype=np.float32)
        new_idx = np.linspace(0, T - 1, new_T, dtype=np.float32)
        out = np.empty((new_T, F), dtype=np.float32)
        for f in range(F):
            out[:, f] = np.interp(new_idx, old_idx, pose_np[:, f])
        if new_T >= T:
            return out[:T]
        pad = np.zeros((T, F), dtype=np.float32)
        pad[:new_T] = out
        pad[new_T:] = out[-1]  # edge replication (avoid sudden zeros)
        return pad

    def _get_aug_params_for_label(self, y_label: int):
        """Scale or override augmentation params for 'normal' class."""
        if self.normal_class_idx is None or y_label != self.normal_class_idx:
            return (self.time_mask_prob, self.joint_dropout_prob, self.noise_std,
                    self.rotation_prob, self.scale_prob, self.temporal_warp_prob)
        # apply scaling or overrides for normal
        tmp = dict(
            time_mask_prob=self.time_mask_prob * self.aug_normal_scale,
            joint_dropout_prob=self.joint_dropout_prob * self.aug_normal_scale,
            noise_std=self.noise_std * self.aug_normal_scale,
            rotation_prob=self.rotation_prob * self.aug_normal_scale,
            scale_prob=self.scale_prob * self.aug_normal_scale,
            temporal_warp_prob=self.temporal_warp_prob * self.aug_normal_scale
        )
        tmp.update({k: self.aug_normal_overrides[k] for k in self.aug_normal_overrides})
        return (tmp['time_mask_prob'], tmp['joint_dropout_prob'], tmp['noise_std'],
                tmp['rotation_prob'], tmp['scale_prob'], tmp['temporal_warp_prob'])

    def __call__(self, batch):
        # batch elements can be (x, y, subj, is_soft) where y is int OR soft label np.ndarray
        xs, ys, subs, is_softs = zip(*batch)
        x = torch.from_numpy(np.stack(xs, 0))  # (B, T, L, 3*V)
        # y: could be ints or soft labels
        if isinstance(ys[0], np.ndarray):
            y = torch.from_numpy(np.stack(ys, 0)).float()  # (B, C)
            y_is_soft = True
        else:
            y = torch.tensor(ys, dtype=torch.long)
            y_is_soft = False

        if x.dim() == 4:
            x = x.flatten(2, 3)  # (B, T, F)
        elif x.dim() != 3:
            raise ValueError(f"Unexpected input shape {tuple(x.shape)}")

        if not self.augment:
            return x, y, list(subs)

        B, T, F = x.shape
        x_np = x.numpy().copy()

        # Per-sample class-aware augmentation
        for b in range(B):
            # Get scalar class label if available
            label_for_aug = None
            if not y_is_soft:
                label_for_aug = int(y[b].item())

            tm_prob, jd_prob, noise_std, rot_prob, sc_prob, tw_prob = self._get_aug_params_for_label(
                label_for_aug if label_for_aug is not None else -1
            )

            sample = x_np[b]

            # Rotation
            if rot_prob > 0.0 and random.random() < rot_prob:
                angle = np.radians(random.uniform(-self.rotation_angle_deg, self.rotation_angle_deg))
                sample = self._rotate_y(sample, angle)

            # Scaling
            if sc_prob > 0.0 and random.random() < sc_prob:
                scale = float(np.random.uniform(self.scale_min, self.scale_max))
                sample = sample * scale

            # Temporal warp
            if tw_prob > 0.0 and random.random() < tw_prob:
                sample = self._temporal_warp(sample)

            x_np[b] = sample

            # Time mask
            if tm_prob > 0.0 and self.time_mask_max_frames > 0 and random.random() < tm_prob:
                L = random.randint(1, min(self.time_mask_max_frames, T))
                s = random.randint(0, T - L)
                x_np[b, s:s + L, :] = 0.0

            # Joint dropout (random joints -> zeroed)
            if jd_prob > 0.0 and self.joint_dropout_frac > 0.0 and (F % 3 == 0) and random.random() < jd_prob:
                joints_total = F // 3
                k = max(1, int(round(joints_total * self.joint_dropout_frac)))
                drop_idx = np.random.choice(joints_total, size=k, replace=False)
                for j in drop_idx:
                    x_np[b, :, 3 * j:3 * j + 3] = 0.0

            # Noise
            if noise_std > 0.0:
                x_np[b] += np.random.randn(*x_np[b].shape).astype(np.float32) * float(noise_std)

        x = torch.from_numpy(x_np)
        return x, y, list(subs)

# --------- Evaluation (BF16-safe + Temperature scaling) ---------
@torch.no_grad()
@torch.no_grad()
def evaluate(model, loader, device, criterion_eval=None, return_preds: bool=False, temperature: float = 1.0):
    """
    BF16-safe: logits converted to fp32 before argmax or further ops.
    Temperature scaling supported (post-hoc calibration).
    """
    model.eval()
    ys, ps = [], []
    logits_all = []
    n, loss_sum = 0, 0.0
    use_cuda = (device.type == 'cuda')
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()

    for xb, yb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        y_is_soft = torch.is_floating_point(yb) if isinstance(yb, torch.Tensor) else False
        yb = yb.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16 if use_bf16 else torch.float16, enabled=use_cuda):
            logits = model(xb)
            if criterion_eval is not None:
                if y_is_soft:
                    log_probs = nn.functional.log_softmax(logits, dim=-1)
                    ce_soft = -(yb * log_probs).sum(dim=-1).mean()
                    loss_sum += ce_soft.item() * xb.size(0)
                else:
                    loss_sum += criterion_eval(logits, yb).item() * xb.size(0)

        logits = logits.float()  # BF16-safe cast
        scaled_logits = logits / float(temperature)
        pred = scaled_logits.argmax(1)

        ps.append(pred.cpu().numpy())
        if y_is_soft:
            ys.append(yb.argmax(dim=1).cpu().numpy())
        else:
            ys.append(yb.cpu().numpy())

        if return_preds:
            logits_all.append(scaled_logits.cpu().numpy())
        n += xb.size(0)

    y = np.concatenate(ys) if ys else np.array([])
    p = np.concatenate(ps) if ps else np.array([])
    out = {
        'acc': float(accuracy_score(y, p)) if len(y) > 0 else 0.0,
        'balanced_acc': float(balanced_accuracy_score(y, p)) if len(y) > 0 else 0.0,
        'macro_f1': float(f1_score(y, p, average='macro', zero_division=0)) if len(y) > 0 else 0.0
    }
    if criterion_eval is not None and n > 0:
        out['val_loss'] = loss_sum / n
    if return_preds:
        out.update({'y': y, 'p': p})
        if logits_all:
            out['logits'] = np.concatenate(logits_all)
    return out

def find_best_temperature(model, val_loader, device, T_range=None):
    """Grid search temperature to minimize NLL on validation set."""
    if T_range is None:
        T_range = np.linspace(0.7, 2.0, 14)
    model.eval()
    all_logits, all_labels = [], []
    use_cuda = (device.type == 'cuda')
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()

    with torch.no_grad():
        for xb, yb, _ in val_loader:
            xb = xb.to(device, non_blocking=True)
            y_is_soft = torch.is_floating_point(yb) if isinstance(yb, torch.Tensor) else False

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16 if use_bf16 else torch.float16, enabled=use_cuda):
                logits = model(xb)
            all_logits.append(logits.float().cpu())  # BF16-safe

            if y_is_soft:
                all_labels.append(yb.argmax(dim=1).cpu())
            else:
                all_labels.append(yb.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    best_T, best_nll = 1.0, float('inf')
    for T in T_range:
        scaled = all_logits / float(T)
        nll = nn.functional.cross_entropy(scaled, all_labels).item()
        if nll < best_nll:
            best_nll = nll
            best_T = float(T)
    return best_T, best_nll


# --------- Training ---------
def train_one_epoch(model, loader, optimizer, device, accumulation_steps=2, scaler=None,
                    criterion=None, mixup_alpha: float = 0.0, grad_clip: float = 0.0):
    model.train()
    total = 0.0
    n = 0

    class _NoScaler:
        def is_enabled(self): return False
        def scale(self, x): return x
        def step(self, opt): opt.step()
        def update(self): pass
        def unscale_(self, opt): pass

    scaler = scaler or _NoScaler()
    optimizer.zero_grad(set_to_none=True)
    use_cuda = (device.type == 'cuda')
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()

    for b_idx, (xb, yb, _) in enumerate(loader):
        xb = xb.to(device, non_blocking=True)

        # robust soft-labels detection
        y_is_soft = torch.is_floating_point(yb) if isinstance(yb, torch.Tensor) else False
        yb = yb.to(device, non_blocking=True)

        lam = 1.0
        yb2 = None
        if mixup_alpha > 0 and xb.size(0) > 1 and not y_is_soft:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            perm = torch.randperm(xb.size(0), device=xb.device)
            xb = lam * xb + (1 - lam) * xb[perm]
            yb2 = yb[perm]

        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16 if use_bf16 else torch.float16, enabled=use_cuda):
            logits = model(xb)
            if y_is_soft:
                # soft targets CE
                log_probs = nn.functional.log_softmax(logits, dim=-1)
                loss = -(yb * log_probs).sum(dim=-1).mean()
            else:
                if yb2 is None:
                    loss = criterion(logits, yb)
                else:
                    loss = lam * criterion(logits, yb) + (1.0 - lam) * criterion(logits, yb2)

        if scaler.is_enabled():
            scaler.scale(loss / accumulation_steps).backward()
        else:
            (loss / accumulation_steps).backward()

        if (b_idx + 1) % accumulation_steps == 0:
            if grad_clip > 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        bs = xb.size(0)
        total += loss.item() * bs
        n += bs

    # Final flush
    if (len(loader) % accumulation_steps) != 0:
        if grad_clip > 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return total / max(1, n)


# --------- Plotting ---------
def plot_training_curves(history: dict, out_dir: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].plot(history['train_loss'], label='Train')
    if 'val_loss' in history:
        axes[0, 0].plot(history['val_loss'], label='Val')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].set_xlabel('Epoch')

    axes[0, 1].plot(history['val_acc'], label='Val Acc')
    axes[0, 1].plot(history['val_balanced_acc'], label='Val Balanced Acc')
    axes[0, 1].set_title("Accuracy Metrics")
    axes[0, 1].legend()

    axes[1, 0].plot(history['val_macro_f1'], label='Val Macro F1')
    axes[1, 0].set_title("Macro F1")
    axes[1, 0].legend()

    axes[1, 1].plot(history['lr'], label='Learning Rate')
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_title('LR')
    axes[1, 1].legend()

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()

# --------- Class Distribution Diagnostic ---------
def print_class_distribution(full_ds, train_ds, val_ds, classes):
    print("\n" + "="*80)
    print("CLASS DISTRIBUTION DIAGNOSTIC")
    print("="*80)

    train_class_dist = Counter()
    for idx in train_ds.indices:
        meta_idx, _ = full_ds.index[idx]
        train_class_dist[full_ds.meta_per_file[meta_idx]['label']] += 1

    val_class_dist = Counter()
    for idx in val_ds.indices:
        meta_idx, _ = full_ds.index[idx]
        val_class_dist[full_ds.meta_per_file[meta_idx]['label']] += 1

    print("\nTRAIN vs VAL distribution:")
    print(f"{'Class':<20} {'Train':>10} {'Val':>10} {'Train%':>8} {'Val%':>8} {'Weight':>8}")
    print("-" * 80)
    for cls in classes:
        tr = train_class_dist.get(cls, 0)
        va = val_class_dist.get(cls, 0)
        tr_pct = 100 * tr / max(1, len(train_ds))
        va_pct = 100 * va / max(1, len(val_ds))
        weight = len(train_ds) / max(1, tr) if tr > 0 else 0
        print(f"{cls:<20} {tr:>10} {va:>10} {tr_pct:>7.2f}% {va_pct:>7.2f}% {weight:>7.2f}x")

    if train_class_dist:
        max_train = max(train_class_dist.values())
        min_train = min(train_class_dist.values())
        ratio = max_train / max(1, min_train)
        print(f"\n{'Imbalance Ratio:':<20} {ratio:>7.1f}:1")
        if ratio > 20:
            print("🚨 SEVERE IMBALANCE! Consider sampler and smoothed/clipped weights.")
        elif ratio > 10:
            print("⚠️  MODERATE IMBALANCE. Sampler or weights will help.")
        else:
            print("✓ Relatively balanced dataset.")
    print("="*80 + "\n")

# --------- Get Current LR ---------
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0

# --------- Worker Init (RNG per worker) ---------
def worker_init_fn(worker_id: int):
    # Ensure different RNG streams per worker
    seed = np.random.SeedSequence().entropy
    random.seed(seed + worker_id)
    np.random.seed(int((seed + worker_id) % (2**32 - 1)))

# --------- Main ---------
def main():
    import argparse
    ap = argparse.ArgumentParser(description="PoseTCN — Fast + Feature-Rich (Hybrid Fusion + Balancing + Temporal Tools)")

    # Paths & basics
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--out", type=str, default="runs_cnn")
    ap.add_argument("--seed", type=int, default=42)

    # Data & windowing
    ap.add_argument("--T", type=int, default=60)
    ap.add_argument("--target_stride_s", type=float, default=0.25)
    ap.add_argument("--default_stride", type=int, default=15)
    ap.add_argument("--max_views", type=int, default=3)
    ap.add_argument("--use_visibility_mask", action="store_true")
    ap.add_argument("--val_size", type=float, default=0.2)

    # Dataloader performance
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--lazy_load", action="store_true",
                    help="Load arrays on-the-fly (lower memory, slower). Default eager load for speed.")
    ap.add_argument("--mp_start_method", type=str, default=None, help="spawn (Windows) / forkserver / fork")

    # Model
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--stochastic_depth", type=float, default=0.05)
    ap.add_argument("--dilations", type=str, default="1,2,4,8,16,32,64")

    # Fusion
    ap.add_argument("--fusion", type=str, default="hybrid", choices=["early", "late", "hybrid"])
    ap.add_argument("--view_fusion", type=str, default="mean", choices=["mean", "attn"])
    ap.add_argument("--hybrid_alpha", type=float, default=0.5, help="Weight for early logits in hybrid mode")

    # Training
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=382)
    ap.add_argument("--accumulation_steps", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--ckpt_prefix", type=str, default="cnn_best")
    ap.add_argument("--report_each", type=int, default=5)
    ap.add_argument("--early_stop_patience", type=int, default=12)

    # Imbalance handling
    ap.add_argument("--imbalance_strategy", type=str, default="sampler", choices=["sampler", "weights", "none"],
                    help="sampler: balanced sampler; weights: class weights only; none: neither")
    ap.add_argument("--use_focal_loss", action="store_true")
    ap.add_argument("--focal_gamma", type=float, default=1.0)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--weight_smooth_power", type=float, default=0.5)
    ap.add_argument("--weight_clip_min", type=float, default=0.5)
    ap.add_argument("--weight_clip_max", type=float, default=8.0)

    # Augmentations
    ap.add_argument("--aug_enable", action="store_true")
    ap.add_argument("--aug_time_mask_prob", type=float, default=0.1)
    ap.add_argument("--aug_time_mask_max_frames", type=int, default=6)
    ap.add_argument("--aug_joint_dropout_prob", type=float, default=0.2)
    ap.add_argument("--aug_joint_dropout_frac", type=float, default=0.1)
    ap.add_argument("--aug_noise_std", type=float, default=0.01)
    ap.add_argument("--aug_rotation_prob", type=float, default=0.2)
    ap.add_argument("--aug_rotation_angle_deg", type=float, default=10.0)
    ap.add_argument("--aug_scale_prob", type=float, default=0.2)
    ap.add_argument("--aug_scale_min", type=float, default=0.95)
    ap.add_argument("--aug_scale_max", type=float, default=1.05)
    ap.add_argument("--aug_temporal_warp_prob", type=float, default=0.0)

    # Normal-class-aware augmentation
    ap.add_argument("--aug_normal_scale", type=float, default=0.5)
    ap.add_argument("--normal_class_name", type=str, default="normal")

    # Mixup
    ap.add_argument("--mixup_alpha", type=float, default=0.0)

    # Scheduler
    ap.add_argument("--use_cosine_schedule", action="store_true")
    ap.add_argument("--warmup_epochs", type=int, default=5)

    # Temporal artifact & soft labels
    ap.add_argument("--use_soft_labels", action="store_true")
    ap.add_argument("--analyze_temporal_artifacts", action="store_true")
    ap.add_argument("--filter_window_purity", type=float, default=0.0)

    # Calibration
    ap.add_argument("--calibrate_temperature", action="store_true")

    # Logging
    ap.add_argument("--plot_curves", action="store_true")
    ap.add_argument("--save_history", action="store_true")

    args = ap.parse_args()

    # Multiprocessing start method
    if args.mp_start_method:
        try:
            mp.set_start_method(args.mp_start_method, force=True)
        except (RuntimeError, ValueError):
            pass
    else:
        try:
            method = "spawn" if platform.system() == "Windows" else "forkserver"
            mp.set_start_method(method, force=True)
        except (RuntimeError, ValueError):
            pass

    os.makedirs(args.out, exist_ok=True)
    seed_everything(args.seed)

    # Collect files
    all_npz = sorted(glob.glob(os.path.join(args.data_dir, "**", "*.npz"), recursive=True))
    if not all_npz:
        print(f"❌ No NPZ files found in {args.data_dir}")
        return

    print(f"📂 Found {len(all_npz)} NPZ files")
    print("Loading dataset...")

    # Build dataset (default eager for speed)
    full_ds = NPZWindowDataset(
        files=all_npz, class_map={}, T=args.T,
        default_stride=args.default_stride,
        target_stride_s=(args.target_stride_s if args.target_stride_s > 0 else None),
        max_views=args.max_views,
        use_visibility_mask=args.use_visibility_mask,
        use_soft_labels=args.use_soft_labels,
        lazy_load=args.lazy_load
    )

    if not full_ds.discovered_labels:
        print("❌ No 'movement_type' found in NPZ files.")
        return

    classes = full_ds.discovered_labels
    class_map = {lab: i for i, lab in enumerate(classes)}
    full_ds.class_map = class_map

    # Sanity check
    try:
        x0, y0, s0, soft0 = full_ds[0]
        print(f"✓ Files: {len(full_ds.meta_per_file)}  |  Windows: {len(full_ds)}  |  Views: {full_ds.num_views}")
        print(f"✓ Sample window: {x0.shape}  |  Label: {'soft' if soft0 else int(y0)}  |  Subject: {s0}")
        assert x0.shape == (args.T, NUM_LANDMARKS, 3 * full_ds.num_views), f"Shape mismatch: {x0.shape}"
    except Exception as e:
        print(f"⚠️  Sanity check warning: {e}")

    # Dilations
    try:
        dilations = [int(x.strip()) for x in args.dilations.split(",") if x.strip()]
        if not dilations:
            dilations = [1, 2, 4, 8, 16, 32]
    except Exception:
        dilations = [1, 2, 4, 8, 16, 32]
        print(f"⚠️  Invalid dilations, using default: {dilations}")

    # Subject-group split
    file_subjects = [m['subject'] for m in full_ds.meta_per_file]
    unique_file_idxs = list(range(len(full_ds.meta_per_file)))

    print("\n🔀 Splitting dataset by subjects...")
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        y_file = [m['label'] for m in full_ds.meta_per_file]
        groups = file_subjects
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
        best = None
        best_diff = 1e9
        for tr_idx, va_idx in sgkf.split(unique_file_idxs, y=y_file, groups=groups):
            diff = abs(len(va_idx) / len(unique_file_idxs) - args.val_size)
            if diff < best_diff:
                best = (tr_idx, va_idx)
                best_diff = diff
        train_files_idx, val_files_idx = best
        print("✓ Using StratifiedGroupKFold")
    except Exception as e:
        print(f"⚠️  StratifiedGroupKFold failed ({e}), using GroupShuffleSplit")
        gss = GroupShuffleSplit(n_splits=1, test_size=args.val_size, random_state=args.seed)
        train_files_idx, val_files_idx = next(gss.split(unique_file_idxs, groups=file_subjects))

    print_subject_split_report(full_ds, train_files_idx, val_files_idx)

    # Map file indices to window indices
    file_to_windows = defaultdict(list)
    for win_idx, (meta_idx, _) in enumerate(full_ds.index):
        file_to_windows[meta_idx].append(win_idx)

    train_win_idxs, val_win_idxs = [], []
    for fi in train_files_idx:
        train_win_idxs.extend(file_to_windows[fi])
    for fi in val_files_idx:
        val_win_idxs.extend(file_to_windows[fi])

    # Purity analysis and filtering
    if args.analyze_temporal_artifacts or args.filter_window_purity > 0:
        _ = analyze_window_purity(full_ds)

    if args.filter_window_purity > 0:
        train_win_idxs = filter_windows_by_purity(full_ds, train_win_idxs, min_purity=args.filter_window_purity)

    # Build subsets
    train_ds = Subset(full_ds, train_win_idxs)
    val_ds = Subset(full_ds, val_win_idxs)

    if len(train_ds) == 0 or len(val_ds) == 0:
        print("❌ Error: train or val dataset is empty")
        return

    print(f"\n✓ Train windows: {len(train_ds):,}  |  Val windows: {len(val_ds):,}")
    print(f"✓ Views used: {full_ds.num_views}  → input_dim = {full_ds.input_dim}")
    print_class_distribution(full_ds, train_ds, val_ds, classes)

    # Sampler strategy
    print(f"\n🎯 Imbalance Strategy: {args.imbalance_strategy.upper()}")
    use_sampler = (args.imbalance_strategy == "sampler")
    use_weights = (args.imbalance_strategy == "weights")
    if use_sampler:
        subset_sampler = make_balanced_sampler_for_subset(full_ds, train_win_idxs)
        print("   ✓ Using balanced sampler (label × subject) from TRAIN subset only")
        shuffle_train = False
    else:
        subset_sampler = None
        shuffle_train = True
        print("   ✓ Using random sampling (shuffle=True)")

    # Collates
    collate_kwargs = dict(
        augment=args.aug_enable,
        time_mask_prob=args.aug_time_mask_prob, time_mask_max_frames=args.aug_time_mask_max_frames,
        joint_dropout_prob=args.aug_joint_dropout_prob, joint_dropout_frac=args.aug_joint_dropout_frac,
        noise_std=args.aug_noise_std,
        rotation_prob=args.aug_rotation_prob, rotation_angle_deg=args.aug_rotation_angle_deg,
        scale_prob=args.aug_scale_prob, scale_min=args.aug_scale_min, scale_max=args.aug_scale_max,
        temporal_warp_prob=args.aug_temporal_warp_prob,
        normal_class_name=args.normal_class_name,
        class_map=class_map,
        aug_normal_scale=args.aug_normal_scale,
        aug_normal_overrides=None
    )
    train_collate = CollateWindows(**collate_kwargs)
    val_collate = CollateWindows(augment=False, class_map=class_map, normal_class_name=args.normal_class_name)

    # Dataloaders (fast path)
    pin = True  # pinned memory helps H2D copies even with 0 workers
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        sampler=subset_sampler,
        shuffle=(shuffle_train if subset_sampler is None else False),
        num_workers=args.num_workers,
        pin_memory=pin,
        collate_fn=train_collate,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None),
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
        collate_fn=val_collate,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None),
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None
    )

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Model
    model = PoseTCNMultiView(
        num_classes=len(classes),
        num_views=full_ds.num_views,
        width=args.width,
        drop=args.dropout,
        stochastic_depth=args.stochastic_depth,
        dilations=dilations,
        fusion=args.fusion,
        view_fusion=args.view_fusion,
        hybrid_alpha=args.hybrid_alpha
    ).to(device)

    # Initialize head bias using train priors (skip if using class weights to avoid double bias)
    if not use_weights:
        try:
            class_counts = np.zeros(len(classes), dtype=np.int64)
            for win_idx in train_win_idxs:
                meta_idx, _ = full_ds.index[win_idx]
                lab = full_ds.meta_per_file[meta_idx]['label']
                class_counts[class_map[lab]] += 1
            priors = class_counts / np.clip(class_counts.sum(), 1, None)
            with torch.no_grad():
                # Both early and late heads exist; set both
                model.head_early.bias.copy_(torch.log(torch.tensor(priors + 1e-8, device=model.head_early.bias.device)))
                model.head_late.bias.copy_(torch.log(torch.tensor(priors + 1e-8, device=model.head_late.bias.device)))
            print("🧭 Initialized classifier biases with train priors (log-probabilities):")
            for c, p in zip(classes, priors):
                print(f"   {c:>18s}: p={p:.4f}")
        except Exception as e:
            print(f"⚠️ Could not initialize head bias to priors: {e}")
    else:
        print("🧭 Skipping bias initialization (using class weights).")

    # Params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parameters: {trainable_params:,} / {total_params:,} trainable")

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.use_cosine_schedule:
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        warm = int(max(0, args.warmup_epochs))
        if warm > 0:
            rem = max(1, args.epochs - warm)
            warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warm)
            cosine_scheduler = CosineAnnealingLR(optimizer, T_max=rem, eta_min=1e-6)
            scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warm])
            print(f"📈 Scheduler: Cosine Annealing with {warm} epoch warmup")
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
            print("📈 Scheduler: Cosine Annealing (no warmup)")
    else:
        try:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
        except TypeError:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
        print("📈 Scheduler: ReduceLROnPlateau")

    # AMP scaler
    use_cuda = (device.type == 'cuda')
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    try:
        scaler = torch.amp.GradScaler(enabled=use_cuda and not use_bf16, device='cuda')
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=use_cuda and not use_bf16)
    print(f"   Mixed Precision: {'BF16' if use_bf16 else 'FP16' if use_cuda else 'Disabled'}")

    # Losses
    if args.use_focal_loss:
        if use_weights:
            cls_w = class_weights_from_train_subset(
                full_ds, train_ds, class_map,
                pow_smooth=args.weight_smooth_power,
                clip=(args.weight_clip_min, args.weight_clip_max)
            ).to(device)
            print(f"🎯 Loss: Focal (γ={args.focal_gamma}) + Smoothed Class Weights + Label Smoothing ({args.label_smoothing})")
            print(f"   Weight smoothing: power={args.weight_smooth_power}, clip=[{args.weight_clip_min}, {args.weight_clip_max}]")
            print(f"   Class weights: min={cls_w.min().item():.2f}x, max={cls_w.max().item():.2f}x, mean={cls_w.mean().item():.2f}x")
            criterion = FocalLoss(alpha=cls_w, gamma=args.focal_gamma, label_smoothing=args.label_smoothing)
            criterion_eval = nn.CrossEntropyLoss(weight=cls_w)  # clean eval CE (no smoothing)
        else:
            print(f"🎯 Loss: Focal (γ={args.focal_gamma}) + Label Smoothing ({args.label_smoothing})")
            criterion = FocalLoss(gamma=args.focal_gamma, label_smoothing=args.label_smoothing)
            criterion_eval = nn.CrossEntropyLoss()
    else:
        if use_weights:
            cls_w = class_weights_from_train_subset(
                full_ds, train_ds, class_map,
                pow_smooth=args.weight_smooth_power,
                clip=(args.weight_clip_min, args.weight_clip_max)
            ).to(device)
            print(f"🎯 Loss: CrossEntropy + Smoothed Class Weights + Label Smoothing ({args.label_smoothing})")
            print(f"   Weight smoothing: power={args.weight_smooth_power}, clip=[{args.weight_clip_min}, {args.weight_clip_max}]")
            print(f"   Class weights: min={cls_w.min().item():.2f}x, max={cls_w.max().item():.2f}x, mean={cls_w.mean().item():.2f}x")
            criterion = nn.CrossEntropyLoss(weight=cls_w, label_smoothing=args.label_smoothing)
            criterion_eval = nn.CrossEntropyLoss(weight=cls_w)  # for metrics only; keep without smoothing to be "cleaner"
        else:
            print(f"🎯 Loss: CrossEntropy + Label Smoothing ({args.label_smoothing})")
            criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
            criterion_eval = nn.CrossEntropyLoss()

    # Training loop
    best_macro_f1 = -1.0
    best_epoch = 0
    best_path = os.path.join(args.out, f"best_{args.ckpt_prefix}.pt")
    last_path = os.path.join(args.out, f"last_{args.ckpt_prefix}.pt")
    patience = max(0, args.early_stop_patience)
    no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_balanced_acc': [], 'val_macro_f1': [], 'lr': []}

    print(f"\n{'='*80}")
    print(f"🚀 Training Configuration")
    print(f"{'='*80}")
    print(f"Epochs: {args.epochs}  |  Batch: {args.batch}  |  Accum: {args.accumulation_steps}  |  LR: {args.lr}")
    print(f"Width: {args.width}  |  Dilations: {dilations}")
    print(f"Dropout: {args.dropout}  |  Stochastic Depth: {args.stochastic_depth}")
    print(f"Fusion: {args.fusion}  |  View fusion: {args.view_fusion}  |  Hybrid α: {args.hybrid_alpha}")
    print(f"Weight Decay: {args.weight_decay}  |  Grad Clip: {args.grad_clip if args.grad_clip > 0 else 'None'}")
    print(f"Imbalance: {args.imbalance_strategy}  |  Focal γ: {args.focal_gamma if args.use_focal_loss else 'N/A'}")
    print(f"Soft Labels: {'ON' if args.use_soft_labels else 'OFF'}  |  Lazy Load: {'ON' if args.lazy_load else 'OFF'}")
    if args.mixup_alpha > 0:
        print(f"Mixup: α={args.mixup_alpha}")
    if args.aug_enable:
        print(f"Augmentation: Enabled (normal scale={args.aug_normal_scale})")
    print(f"Early Stopping: {patience} epochs")
    if args.calibrate_temperature:
        print(f"Post-hoc: Temperature scaling enabled")
    print(f"{'='*80}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device,
                                     accumulation_steps=args.accumulation_steps, scaler=scaler,
                                     criterion=criterion, mixup_alpha=args.mixup_alpha,
                                     grad_clip=args.grad_clip)
        val_metrics = evaluate(model, val_loader, device, criterion_eval=criterion_eval,
                               return_preds=(args.report_each > 0 and (epoch % args.report_each == 0)))
        dt = time.time() - t0

        macro_f1 = val_metrics['macro_f1']
        balanced_acc = val_metrics['balanced_acc']
        raw_acc = val_metrics['acc']
        val_loss = val_metrics.get('val_loss', float('nan'))
        current_lr = get_lr(optimizer)

        print(f"[{epoch:03d}/{args.epochs}] "
              f"loss: {train_loss:.4f} → {val_loss:.4f} | "
              f"acc: {raw_acc:.3f} | bal_acc: {balanced_acc:.3f} | "
              f"f1: {macro_f1:.3f} | lr: {current_lr:.2e} | {dt:.1f}s")

        if 'y' in val_metrics and 'p' in val_metrics:
            try:
                print(classification_report(val_metrics['y'], val_metrics['p'],
                                            target_names=classes, digits=3, zero_division=0))
            except Exception as e:
                print(f"  ⚠️  Could not generate classification report: {e}")

        # Save last checkpoint
        torch.save({
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'classes': classes, 'best_macro_f1': best_macro_f1,
            'args': vars(args)
        }, last_path)

        improved = macro_f1 > best_macro_f1 + 1e-4
        if improved:
            best_macro_f1 = macro_f1
            best_epoch = epoch
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'classes': classes, 'best_macro_f1': best_macro_f1,
                'args': vars(args)
            }, best_path)
            print(f"  ✨ New best! F1: {macro_f1:.4f}, Bal Acc: {balanced_acc:.4f}")
            no_improve = 0
        else:
            no_improve += 1
            if patience > 0:
                print(f"  ⏳ No improvement for {no_improve}/{patience} epochs")

        # Scheduler step
        if args.use_cosine_schedule:
            try:
                scheduler.step()
            except Exception:
                pass
        else:
            scheduler.step(macro_f1)

        # History
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(raw_acc)
        history['val_balanced_acc'].append(balanced_acc)
        history['val_macro_f1'].append(macro_f1)
        history['lr'].append(current_lr)

        if patience > 0 and no_improve >= patience:
            print(f"\n⏹️  Early stopping triggered ({no_improve} epochs without improvement)")
            break

        if args.plot_curves and epoch % 5 == 0:
            plot_training_curves(history, args.out)

    print(f"\n{'='*80}")
    print(f"✅ Training Complete!")
    print(f"{'='*80}")
    print(f"Best Macro F1: {best_macro_f1:.4f} (epoch {best_epoch})")
    print(f"Checkpoint: {best_path}")
    print(f"{'='*80}\n")

    # Post-hoc Temperature calibration
    if args.calibrate_temperature and os.path.isfile(best_path):
        print(f"\n{'='*80}")
        print(f"🌡️  Post-hoc Temperature Calibration")
        print(f"{'='*80}")
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print("Finding optimal temperature on validation set...")
        best_T, best_nll = find_best_temperature(model, val_loader, device)
        print(f"✓ Best temperature: {best_T:.3f} (NLL: {best_nll:.4f})")

        print("\nEvaluating with calibrated predictions...")
        cal_metrics = evaluate(model, val_loader, device, criterion_eval=criterion_eval,
                               return_preds=True, temperature=best_T)
        print(f"Calibrated Results:")
        print(f"  Accuracy: {cal_metrics['acc']:.4f}")
        print(f"  Balanced Acc: {cal_metrics['balanced_acc']:.4f}")
        print(f"  Macro F1: {cal_metrics['macro_f1']:.4f}")

        if 'y' in cal_metrics and 'p' in cal_metrics:
            print("\nCalibrated Classification Report:")
            try:
                print(classification_report(cal_metrics['y'], cal_metrics['p'],
                                            target_names=classes, digits=3, zero_division=0))
            except Exception as e:
                print(f"  ⚠️  Could not generate classification report: {e}")

        # Save temperature into checkpoint
        ckpt['best_temperature'] = best_T
        torch.save(ckpt, best_path)
        print(f"\n✓ Saved temperature={best_T:.3f} to checkpoint")
        if 'logits' in cal_metrics:
            cal_results_path = os.path.join(args.out, f"calibrated_results_{args.ckpt_prefix}.npz")
            np.savez_compressed(cal_results_path,
                                logits=cal_metrics['logits'],
                                labels=cal_metrics['y'],
                                predictions=cal_metrics['p'],
                                temperature=best_T,
                                classes=np.array(classes, dtype=object))
            print(f"✓ Saved calibrated results to {cal_results_path}")

    if args.plot_curves:
        plot_training_curves(history, args.out)

    if args.save_history:
        history_path = os.path.join(args.out, f"history_{args.ckpt_prefix}.json")
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"📊 Training history saved to {history_path}")

if __name__ == "__main__":
    main()
