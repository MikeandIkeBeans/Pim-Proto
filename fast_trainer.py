#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PIM Trainer — FAST Anti-Collapse (macro-F1 early stop, quick turnaround)

Highlights
- Robust mixed loader for .npy/.npz:
  • (T,33,3|4), (T,99|132), (T,3|4,33), (33,T,3|4), object arrays of (33,3|4)
  • TABLE: rows [timestamp, landmark_id, x, y, z] (groups by timestamp; safe fallback)
- Torso-based normalization (hip-centered; multi-cue scale)
- BalancedSoftmax + Focal (training-window priors) => avoids single-class predictions
- Weight-norm head + dropout
- FAST loop: capped windows, capped steps/epoch, eval_every N epochs, early stop on macro-F1

Usage (fastest smoke test):
python pim_trainer_fast_mixed.py --data_dir /path/to/data --augment --fast
"""

import os, json, random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import f1_score, balanced_accuracy_score

# ------------------------- constants -------------------------
PIM_CLASSES = [
    "normal","decorticate","dystonia","chorea","myoclonus",
    "decerebrate","fencer posture","ballistic","tremor","versive head"
]
NUM_LANDMARKS = 33
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 11, 12, 23, 24

# ------------------------- utils -------------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

def parse_label_subject_from_filename(path: str):
    stem = Path(path).stem.lower()
    t = stem.split("_")
    if len(t) < 3:
        raise ValueError(f"Bad filename (need class_subject_class): {stem}")
    label = {"fencer":"fencer posture","versive":"versive head"}.get(t[0], t[0])
    subject = t[1]
    return label, subject

# ------------------------- robust loader -------------------------
def load_npy_sequence(path: str, num_landmarks: int = 33) -> np.ndarray:
    """
    Robust loader for pose -> (T,33,3) float32.
    
    FIXED for table format where timestamps are all identical
    """
    def _from_npz(npz):
        if "data" in npz: return npz["data"]
        if "arr_0" in npz: return npz["arr_0"]
        ks = list(npz.keys())
        if not ks: raise ValueError("Empty .npz archive.")
        return npz[ks[0]]

    def _ensure_float32(x): return x.astype(np.float32, copy=False)

    p = Path(path)
    arr = _from_npz(np.load(path, allow_pickle=True)) if p.suffix.lower() == ".npz" else np.load(path, allow_pickle=True)

    # -------- OBJECT ARRAY: per-frame matrices --------
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        frames = []
        for i, f in enumerate(arr):
            f = np.asarray(f)
            if f.ndim != 2:
                raise ValueError(f"{path}: frame {i} ndim={f.ndim}, expected 2.")
            if f.shape == (num_landmarks, 3) or f.shape == (num_landmarks, 4):
                frames.append(_ensure_float32(f[:, :3]))
            elif f.shape[1] == num_landmarks and f.shape[0] in (3, 4):
                frames.append(_ensure_float32(f.T[:, :3]))
            else:
                raise ValueError(f"{path}: frame {i} shape {f.shape} not compatible with (33,3|4).")
        return np.stack(frames, axis=0)

    # Must be numeric ndarray from here
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"{path}: unsupported type {type(arr)}")

    # -------- TABLE FORMAT: [timestamp, landmark_id, x, y, z] --------
    # THIS IS YOUR DATA FORMAT!
    if arr.ndim == 2 and arr.shape[1] == 5:
        # Check if this looks like table format
        lm_col = arr[:, 1]
        try:
            lm_int = lm_col.astype(np.int64, copy=False)
        except Exception:
            lm_int = np.round(lm_col).astype(np.int64, copy=False)
        
        # Check if landmark IDs are valid (0-32)
        within = (lm_int >= 0) & (lm_int < num_landmarks)
        
        if within.mean() > 0.9:  # 90%+ of rows have valid landmark IDs
            print(f"[DEBUG] Loading table format: {path}")
            
            ts = arr[:, 0]
            xyz = arr[:, 2:5].astype(np.float32)
            
            # Try timestamp-based grouping first
            uniq_ts = np.unique(ts)
            
            if uniq_ts.size >= 2:
                # Multiple timestamps - use timestamp grouping
                order = np.argsort(ts, kind="mergesort")
                ts_sorted = ts[order]
                lm_sorted = lm_int[order]
                xyz_sorted = xyz[order]
                
                boundaries = np.flatnonzero(np.diff(ts_sorted)) + 1
                starts = np.r_[0, boundaries]
                ends   = np.r_[boundaries, ts_sorted.size]
                
                T = starts.size
                seq = np.zeros((T, num_landmarks, 3), dtype=np.float32)
                
                for i in range(T):
                    sl = slice(starts[i], ends[i])
                    ids = lm_sorted[sl]
                    valid = (ids >= 0) & (ids < num_landmarks)
                    if valid.any():
                        seq[i, ids[valid]] = xyz_sorted[sl][valid]
                
                return seq
            
            else:
                # Single timestamp (YOUR CASE) - group by 33-row blocks
                print(f"  [INFO] Single timestamp detected, using block grouping")
                
                n_rows = lm_int.size
                T = n_rows // num_landmarks  # Number of frames
                
                if T == 0:
                    raise ValueError(f"{path}: insufficient rows ({n_rows}) for even one frame")
                
                seq = np.zeros((T, num_landmarks, 3), dtype=np.float32)
                
                # Process in blocks of 33
                for t in range(T):
                    start_row = t * num_landmarks
                    end_row = min(start_row + num_landmarks, n_rows)
                    
                    block_ids = lm_int[start_row:end_row]
                    block_xyz = xyz[start_row:end_row]
                    
                    # Place each landmark in correct position
                    valid = (block_ids >= 0) & (block_ids < num_landmarks)
                    if valid.any():
                        seq[t, block_ids[valid]] = block_xyz[valid]
                
                print(f"  [INFO] Loaded {T} frames from {n_rows} rows")
                return seq
        
        # If not table format, continue to other parsers...

    # -------- SHAPE-BASED FORMATS --------
    if arr.ndim == 3:
        T, A, B = arr.shape
        if A == num_landmarks and B in (3, 4):  # (T,33,3|4)
            return _ensure_float32(arr[:, :, :3])
        if B == num_landmarks and A in (3, 4):  # (T,3|4,33) -> transpose
            return _ensure_float32(np.transpose(arr, (0, 2, 1))[:, :, :3])
        if T == num_landmarks and B in (3, 4):  # (33,T,3|4) -> transpose
            return _ensure_float32(np.transpose(arr, (1, 0, 2))[:, :, :3])
        raise ValueError(f"{path}: unrecognized 3D shape {arr.shape}; expected (T,33,3|4) or variants.")

    if arr.ndim == 2:
        T, D = arr.shape
        if D in (num_landmarks * 3, num_landmarks * 4):  # (T,99|132)
            C = 3 if D == num_landmarks * 3 else 4
            return _ensure_float32(arr.reshape(T, num_landmarks, C)[:, :, :3])
        if T in (num_landmarks * 3, num_landmarks * 4):  # (99|132,T)
            C = 3 if T == num_landmarks * 3 else 4
            return _ensure_float32(arr.T.reshape(arr.shape[1], num_landmarks, C)[:, :, :3])

    raise ValueError(f"{path}: unsupported array layout (ndim={arr.ndim}, shape={arr.shape}).")

# ------------------------- normalization & aug -------------------------
def _pair_dist(frame: np.ndarray, a: int, b: int):
    va, vb = frame[a], frame[b]
    if not (np.any(va != 0) and np.any(vb != 0)): return None
    return float(np.linalg.norm(va - vb) + 1e-8)

def normalize_sequence(seq: np.ndarray) -> np.ndarray:
    """Hip-centered; scale by median of torso cues (shoulders/hips/shoulder-hip)."""
    T = seq.shape[0]
    hips_ok = np.any(seq[:, L_HIP] != 0, axis=1) & np.any(seq[:, R_HIP] != 0, axis=1)
    center = 0.5 * (seq[:, L_HIP] + seq[:, R_HIP]) if hips_ok.any() \
             else 0.5 * (seq[:, L_SHOULDER] + seq[:, R_SHOULDER])
    seq = seq - center[:, None, :]

    dists = []
    for t in range(T):
        frame = seq[t]
        ds = _pair_dist(frame, L_SHOULDER, R_SHOULDER)
        dh = _pair_dist(frame, L_HIP, R_HIP)
        dt = _pair_dist(frame, L_SHOULDER, L_HIP)
        cues = [d for d in (ds, dh, dt) if d is not None]
        dists.append(np.median(cues) if cues else np.nan)
    scale = np.nanmedian(np.array(dists, dtype=np.float32))
    if not np.isfinite(scale) or scale < 1e-4: scale = 1.0

    seq = np.clip((seq / scale).astype(np.float32), -5.0, 5.0)
    return seq

def add_noise(seq: np.ndarray, sigma: float = 0.01) -> np.ndarray:
    return seq + np.random.normal(0, sigma, seq.shape).astype(np.float32)

# ------------------------- dataset -------------------------
class PIMWindowDataset(Dataset):
    """Windows are fixed-length and flattened to (T, 99)."""
    def __init__(self, files, labels, window_size=30, stride=30,
                 normalize=True, augment=False, cache_sequences=True,
                 max_windows_per_file: int = None):
        self.files = files
        self.labels = labels
        self.wsize = window_size
        self.stride = stride
        self.normalize = normalize
        self.augment = augment
        self.cache = cache_sequences
        self.max_windows_per_file = max_windows_per_file
        self._cache: Dict[int, np.ndarray] = {}
        self._cached_norm = (cache_sequences and normalize)
        self.windows = []
        self._build()

    def _build(self):
        self.windows.clear()
        rng = random.Random(0)
        for i, p in enumerate(self.files):
            seq = load_npy_sequence(p)
            if self._cached_norm:
                seq = normalize_sequence(seq)
                if self.cache: self._cache[i] = seq
            T = seq.shape[0]
            starts = list(range(0, max(1, T - self.wsize + 1), self.stride)) or [0]
            if self.max_windows_per_file is not None and len(starts) > self.max_windows_per_file:
                starts = rng.sample(starts, self.max_windows_per_file)
            for st in starts:
                self.windows.append({"fi": i, "st": st})

    def __len__(self): return len(self.windows)
    def _get_seq(self, i: int) -> np.ndarray:
        if i in self._cache: return self._cache[i]
        seq = load_npy_sequence(self.files[i])
        if self.normalize: seq = normalize_sequence(seq)
        if self.cache: self._cache[i] = seq
        return seq

    def __getitem__(self, idx: int):
        meta = self.windows[idx]
        fi, st = meta["fi"], meta["st"]
        seq = self._get_seq(fi)
        end = st + self.wsize
        win = seq[st:end]
        if win.shape[0] < self.wsize:
            pad = self.wsize - win.shape[0]
            win = np.concatenate([win, np.repeat(win[-1:], pad, axis=0)], axis=0)
        if self.augment:
            win = add_noise(win, sigma=0.01)
        feat = win.reshape(self.wsize, -1).astype(np.float32)  # (T, 99)
        return feat, self.wsize, self.labels[fi], fi

# ------------------------- sampler & counts -------------------------
def make_window_sampler(ds: PIMWindowDataset, num_classes: int, samples_per_epoch: int = None):
    """WeightedRandomSampler across WINDOWS; optional cap for fast epochs."""
    win_labels = np.array([ds.labels[w["fi"]] for w in ds.windows])
    counts = np.bincount(win_labels, minlength=num_classes).clip(min=1)
    class_w = (counts.sum() / (num_classes * counts.astype(np.float32)))
    class_w = (class_w / class_w.mean()).clip(max=5.0)
    weights = class_w[win_labels]
    n_samples = int(samples_per_epoch) if samples_per_epoch is not None else len(weights)
    return WeightedRandomSampler(weights, num_samples=n_samples, replacement=True)

def window_class_counts(ds: PIMWindowDataset, num_classes: int) -> torch.Tensor:
    win_labels = [ds.labels[w["fi"]] for w in ds.windows]
    counts = np.bincount(win_labels, minlength=num_classes)
    counts[counts == 0] = 1
    return torch.tensor(counts, dtype=torch.float32)

# ------------------------- model -------------------------
class AttentiveBiLSTM(nn.Module):
    def __init__(self, input_dim, num_classes, hidden=192, layers=2, drop=0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden, layers, bidirectional=True, batch_first=True,
            dropout=drop if layers > 1 else 0.0
        )
        attdim = hidden * 2
        self.attn = nn.Sequential(nn.Linear(attdim, attdim), nn.Tanh(), nn.Linear(attdim, 1))
        self.drop = nn.Dropout(drop)
        self.fc = nn.Sequential(
            nn.Linear(attdim, attdim // 2),
            nn.ReLU(True),
            nn.Dropout(drop),
            nn.utils.weight_norm(nn.Linear(attdim // 2, num_classes))
        )
    def forward(self, x, lengths):
        if torch.all(lengths == lengths[0]):
            out, _ = self.lstm(x)
        else:
            pack = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.lstm(pack)
            out, _ = pad_packed_sequence(out, batch_first=True)
        s = self.attn(out).squeeze(-1)
        mask = torch.arange(out.size(1), device=out.device)[None, :] < lengths[:, None]
        s = s.masked_fill(~mask, torch.finfo(s.dtype).min)
        w = torch.softmax(s, dim=1)
        ctx = (out * w[:, :, None]).sum(dim=1)
        return self.fc(self.drop(ctx))

# ------------------------- loss -------------------------
class BalancedSoftmaxWithFocal(nn.Module):
    def __init__(self, cls_counts: torch.Tensor, gamma=1.5, label_smoothing=0.05):
        super().__init__()
        prior = (cls_counts / cls_counts.sum()).clamp_min(1e-8)
        self.register_buffer("log_prior", torch.log(prior))
        self.gamma = gamma
        self.smooth = label_smoothing
    def forward(self, logits, target):
        la_logits = logits + self.log_prior
        logp = torch.log_softmax(la_logits, dim=1)
        p = torch.softmax(la_logits, dim=1)
        n_cls = logits.size(1)
        with torch.no_grad():
            t = torch.zeros_like(logits).scatter_(1, target.view(-1,1), 1.0)
            if self.smooth > 0:
                t = (1 - self.smooth) * t + self.smooth / n_cls
        ce = -(t * logp).sum(dim=1)
        pt = (t * p).sum(dim=1).clamp_min(1e-8)
        fl = ((1 - pt) ** self.gamma) * ce
        return fl.mean()

# ------------------------- collate -------------------------
def collate_batch(batch):
    xs = np.stack([b[0] for b in batch], axis=0).astype(np.float32)
    L  = np.array([b[1] for b in batch], dtype=np.int64)
    y  = np.array([b[2] for b in batch], dtype=np.int64)
    fi = np.array([b[3] for b in batch], dtype=np.int64)
    return torch.from_numpy(xs), torch.from_numpy(L), torch.from_numpy(y), torch.from_numpy(fi)

# ------------------------- train / eval -------------------------
def train_epoch(model, loader, crit, opt, dev, grad_clip=1.0):
    model.train(); total = 0.0
    scaler = torch.cuda.amp.GradScaler(enabled=(dev.type == "cuda"))
    opt.zero_grad(set_to_none=True)
    for x, L, y, _ in loader:
        x, L, y = x.to(dev, non_blocking=True), L.to(dev), y.to(dev)
        with torch.cuda.amp.autocast(enabled=(dev.type == "cuda")):
            logits = model(x, L)
            loss = crit(logits, y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        total += float(loss.item())
    return total / max(1, len(loader))

@torch.no_grad()
def evaluate(model, loader, crit, dev, num_classes: int, class_names: List[str],
             log_prior: torch.Tensor = None):
    model.eval(); total = 0.0; ypred = []; ytrue = []
    for x, L, y, _ in loader:
        x, L, y = x.to(dev, non_blocking=True), L.to(dev), y.to(dev)
        logits = model(x, L)
        if log_prior is not None:
            logits = logits + log_prior
            loss = nn.CrossEntropyLoss()(logits, y)
        else:
            loss = crit(logits, y)
        total += float(loss.item())
        ypred.append(torch.argmax(logits, dim=1).cpu().numpy()); ytrue.append(y.cpu().numpy())
    yp = np.concatenate(ypred) if ypred else np.array([])
    yt = np.concatenate(ytrue) if ytrue else np.array([])
    acc = float((yp == yt).mean()) if yt.size else 0.0

    hist = np.bincount(yp, minlength=num_classes) if yp.size else np.zeros(num_classes, int)
    print("Predicted windows per class:")
    for i, name in enumerate(class_names):
        print(f"  {name:20s}: {int(hist[i])}")

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(yt, yp): cm[t, p] += 1

    macro_f1 = f1_score(yt, yp, average='macro') if yt.size else 0.0
    bal_acc = balanced_accuracy_score(yt, yp) if yt.size else 0.0
    return {"loss": total / max(1, len(loader)), "acc": acc, "cm": cm,
            "macro_f1": float(macro_f1), "balanced_acc": float(bal_acc)}

# ------------------------- config -------------------------
@dataclass
class TrainConfig:
    data_dir: str
    output_dir: str = "./pim_runs"
    window_size: int = 30
    train_stride: int = 10        # faster than 5; override with --fast preset if desired
    eval_stride: int = 30
    normalize: bool = True
    augment: bool = True
    batch_size: int = 192
    epochs: int = 20
    lr: float = 2e-4
    weight_decay: float = 3e-4
    grad_clip: float = 1.0
    val_split: float = 0.15
    test_split: float = 0.15
    hidden_size: int = 160
    num_layers: int = 2
    dropout: float = 0.5
    num_workers: int = 8
    seed: int = 42

    # Turnaround controls:
    max_windows_per_file: int = 160         # train cap
    eval_max_windows_per_file: int = 60     # val/test cap
    train_samples_per_epoch: int = 6000     # sampler length (batch_size * steps)
    eval_every: int = 2                     # run validation every N epochs
    patience: int = 3                       # macro-F1 early stop patience

    # Other toggles:
    use_logit_adjust_eval: bool = True      # add priors to logits at eval    # add priors to logits at eval
    compile_model: bool = False             # torch.compile optional
        
        # NEW: Fast iteration controls
    subset_fraction: float = None           # Use subset of data for quick tests
    skip_early_eval: bool = False           # Skip validation on first 2 epochs
# ------------------------- split -------------------------
def stratified_group_split_smart(files, labels_idx, groups, num_classes,
                                 val_ratio=0.15, test_ratio=0.15, seed=42, ensure_coverage=True):
    rng = random.Random(seed)
    g2i = defaultdict(list)
    for i, g in enumerate(groups): g2i[g].append(i)
    ghist = {g: Counter(labels_idx[i] for i in idxs) for g, idxs in g2i.items()}
    gcnt = Counter(labels_idx)
    tval  = {c: int(round(gcnt[c] * val_ratio))  for c in range(num_classes)}
    ttest = {c: int(round(gcnt[c] * test_ratio)) for c in range(num_classes)}
    if ensure_coverage:
        for c in range(num_classes):
            if gcnt[c] >= 3:
                tval[c]  = max(1, tval[c]); ttest[c] = max(1, ttest[c])
    gl = list(g2i.keys()); rng.shuffle(gl)
    split = {}; cval = Counter(); ctest = Counter()
    def fit(h, cur, tgt): return sum(min(h.get(c,0), max(0, tgt.get(c,0) - cur.get(c,0))) for c in range(num_classes))
    for g in gl:
        h = ghist[g]; sv = fit(h, cval, tval); st = fit(h, ctest, ttest)
        if st >= sv and st > 0: split[g] = "test"; ctest.update(h)
        elif sv > 0:           split[g] = "val";  cval.update(h)
        else:                  split[g] = "train"
    if ensure_coverage:
        def ensure(name, cur):
            miss = [c for c in range(num_classes) if gcnt[c] > 0 and cur.get(c,0) == 0]
            for c in miss:
                cand = next((g for g in gl if split[g] == "train" and ghist[g].get(c,0) > 0), None)
                if cand: split[cand] = name; cur.update(ghist[cand])
        ensure("val", cval); ensure("test", ctest)
    tr, va, te = [], [], []
    for g, idxs in g2i.items():
        s = split[g]
        (tr if s=="train" else va if s=="val" else te).extend(idxs)
    rng.shuffle(tr); rng.shuffle(va); rng.shuffle(te)
    return tr, va, te

def select_subset_indices(files, labels_idx, subjects, num_classes, 
                         subset_fraction, seed=42):
    """Select a subset of data while ensuring all classes are represented."""
    if subset_fraction is None or subset_fraction >= 1.0:
        return list(range(len(files)))
    
    rng = random.Random(seed)
    indices = list(range(len(files)))
    
    # Group by class
    class_indices = defaultdict(list)
    for i in indices:
        class_indices[labels_idx[i]].append(i)
    
    # Select from each class
    selected = []
    for c in range(num_classes):
        class_samples = class_indices[c]
        n_take = max(3, int(len(class_samples) * subset_fraction))  # At least 3 per class
        rng.shuffle(class_samples)
        selected.extend(class_samples[:n_take])
    
    rng.shuffle(selected)
    print(f"📊 Using subset: {len(selected)}/{len(files)} files ({subset_fraction*100:.0f}%)")
    return selected
# ------------------------- main -------------------------
def train(cfg: TrainConfig):
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Discover both .npy and .npz
    f_npy = sorted(Path(cfg.data_dir).glob("*.npy"))
    f_npz = sorted(Path(cfg.data_dir).glob("*.npz"))
    files_all = f_npy + f_npz

    labels_str, subjects, files = [], [], []
    for f in files_all:
        try:
            lab, sub = parse_label_subject_from_filename(str(f))
        except Exception:
            # Skip files that don’t follow naming convention
            continue
        if lab in PIM_CLASSES:
            labels_str.append(lab); subjects.append(sub); files.append(str(f))

    active = sorted(set(labels_str))
    label_to_idx = {c: i for i, c in enumerate(active)}
    labels_idx = [label_to_idx[s] for s in labels_str]
    num_classes = len(active)
    print("Active classes:", active)

    # NEW: Optional subset selection for faster experimentation
    if hasattr(cfg, 'subset_fraction') and cfg.subset_fraction is not None:
        subset_indices = select_subset_indices(
            files, labels_idx, subjects, num_classes, 
            cfg.subset_fraction, seed=cfg.seed
        )
        files = [files[i] for i in subset_indices]
        labels_idx = [labels_idx[i] for i in subset_indices]
        subjects = [subjects[i] for i in subset_indices]

    # Split
    tr_idx, va_idx, te_idx = stratified_group_split_smart(
        files, labels_idx, subjects, num_classes,
        val_ratio=cfg.val_split, test_ratio=cfg.test_split, seed=cfg.seed, ensure_coverage=True
    )
    def report(name, idxs):
        cnt = Counter(labels_idx[i] for i in idxs)
        print(f"[{name}] files per class:", {active[c]: cnt.get(c, 0) for c in range(num_classes)})
    report("TRAIN", tr_idx); report("VAL", va_idx); report("TEST", te_idx)

    # Datasets (caps for fast eval)
    ds_tr = PIMWindowDataset([files[i] for i in tr_idx], [labels_idx[i] for i in tr_idx],
                             window_size=cfg.window_size, stride=cfg.train_stride,
                             normalize=cfg.normalize, augment=cfg.augment, cache_sequences=True,
                             max_windows_per_file=cfg.max_windows_per_file)
    ds_va = PIMWindowDataset([files[i] for i in va_idx], [labels_idx[i] for i in va_idx],
                             window_size=cfg.window_size, stride=cfg.eval_stride,
                             normalize=cfg.normalize, augment=False, cache_sequences=True,
                             max_windows_per_file=cfg.eval_max_windows_per_file)
    ds_te = PIMWindowDataset([files[i] for i in te_idx], [labels_idx[i] for i in te_idx],
                             window_size=cfg.window_size, stride=cfg.eval_stride,
                             normalize=cfg.normalize, augment=False, cache_sequences=True,
                             max_windows_per_file=cfg.eval_max_windows_per_file)

    if len(ds_tr) > 0:
        x0, _, y0, _ = ds_tr[0]
        print("First-window stats mean/std/min/max:",
              float(x0.mean()), float(x0.std()), float(x0.min()), float(x0.max()))
        print("First-window label:", active[y0])

    # Loaders
    tr_sampler = make_window_sampler(ds_tr, num_classes, samples_per_epoch=cfg.train_samples_per_epoch)
    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, sampler=tr_sampler,
                       num_workers=cfg.num_workers, pin_memory=True,
                       persistent_workers=(cfg.num_workers > 0),
                       prefetch_factor=4 if cfg.num_workers > 0 else None,
                       collate_fn=collate_batch, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False,
                       num_workers=cfg.num_workers, pin_memory=True,
                       persistent_workers=(cfg.num_workers > 0),
                       prefetch_factor=4 if cfg.num_workers > 0 else None,
                       collate_fn=collate_batch)
    dl_te = DataLoader(ds_te, batch_size=cfg.batch_size, shuffle=False,
                       num_workers=cfg.num_workers, pin_memory=True,
                       persistent_workers=(cfg.num_workers > 0),
                       prefetch_factor=4 if cfg.num_workers > 0 else None,
                       collate_fn=collate_batch)

    # Device / Model / Loss / Optim
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentiveBiLSTM(NUM_LANDMARKS * 3, num_classes,
                            hidden=cfg.hidden_size, layers=cfg.num_layers, drop=cfg.dropout).to(dev)

    if hasattr(torch, "compile") and cfg.compile_model:
        try:
            model = torch.compile(model)
            print("Compiled model with torch.compile")
        except Exception as e:
            print("torch.compile failed; continuing without.", e)

    cls_counts = window_class_counts(ds_tr, num_classes).to(dev)
    print("\nTrain-window counts:", {active[i]: int(cls_counts[i].item()) for i in range(num_classes)})

    crit = BalancedSoftmaxWithFocal(cls_counts, gamma=1.5, label_smoothing=0.05)
    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    log_prior = torch.log((cls_counts / cls_counts.sum()).clamp_min(1e-8)).to(dev) if cfg.use_logit_adjust_eval else None

    # FAST early stopping on macro-F1
    best_mf1 = -1.0; best_path = os.path.join(cfg.output_dir, "best.pt")
    epochs_no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.epochs}")
        tl = train_epoch(model, dl_tr, crit, opt, dev, grad_clip=cfg.grad_clip)

        skip_early = hasattr(cfg, 'skip_early_eval') and cfg.skip_early_eval and epoch <= 2
        do_eval = (not skip_early) and ((epoch % cfg.eval_every == 0) or (epoch == cfg.epochs))
        if do_eval:
            va = evaluate(model, dl_va, crit, dev, num_classes, active, log_prior=log_prior)
            print(f"Train Loss: {tl:.4f} | Val Loss: {va['loss']:.4f} | "
                  f"Val Acc: {va['acc']:.4f} | Val mF1: {va['macro_f1']:.4f} | "
                  f"Val BalAcc: {va['balanced_acc']:.4f}")
            mf1 = va["macro_f1"]
            if mf1 > best_mf1:
                best_mf1 = mf1; epochs_no_improve = 0
                torch.save({"model": model.state_dict(), "active_classes": active}, best_path)
                print(f"  ✓ Saved best (mF1={mf1:.4f}) to {best_path}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= cfg.patience:
                    print(f"Early stopping (no mF1 improvement for {cfg.patience} evals).")
                    break

        sched.step()

    # Test best
    if os.path.exists(best_path):
        ck = torch.load(best_path, map_location=dev)
        model.load_state_dict(ck["model"])
        active = ck.get("active_classes", active)

    te = evaluate(model, dl_te, crit, dev, num_classes, active, log_prior=log_prior)
    print("\n=== TEST ===")
    print(f"Loss: {te['loss']:.4f} | Acc: {te['acc']:.4f} | Macro-F1: {te['macro_f1']:.4f} | "
          f"Balanced-Acc: {te['balanced_acc']:.4f}")
    print("Confusion matrix (rows=true, cols=pred):\n", te["cm"])

# ------------------------- CLI -------------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output_dir", default="./pim_runs")
    p.add_argument("--window_size", type=int, default=30)
    p.add_argument("--train_stride", type=int, default=10)
    p.add_argument("--eval_stride", type=int, default=30)
    p.add_argument("--normalize", action="store_true"); p.add_argument("--no-normalize", dest="normalize", action="store_false"); p.set_defaults(normalize=True)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--batch_size", type=int, default=192)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=3e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--val_split", type=float, default=0.15)
    p.add_argument("--test_split", type=float, default=0.15)
    p.add_argument("--hidden_size", type=int, default=160)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_windows_per_file", type=int, default=160)
    p.add_argument("--eval_max_windows_per_file", type=int, default=60)
    p.add_argument("--train_samples_per_epoch", type=int, default=6000)
    p.add_argument("--eval_every", type=int, default=2)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--use_logit_adjust_eval", action="store_true")
    p.add_argument("--compile_model", action="store_true")
    p.add_argument("--fast", action="store_true", help="Preset for faster iteration")
    # NEW: Ultra-fast presets
    p.add_argument("--ultra_fast", action="store_true", 
                   help="Ultra-fast preset (96-dim, 1 layer, 2k samples/epoch)")
    p.add_argument("--micro_fast", action="store_true",
                   help="Micro-fast preset for quick architecture tests (64-dim, 1k samples)")
    p.add_argument("--subset_fraction", type=float, default=None,
                   help="Use only this fraction of data (0.0-1.0) for quick experiments")
    p.add_argument("--skip_early_eval", action="store_true",
                   help="Skip validation on first 2 epochs to save time")
    args = p.parse_args()

    # MICRO_FAST preset - for super quick architecture tests (1-2 minutes)
    if args.micro_fast:
        print("🚀 Using MICRO_FAST preset for quick architecture testing")
        args.hidden_size = 64
        args.num_layers = 1
        args.batch_size = 256
        args.epochs = 5
        args.train_stride = 20
        args.max_windows_per_file = 30
        args.eval_max_windows_per_file = 10
        args.train_samples_per_epoch = 1000
        args.eval_every = 1
        args.patience = 2
        args.use_logit_adjust_eval = True
        args.num_workers = 2
        if args.subset_fraction is None:
            args.subset_fraction = 0.2  # Use only 20% of data by default

    # ULTRA_FAST preset - for hyperparameter tuning (3-5 minutes)
    elif args.ultra_fast:
        print("⚡ Using ULTRA_FAST preset for rapid experimentation")
        args.hidden_size = 96
        args.num_layers = 1
        args.batch_size = 256
        args.epochs = 8
        args.train_stride = 15
        args.max_windows_per_file = 60
        args.eval_max_windows_per_file = 20
        args.train_samples_per_epoch = 2000
        args.eval_every = 1
        args.patience = 2
        args.use_logit_adjust_eval = True
        args.num_workers = 4
        if args.subset_fraction is None:
            args.subset_fraction = 0.5  # Use 50% of data by default
    if args.fast:
        args.hidden_size = 128
        args.num_layers = 2
        args.batch_size = max(128, args.batch_size)
        args.epochs = min(args.epochs, 12)
        args.train_stride = 10
        args.max_windows_per_file = 120
        args.eval_max_windows_per_file = 40
        args.train_samples_per_epoch = 4000
        args.eval_every = 2
        args.patience = 3
        args.use_logit_adjust_eval = True

    # Remove preset flags that aren't TrainConfig fields
    args_dict = vars(args)
    preset_flags = ['fast', 'micro_fast', 'ultra_fast']
    config_dict = {k: v for k, v in args_dict.items() if k not in preset_flags}
    
    cfg = TrainConfig(**config_dict)
    print("=== CONFIG ==="); print(json.dumps(asdict(cfg), indent=2))
    train(cfg)