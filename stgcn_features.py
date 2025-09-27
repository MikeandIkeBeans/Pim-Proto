import numpy as np
import mediapipe as mp

LM = mp.solutions.pose.PoseLandmark
CENTER = LM.PELVIS.value if hasattr(LM, "PELVIS") else LM.LEFT_HIP.value
R_SHOULDER = LM.RIGHT_SHOULDER.value
L_SHOULDER = LM.LEFT_SHOULDER.value

def _to_TVC(x):
    # Accept [T,V,3], [V,3,T], [3,T,V]; unify to [T,V,3]
    x = np.asarray(x)
    if x.ndim != 3:
        raise ValueError(f"expect 3D array, got {x.shape}")
    T, V, C = x.shape if x.shape[2] == 3 else (x.shape[0], x.shape[1], x.shape[2])
    if x.shape == (T, V, 3):
        return x
    if x.shape[0] == 3:  # [3,T,V]
        return np.transpose(x, (1, 2, 0))
    if x.shape[2] == T:  # [V,3,T]
        return np.transpose(x, (2, 0, 1))
    raise ValueError(f"unhandled shape {x.shape}")

def _normalize(seq_tvc):
    # center & scale per-window
    # center on pelvis (or left_hip fallback)
    center = seq_tvc[:, CENTER, :].copy()  # [T,3]
    seq_tvc = seq_tvc - center[:, None, :]  # broadcast subtract

    # scale by shoulder width (avg over time for stability)
    shoulder = seq_tvc[:, R_SHOULDER, :] - seq_tvc[:, L_SHOULDER, :]
    scale = np.clip(np.linalg.norm(shoulder, axis=-1).mean(), 1e-6, None)
    seq_tvc /= scale
    return seq_tvc

def _bones_from_edges(seq_tvc, edges):
    # seq_tvc: [T,V,3]
    T, V, _ = seq_tvc.shape
    bones = np.zeros_like(seq_tvc)
    parent = {v: u for (u, v) in edges} | {u: v for (u, v) in edges}  # bidirectional
    # pick a single parent closer to center: here choose the one with smaller index if unsure
    for u, v in edges:
        bones[:, v, :] = seq_tvc[:, v, :] - seq_tvc[:, u, :]
        bones[:, u, :] = seq_tvc[:, u, :] - seq_tvc[:, v, :]  # symmetrical — ok for undirected graph
    return bones

def sequences_to_stgcn_batches(seq_list, edges):
    """
    seq_list: list of windows, each ~ [T, V=33, 3] or compatible
    returns:
      X_joints [N, 3, T, 33], X_bones [N, 3, T, 33]
    """
    joints, bones = [], []
    for seq in seq_list:
        tvc = _normalize(_to_TVC(seq))    # [T,V,3]
        bvc = _bones_from_edges(tvc, edges)
        # to [C,T,V]
        joints.append(np.transpose(tvc, (2, 0, 1)))
        bones.append(np.transpose(bvc, (2, 0, 1)))
    Xj = np.stack(joints, axis=0).astype(np.float32)
    Xb = np.stack(bones, axis=0).astype(np.float32)
    return Xj, Xb