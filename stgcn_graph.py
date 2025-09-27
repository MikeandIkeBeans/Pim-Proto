import torch
import numpy as np
import mediapipe as mp

# Choose a center joint for partitioning (pelvis/hips midpoint)
# MediaPipe indices: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
# We'll use the "POSE_LANDMARKS" enum to get integer ids.
LM = mp.solutions.pose.PoseLandmark
CENTER = LM.PELVIS.value if hasattr(LM, "PELVIS") else LM.LEFT_HIP.value  # fallback if PELVIS not present

def mp_edges():
    """Return undirected edge list from MediaPipe POSE_CONNECTIONS as (u,v) with integer ids."""
    E = []
    for u, v in mp.solutions.pose.POSE_CONNECTIONS:
        E.append((int(u), int(v)))
    return E

def normalize_adjacency(A):
    """D^-1/2 (A+I) D^-1/2 normalization."""
    V = A.shape[-1]
    A_hat = A + np.eye(V, dtype=np.float32)
    D = np.sum(A_hat, axis=-1)
    D_inv_sqrt = np.power(D, -0.5, where=D>0).astype(np.float32)
    return (D_inv_sqrt[:, None] * A_hat) * D_inv_sqrt[None, :]

def build_partitions(V=33, center=CENTER):
    """ST-GCN 3-partition adjacency: self, centripetal, centrifugal."""
    E = mp_edges()
    # base adjacency
    A = np.zeros((V, V), dtype=np.float32)
    for u, v in E:
        A[u, v] = 1.0
        A[v, u] = 1.0

    # shortest-path distances to center
    # BFS
    from collections import deque, defaultdict
    g = defaultdict(list)
    for u, v in E:
        g[u].append(v); g[v].append(u)
    dist = np.full(V, np.inf, dtype=np.float32); dist[center] = 0
    q = deque([center])
    while q:
        x = q.popleft()
        for y in g[x]:
            if dist[y] == np.inf:
                dist[y] = dist[x] + 1
                q.append(y)

    # partitions
    A_self = np.eye(V, dtype=np.float32)

    A_centripetal = np.zeros((V, V), dtype=np.float32)
    A_centrifugal = np.zeros((V, V), dtype=np.float32)
    for u, v in E:
        if dist[u] < dist[v]:  # v farther from center
            A_centripetal[u, v] = 1.0  # edge directed toward center
            A_centrifugal[v, u] = 1.0
        elif dist[v] < dist[u]:
            A_centripetal[v, u] = 1.0
            A_centrifugal[u, v] = 1.0
        else:
            # same ring — split evenly
            A_centripetal[u, v] = A_centrifugal[u, v] = 0.5
            A_centripetal[v, u] = A_centrifugal[v, u] = 0.5

    # normalize each
    A0 = normalize_adjacency(A_self)
    A1 = normalize_adjacency(A_centripetal)
    A2 = normalize_adjacency(A_centrifugal)

    A_stack = np.stack([A0, A1, A2], axis=0)  # [K=3, V, V]
    return torch.from_numpy(A_stack)  # float32