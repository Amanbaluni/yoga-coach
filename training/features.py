"""Feature code shared by training. MUST match features.js exactly."""
import numpy as np

T = 16                                   # frames per window
KEY = [0] + list(range(11, 33))          # nose + body landmarks
TRI = [(23,25,27),(24,26,28),(11,13,15),(12,14,16),(13,11,23),(14,12,24),(11,23,25),(12,24,26)]
SWAP = list(range(33))
for a, b in [(1,4),(2,5),(3,6),(7,8),(9,10),(11,12),(13,14),(15,16),(17,18),(19,20),(21,22),(23,24),(25,26),(27,28),(29,30),(31,32)]:
    SWAP[a], SWAP[b] = b, a

def _angle(a, b, c):
    v1, v2 = a - b, c - b
    m = np.linalg.norm(v1) * np.linalg.norm(v2) or 1.0
    return np.degrees(np.arccos(np.clip(v1 @ v2 / m, -1, 1)))

def angles(P):                           # P: (33,2) pixel coords
    return np.array([_angle(P[a], P[b], P[c]) for a, b, c in TRI])

def torso(P):
    return np.linalg.norm((P[11] + P[12]) / 2 - (P[23] + P[24]) / 2) or 1.0

def feats(P):                            # -> 46 position values + 8 angles
    hc = (P[23] + P[24]) / 2
    pos = ((P[KEY] - hc) / torso(P)).ravel()
    return np.concatenate([pos, angles(P) / 180]).astype(np.float32)

def canon_angles(a):                     # left/right-agnostic ordering (bent knee first)
    return a.reshape(4, 2)[:, ::-1].ravel() if a[0] > a[1] else a

def augment(P, width, rng):
    Q = P[SWAP].copy() if rng.random() < 0.5 else P.copy()
    if Q is not P and np.array_equal(Q, P[SWAP]): Q[:, 0] = width - Q[:, 0]
    hc = (Q[23] + Q[24]) / 2
    t = np.radians(rng.uniform(-15, 15)); R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    Q = (Q - hc) @ R.T + hc
    return Q + rng.normal(0, 0.02 * torso(Q), Q.shape)

def window(P, rng):                      # simulate T frames of a held pose with tracking jitter
    s = torso(P)
    return np.stack([feats(P + rng.normal(0, 0.01 * s, P.shape)) for _ in range(T)])
