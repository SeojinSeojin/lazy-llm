import numpy as np

def _normalize(X):
    X = np.asarray(X, float)
    mn, mx = X.min(axis=0), X.max(axis=0)
    denom = np.where(mx - mn == 0, 1, mx - mn)
    return (X - mn) / denom

def sample_kcenter(rows, k, *, seed=None, feature_fn=lambda r: r, **kwargs):
    rng = np.random.RandomState(seed)
    X = _normalize(np.vstack([feature_fn(r) for r in rows]))
    n = len(rows)
    idxs = [rng.randint(0, n)]

    # Greedy selection: pick the point farthest from the current set
    for _ in range(1, min(k, n)):
        # compute min distance to any selected center
        dists = np.min(
            [np.linalg.norm(X - X[i], axis=1) for i in idxs],
            axis=0
        )
        next_idx = int(np.argmax(dists))
        idxs.append(next_idx)
    return [rows[i] for i in idxs]
