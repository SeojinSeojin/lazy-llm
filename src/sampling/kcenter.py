import numpy as np

def _normalize(X):
    X = np.asarray(X, float)
    mn, mx = X.min(axis=0), X.max(axis=0)
    denom = np.where(mx - mn == 0, 1, mx - mn)
    return (X - mn) / denom

def sample_kcenter(i, k):
    """Reorder i.rows so that first k elements are k-center samples."""
    X = _normalize(np.vstack(i.rows))
    n = len(i.rows)
    if n == 0 or k <= 0:
        return

    idxs = [np.random.randint(0, n)]
    for _ in range(1, min(k, n)):
        dists = np.min([np.linalg.norm(X - X[j], axis=1) for j in idxs], axis=0)
        next_idx = int(np.argmax(dists))
        if next_idx in idxs:
            break
        idxs.append(next_idx)

    remaining = [j for j in range(n) if j not in idxs]
    i.rows[:] = [i.rows[j] for j in idxs] + [i.rows[j] for j in remaining]
