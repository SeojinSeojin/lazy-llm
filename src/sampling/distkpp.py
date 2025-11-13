import numpy as np

def _normalize(X):
    X = np.asarray(X, float)
    mn, mx = X.min(axis=0), X.max(axis=0)
    denom = np.where(mx - mn == 0, 1, mx - mn)
    return (X - mn) / denom

def sample_distkpp(i, k):
    """Reorder i.rows so that the first k are k-means++ style samples."""
    X = _normalize(np.vstack(i.rows))
    n = len(i.rows)
    if n == 0 or k <= 0:
        return

    idxs = [np.random.randint(0, n)]
    for _ in range(1, min(k, n)):
        d2 = np.min([np.sum((X - X[j]) ** 2, axis=1) for j in idxs], axis=0)
        probs = d2 / (d2.sum() if d2.sum() > 0 else 1.0)
        nxt = np.random.choice(n, p=probs)
        if nxt in idxs:
            break
        idxs.append(nxt)

    remaining = [j for j in range(n) if j not in idxs]
    i.rows[:] = [i.rows[j] for j in idxs] + [i.rows[j] for j in remaining]
