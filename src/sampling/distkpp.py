import numpy as np

def _normalize(X):
    X = np.asarray(X, float)
    mn, mx = X.min(axis=0), X.max(axis=0)
    denom = np.where(mx - mn == 0, 1, mx - mn)
    return (X - mn) / denom

def sample_distkpp(rows, k, *, seed=None, feature_fn=lambda r: r, **kwargs):
    rng = np.random.RandomState(seed)
    X = _normalize(np.vstack([feature_fn(r) for r in rows]))
    n = len(rows)
    idxs = [rng.randint(0, n)]
    for _ in range(1, min(k, n)):
        # euclidian distance squared
        d2 = np.min([np.sum((X - X[i]) ** 2, axis=1) for i in idxs], axis=0)
        probs = d2 / (d2.sum() if d2.sum() > 0 else 1.0)
        nxt = rng.choice(n, p=probs)
        if nxt in idxs:
            nxt = int(np.argmax(d2))
        idxs.append(nxt)
    return [rows[i] for i in idxs]
