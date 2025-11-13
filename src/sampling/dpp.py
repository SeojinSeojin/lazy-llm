import numpy as np

def _normalize(X):
    X = np.asarray(X, float)
    mn, mx = X.min(axis=0), X.max(axis=0)
    denom = np.where(mx - mn == 0, 1, mx - mn)
    return (X - mn) / denom

def sample_dpp(i, k):
    """Reorder i.rows using DPP-like diversity criterion (RBF kernel)."""
    X = _normalize(np.vstack(i.rows))
    n = len(i.rows)
    if n == 0 or k <= 0:
        return

    sigma = 1.0
    sq_dists = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
    K = np.exp(-sq_dists / (2 * sigma ** 2))
    eigvals, eigvecs = np.linalg.eigh(K)
    eigvals = np.clip(eigvals, 0, 1)

    selected = [j for j, val in enumerate(eigvals) if np.random.rand() < val]
    if not selected:
        selected = [int(np.argmax(eigvals))]

    V = eigvecs[:, selected]
    Y = []
    for _ in range(min(k, len(selected))):
        probs = np.sum(V ** 2, axis=1)
        probs /= probs.sum()
        j = np.random.choice(n, p=probs)
        Y.append(j)
        vi = V[j, :].copy()
        if np.allclose(vi, 0):
            break
        vi /= np.linalg.norm(vi)
        V = V - np.outer(V @ vi, vi)
        if V.shape[1] > 1:
            V, _ = np.linalg.qr(V)

    if len(Y) < k:
        remain = [j for j in range(n) if j not in Y]
        Y += list(np.random.choice(remain, size=k - len(Y), replace=False))

    remaining = [j for j in range(n) if j not in Y]
    i.rows[:] = [i.rows[j] for j in Y] + [i.rows[j] for j in remaining]
