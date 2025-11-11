import numpy as np

def _normalize(X):
    X = np.asarray(X, float)
    mn, mx = X.min(axis=0), X.max(axis=0)
    denom = np.where(mx - mn == 0, 1, mx - mn)
    return (X - mn) / denom

def sample_dpp(rows, k, *, seed=None, feature_fn=lambda r: r, sigma=1.0, **kwargs):
    rng = np.random.RandomState(seed)
    X = _normalize(np.vstack([feature_fn(r) for r in rows]))
    n = len(rows)

    # compute RBF kernel similarity matrix
    sq_dists = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
    K = np.exp(-sq_dists / (2 * sigma ** 2))

    # eigen decomposition of kernel matrix
    eigvals, eigvecs = np.linalg.eigh(K)
    eigvals = np.clip(eigvals, 0, 1)

    # sample eigenvectors based on eigenvalues
    selected_eigs = []
    for i in range(len(eigvals)):
        if rng.rand() < eigvals[i]:
            selected_eigs.append(i)

    if not selected_eigs:
        selected_eigs = [int(np.argmax(eigvals))]

    V = eigvecs[:, selected_eigs]
    Y = []
    for _ in range(min(k, len(selected_eigs))):
        # compute selection probabilities for each item
        probs = np.sum(V ** 2, axis=1)
        probs = probs / probs.sum()
        i = rng.choice(n, p=probs)
        Y.append(i)

        # update orthogonal basis
        vi = V[i, :].copy()
        if np.allclose(vi, 0):
            break
        vi = vi / np.linalg.norm(vi)
        V = V - np.outer(V @ vi, vi)
        # re-orthogonalize to maintain numerical stability
        if V.shape[1] > 1:
            V, _ = np.linalg.qr(V)

    # if not enough selected, fill up with randoms
    if len(Y) < k:
        remain = [i for i in range(n) if i not in Y]
        Y += list(rng.choice(remain, size=k - len(Y), replace=False))

    return [rows[i] for i in Y[:k]]
