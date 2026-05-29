"""
MMD utilities — Gaussian-kernel MMD² estimators with median-heuristic
bandwidth, used by reptile_mmd_signal and reptile_mmd_window for
cross-participant similarity-matrix construction.

Convention: we return -MMD² as the "similarity" so that higher = more
similar (matching the anchor-high/low convention used by the MI sampler
in reptile_mi.sample_mi_guided_episode).
"""
import numpy as np


def gaussian_kernel(X, Y, gamma):
    """
    Gaussian (RBF) kernel matrix between rows of X (n, d) and Y (m, d).
    Returns kernel matrix of shape (n, m).
    """
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    XX = np.sum(X * X, axis=1, keepdims=True)       # (n, 1)
    YY = np.sum(Y * Y, axis=1, keepdims=True).T     # (1, m)
    XY = X @ Y.T                                     # (n, m)
    sq_dist = XX + YY - 2.0 * XY
    sq_dist = np.maximum(sq_dist, 0.0)
    return np.exp(-gamma * sq_dist)


def median_heuristic_bandwidth(X, Y, max_samples=2000, rng=None):
    """
    Median heuristic for RBF kernel bandwidth.
    gamma = 1 / (2 * sigma^2) where sigma^2 is the median pairwise
    squared Euclidean distance over a subsample of the pooled X ∪ Y.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    pool = np.concatenate([np.asarray(X), np.asarray(Y)], axis=0)
    if len(pool) > max_samples:
        idx = rng.choice(len(pool), max_samples, replace=False)
        pool = pool[idx]
    if len(pool) < 2:
        return 1.0
    PP = np.sum(pool * pool, axis=1, keepdims=True)
    sq_dist = PP + PP.T - 2.0 * (pool @ pool.T)
    sq_dist = np.maximum(sq_dist, 0.0)
    iu = np.triu_indices(len(pool), k=1)
    median_sq = float(np.median(sq_dist[iu]))
    sigma_sq = max(median_sq / 2.0, 1e-12)
    return 1.0 / (2.0 * sigma_sq)


def mmd2_biased(X, Y, gamma):
    """
    Biased MMD² with Gaussian kernel:
        MMD²(X, Y) = mean(K(X, X)) + mean(K(Y, Y)) - 2 * mean(K(X, Y)).
    """
    Kxx = gaussian_kernel(X, X, gamma).mean()
    Kyy = gaussian_kernel(Y, Y, gamma).mean()
    Kxy = gaussian_kernel(X, Y, gamma).mean()
    return float(Kxx + Kyy - 2.0 * Kxy)


def mmd2_unbiased(X, Y, gamma):
    """
    Unbiased MMD² with Gaussian kernel: diagonal terms excluded for
    K(X, X) and K(Y, Y).
    """
    m, n = len(X), len(Y)
    if m < 2 or n < 2:
        return mmd2_biased(X, Y, gamma)
    Kxx = gaussian_kernel(X, X, gamma)
    Kyy = gaussian_kernel(Y, Y, gamma)
    Kxy = gaussian_kernel(X, Y, gamma)
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    return float(Kxx.sum() / (m * (m - 1))
                 + Kyy.sum() / (n * (n - 1))
                 - 2.0 * Kxy.mean())


def mmd_similarity(X, Y, rng=None, max_samples_bandwidth=2000, unbiased=False):
    """
    Convenience: returns -MMD²(X, Y) so that higher = more similar.
    Uses median-heuristic bandwidth estimated from X ∪ Y.
    """
    gamma = median_heuristic_bandwidth(
        X, Y, max_samples=max_samples_bandwidth, rng=rng)
    if unbiased:
        return -mmd2_unbiased(X, Y, gamma)
    return -mmd2_biased(X, Y, gamma)
