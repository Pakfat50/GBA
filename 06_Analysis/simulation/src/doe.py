"""Deterministic full factorial and face-centred response-surface designs.

Coded variables are in [-1, 1]. No statistical significance is asserted from
replication of identical deterministic simulations. Effects are engineering
contrasts over the specified ranges, not stochastic variance estimates.
"""
from itertools import combinations, product
import numpy as np


def factorial(n=5):
    return np.asarray(list(product((-1., 1.), repeat=n)))


def face_centered(n=5):
    return np.vstack((factorial(n), np.zeros((1, n)), np.eye(n), -np.eye(n)))


def features(z, quadratic=True):
    z = np.atleast_2d(np.asarray(z, dtype=float))
    n = z.shape[1]
    columns = [np.ones(len(z))] + [z[:, i] for i in range(n)]
    labels = ['intercept'] + [str(i) for i in range(n)]
    for i, j in combinations(range(n), 2):
        columns.append(z[:, i] * z[:, j])
        labels.append(f'{i}:{j}')
    if quadratic:
        columns.extend(z[:, i] ** 2 for i in range(n))
        labels.extend(f'{i}^2' for i in range(n))
    return np.column_stack(columns), labels


def contrasts(response):
    """High-minus-low contrasts, interactions use product-coded levels."""
    x, labels = features(factorial(), quadratic=False)
    return {name: float(2 * x[:, i] @ response / len(x))
            for i, name in enumerate(labels) if i}


def decode(z, config):
    z = np.asarray(z)
    if z.shape != (5,) or not np.isfinite(z).all():
        raise ValueError('Five finite coded factors required')
    ratios = 1 + z[:4] * np.asarray(config['linear_half_ranges'])
    if np.any(ratios <= 0):
        raise ValueError('Physical parameter ratios must be positive')
    return np.r_[ratios, config['pole_center_hz'] *
                 2 ** (config['pole_log2_half_range'] * z[4])]
