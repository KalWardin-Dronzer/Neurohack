from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class Estimate:
    mean: float
    lo: float
    hi: float
    n: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def bootstrap_ci(values: Sequence[float], n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05) -> Estimate:
    """Mean with a percentile-bootstrap interval, resampling questions.

    Questions from the same conversation are correlated, so these intervals are narrower than a
    conversation-level bootstrap would give; with only ten conversations that one would be very wide.
    """
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    chunk = 1000
    for start in range(0, n_boot, chunk):
        stop = min(start + chunk, n_boot)
        means[start:stop] = data[rng.integers(0, data.size, size=(stop - start, data.size))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return Estimate(mean=float(data.mean()), lo=float(lo), hi=float(hi), n=int(data.size))


def paired_difference(a: Sequence[float], b: Sequence[float], n_boot: int = 10_000, seed: int = 0) -> Estimate:
    """Mean of a - b over the same questions; an interval excluding 0 means the gap is not resampling noise."""
    left, right = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if left.shape != right.shape:
        raise ValueError("paired samples must have the same length")
    return bootstrap_ci(left - right, n_boot=n_boot, seed=seed)


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if len(values) else 0.0
