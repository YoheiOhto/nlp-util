from __future__ import annotations
import random


def bootstrap_ci(
    scores: list[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for a list of scores.

    Args:
        scores: Per-sample scores (e.g., per-document F1).
        n_bootstrap: Number of bootstrap resamples.
        ci: Confidence level (0.95 = 95% CI).
        seed: Random seed.

    Returns:
        (lower, upper) bounds at the requested CI.
    """
    rng = random.Random(seed)
    n = len(scores)
    means = sorted(
        sum(rng.choices(scores, k=n)) / n
        for _ in range(n_bootstrap)
    )
    lo = int((1 - ci) / 2 * n_bootstrap)
    hi = int((1 + ci) / 2 * n_bootstrap)
    return means[lo], means[hi]


def compare_models(
    scores_a: list[float],
    scores_b: list[float],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap test for whether model B beats model A.

    Args:
        scores_a: Per-sample scores for baseline model.
        scores_b: Per-sample scores for candidate model.
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed.

    Returns:
        Dict with mean_a, mean_b, diff, p_value (one-tailed: B > A).
    """
    rng = random.Random(seed)
    n = len(scores_a)
    observed_diff = sum(scores_b) / n - sum(scores_a) / n
    exceed = sum(
        1
        for _ in range(n_bootstrap)
        if sum(scores_b[i] - scores_a[i] for i in (rng.randrange(n) for _ in range(n))) / n
        >= observed_diff
    )
    return {
        "mean_a":  sum(scores_a) / n,
        "mean_b":  sum(scores_b) / n,
        "diff":    observed_diff,
        "p_value": exceed / n_bootstrap,
    }
