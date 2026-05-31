from __future__ import annotations

from dataclasses import dataclass

import numpy as np


MIN_CELL_WEIGHT = 35.0


@dataclass(frozen=True)
class CalibratedIntervals:
    center_0: np.ndarray
    center_1: np.ndarray
    radius_0: np.ndarray
    radius_1: np.ndarray

    def lower(self, arm: int) -> np.ndarray:
        center = self.center_1 if arm == 1 else self.center_0
        radius = self.radius_1 if arm == 1 else self.radius_0
        return center - radius

    def upper(self, arm: int) -> np.ndarray:
        center = self.center_1 if arm == 1 else self.center_0
        radius = self.radius_1 if arm == 1 else self.radius_0
        return center + radius


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    if values.size == 0:
        return 0.0
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    cutoff = quantile * cumulative[-1]
    return float(sorted_values[np.searchsorted(cumulative, cutoff, side="left")])


def arm_weights(treatment: np.ndarray, propensity: np.ndarray, arm: int) -> np.ndarray:
    probabilities = propensity if arm == 1 else 1.0 - propensity
    return (treatment == arm).astype(float) / probabilities


def fit_weighted_base_radii(
    residuals: np.ndarray,
    treatment: np.ndarray,
    propensity: np.ndarray,
    alpha: float,
) -> dict[int, float]:
    radii: dict[int, float] = {}
    for arm in (0, 1):
        mask = treatment == arm
        values = residuals[arm][mask]
        weights = arm_weights(treatment, propensity, arm)[mask]
        radii[arm] = weighted_quantile(values, weights, 1.0 - alpha)
    return radii


def fit_split_radii(
    residuals: dict[int, np.ndarray],
    treatment: np.ndarray,
    propensity: np.ndarray,
    alpha: float,
    weighted: bool,
) -> dict[int, float]:
    radii: dict[int, float] = {}
    for arm in (0, 1):
        mask = treatment == arm
        values = residuals[arm][mask]
        if weighted:
            weights = arm_weights(treatment, propensity, arm)[mask]
            radii[arm] = weighted_quantile(values, weights, 1.0 - alpha)
        else:
            radii[arm] = float(np.quantile(values, 1.0 - alpha))
    return radii


def make_split_intervals(
    centers_0: np.ndarray,
    centers_1: np.ndarray,
    radii: dict[int, float],
) -> CalibratedIntervals:
    return CalibratedIntervals(
        center_0=centers_0,
        center_1=centers_1,
        radius_0=np.full_like(centers_0, radii[0], dtype=float),
        radius_1=np.full_like(centers_1, radii[1], dtype=float),
    )


def fit_group_radii(
    residuals: dict[int, np.ndarray],
    treatment: np.ndarray,
    propensity: np.ndarray,
    groups: dict[str, np.ndarray],
    alpha: float,
) -> dict[int, dict[str, float]]:
    group_radii: dict[int, dict[str, float]] = {0: {}, 1: {}}
    for arm in (0, 1):
        weights = arm_weights(treatment, propensity, arm)
        for name, group_mask in groups.items():
            mask = group_mask & (treatment == arm)
            if weights[mask].sum() < MIN_CELL_WEIGHT:
                continue
            group_radii[arm][name] = weighted_quantile(residuals[arm][mask], weights[mask], 1.0 - alpha)
    return group_radii


def make_group_intervals(
    centers_0: np.ndarray,
    centers_1: np.ndarray,
    base_radii: dict[int, float],
    group_radii: dict[int, dict[str, float]],
    groups: dict[str, np.ndarray],
) -> CalibratedIntervals:
    radii = {
        0: np.full_like(centers_0, base_radii[0], dtype=float),
        1: np.full_like(centers_1, base_radii[1], dtype=float),
    }
    for arm in (0, 1):
        for name, radius in group_radii[arm].items():
            radii[arm][groups[name]] = np.maximum(radii[arm][groups[name]], radius)
    return CalibratedIntervals(centers_0, centers_1, radii[0], radii[1])


def assign_buckets(radius: np.ndarray, n_buckets: int) -> np.ndarray:
    if np.allclose(radius, radius[0]):
        return np.zeros(radius.shape[0], dtype=int)
    edges = np.quantile(radius, np.linspace(0.0, 1.0, n_buckets + 1)[1:-1])
    return np.digitize(radius, edges, right=True)


def fit_dr_causal_batch_mvp(
    residuals: dict[int, np.ndarray],
    treatment: np.ndarray,
    propensity: np.ndarray,
    groups: dict[str, np.ndarray],
    features: np.ndarray,
    coverage_models,
    base_radii: dict[int, float],
    alpha: float,
    n_buckets: int,
    max_rounds: int,
    learning_rate: float,
    tolerance: float,
    audit_mode: str = "full",
) -> tuple[dict[int, np.ndarray], list[dict[str, float]]]:
    radii = {
        0: np.full(treatment.shape[0], base_radii[0], dtype=float),
        1: np.full(treatment.shape[0], base_radii[1], dtype=float),
    }
    history: list[dict[str, float]] = []
    target = 1.0 - alpha

    for round_index in range(max_rounds):
        best_update: tuple[int, str, int, float, float] | None = None
        best_gap = 0.0
        for arm in (0, 1):
            covered = (residuals[arm] <= radii[arm]).astype(float)
            outcome_regression = coverage_models.predict(features, radii[arm], arm)
            treatment_probability = propensity if arm == 1 else 1.0 - propensity
            dr_scores = outcome_regression + (treatment == arm) * (covered - outcome_regression) / treatment_probability
            dr_scores = np.clip(dr_scores, -0.25, 1.25)
            buckets = assign_buckets(radii[arm], n_buckets)
            for group_name, bucket, mask in iter_audit_cells(groups, buckets, audit_mode, n_buckets):
                if mask.sum() < MIN_CELL_WEIGHT:
                    continue
                coverage = float(dr_scores[mask].mean())
                gap = target - coverage
                if abs(gap) > abs(best_gap):
                    best_gap = gap
                    best_update = (arm, group_name, bucket, coverage, float(mask.sum()))

        if best_update is None or abs(best_gap) <= tolerance:
            history.append({"round": round_index, "max_gap": abs(best_gap), "coverage": np.nan})
            break

        arm, group_name, bucket, coverage, cell_size = best_update
        buckets = assign_buckets(radii[arm], n_buckets)
        update_mask = audit_update_mask(groups, buckets, group_name, bucket, audit_mode)
        signed_step = learning_rate if best_gap > 0 else -learning_rate
        radii[arm][update_mask] = np.maximum(0.01, radii[arm][update_mask] + signed_step)
        history.append(
            {
                "round": round_index,
                "arm": arm,
                "coverage": coverage,
                "max_gap": abs(best_gap),
                "cell_size": cell_size,
                "mean_radius": float(radii[arm].mean()),
            }
        )
    return radii, history


def iter_audit_cells(
    groups: dict[str, np.ndarray],
    buckets: np.ndarray,
    audit_mode: str,
    n_buckets: int,
):
    if audit_mode == "full":
        for group_name, group_mask in groups.items():
            for bucket in range(n_buckets):
                yield group_name, bucket, group_mask & (buckets == bucket)
    elif audit_mode == "no_bucket":
        for group_name, group_mask in groups.items():
            yield group_name, -1, group_mask
    elif audit_mode == "bucket_only":
        all_mask = np.ones(buckets.shape[0], dtype=bool)
        for bucket in range(n_buckets):
            yield "all", bucket, all_mask & (buckets == bucket)
    else:
        raise ValueError(f"Unknown audit_mode: {audit_mode}")


def audit_update_mask(
    groups: dict[str, np.ndarray],
    buckets: np.ndarray,
    group_name: str,
    bucket: int,
    audit_mode: str,
) -> np.ndarray:
    if audit_mode == "full":
        return groups[group_name] & (buckets == bucket)
    if audit_mode == "no_bucket":
        return groups[group_name]
    if audit_mode == "bucket_only":
        return buckets == bucket
    raise ValueError(f"Unknown audit_mode: {audit_mode}")
