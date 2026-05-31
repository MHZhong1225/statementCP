from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROPENSITY_FLOOR = 0.05


@dataclass(frozen=True)
class PropensityModel:
    model: object
    feature_count: int

    def predict(self, features: np.ndarray) -> np.ndarray:
        probabilities = self.model.predict_proba(features[:, : self.feature_count])[:, 1]
        return np.clip(probabilities, PROPENSITY_FLOOR, 1.0 - PROPENSITY_FLOOR)


@dataclass
class CoverageCDFModels:
    models: dict[int, HistGradientBoostingClassifier]
    feature_count: int

    def predict(self, features: np.ndarray, radii: np.ndarray, arm: int) -> np.ndarray:
        design = build_coverage_design(features[:, : self.feature_count], radii)
        probabilities = self.models[arm].predict_proba(design)[:, 1]
        return np.clip(probabilities, 0.01, 0.99)


@dataclass
class CrossFittedCoverageCDFModels:
    fold_ids: np.ndarray
    fold_models: dict[int, CoverageCDFModels]

    def predict(self, features: np.ndarray, radii: np.ndarray, arm: int) -> np.ndarray:
        predictions = np.empty(features.shape[0], dtype=float)
        for fold_id, models in self.fold_models.items():
            mask = self.fold_ids == fold_id
            if np.any(mask):
                predictions[mask] = models.predict(features[mask], radii[mask], arm)
        return predictions


def fit_propensity_model(features: np.ndarray, treatment: np.ndarray, mode: str) -> PropensityModel:
    if mode == "oracle":
        raise ValueError("Oracle propensities are stored in the synthetic data object.")
    feature_count = features.shape[1] if mode == "correct" else min(2, features.shape[1])
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, C=1.0),
    )
    model.fit(features[:, :feature_count], treatment)
    return PropensityModel(model=model, feature_count=feature_count)


def fit_coverage_cdf_models(
    features: np.ndarray,
    treatment: np.ndarray,
    residuals: dict[int, np.ndarray],
    mode: str,
    seed: int,
    n_radius_grid: int = 24,
    max_observations_per_arm: int = 3500,
) -> CoverageCDFModels:
    feature_count = features.shape[1] if mode == "flexible" else min(2, features.shape[1])
    rng = np.random.default_rng(seed)
    models: dict[int, HistGradientBoostingClassifier] = {}
    for arm in (0, 1):
        arm_indices = np.flatnonzero(treatment == arm)
        if arm_indices.size > max_observations_per_arm:
            arm_indices = rng.choice(arm_indices, size=max_observations_per_arm, replace=False)
        arm_residuals = residuals[arm][arm_indices]
        radius_grid = np.quantile(arm_residuals, np.linspace(0.04, 0.98, n_radius_grid))
        repeated_features = np.repeat(features[arm_indices, :feature_count], n_radius_grid, axis=0)
        repeated_radii = np.tile(radius_grid, arm_indices.size)
        labels = (np.repeat(arm_residuals, n_radius_grid) <= repeated_radii).astype(int)
        design = build_coverage_design(repeated_features, repeated_radii)
        model = HistGradientBoostingClassifier(
            max_iter=130 if mode == "flexible" else 50,
            learning_rate=0.07,
            max_leaf_nodes=31 if mode == "flexible" else 7,
            l2_regularization=0.01,
            random_state=seed + arm,
        )
        model.fit(design, labels)
        models[arm] = model
    return CoverageCDFModels(models=models, feature_count=feature_count)


def build_coverage_design(features: np.ndarray, radii: np.ndarray) -> np.ndarray:
    radius_column = radii.reshape(-1, 1)
    return np.hstack([features, radius_column, np.log1p(radius_column)])
