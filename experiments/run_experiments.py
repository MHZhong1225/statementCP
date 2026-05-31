from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.conformal import (
    CalibratedIntervals,
    fit_dr_causal_batch_mvp,
    fit_group_radii,
    fit_split_radii,
    fit_weighted_base_radii,
    make_group_intervals,
    make_split_intervals,
)
from src.data import (
    SyntheticData,
    build_groups,
    generate_synthetic_data,
    load_acic2016_replication,
    load_ihdp_replication,
    load_mimic_early_icu_cohort,
    split_data,
    subset_data,
)
from src.metrics import evaluate_method, summarize_results
from src.models import fit_outcome_models
from src.nuisance import CrossFittedCoverageCDFModels, fit_coverage_cdf_models, fit_propensity_model
from src.plots import write_plots, write_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["synthetic", "ihdp", "acic2016", "mimic_early_icu"], default="synthetic")
    parser.add_argument("--scenario", choices=["rct", "observational", "both"], default="both")
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--n-samples", type=int, default=18000)
    parser.add_argument("--n-features", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--buckets", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--crossfit-folds", type=int, default=3)
    return parser.parse_args()


def residuals_by_arm(models, features: np.ndarray, outcome: np.ndarray) -> dict[int, np.ndarray]:
    return {
        0: np.abs(outcome - models.predict(features, 0)),
        1: np.abs(outcome - models.predict(features, 1)),
    }


def centers_by_arm(models, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return models.predict(features, 0), models.predict(features, 1)


def group_radii_from_patched(
    patched_radii: dict[int, np.ndarray],
    groups: dict[str, np.ndarray],
    quantile: float = 0.90,
) -> dict[int, dict[str, float]]:
    group_radii: dict[int, dict[str, float]] = {0: {}, 1: {}}
    for arm in (0, 1):
        for group_name, group_mask in groups.items():
            if group_mask.sum() >= 20:
                group_radii[arm][group_name] = float(np.quantile(patched_radii[arm][group_mask], quantile))
    return group_radii


def k_fold_indices(n_samples: int, n_folds: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    return [fold for fold in np.array_split(indices, n_folds) if fold.size > 0]


def fit_unified_crossfit_dr_group_radii(
    calibration: SyntheticData,
    calibration_residuals: dict[int, np.ndarray],
    args: argparse.Namespace,
    seed: int,
) -> tuple[dict[int, float], dict[int, dict[str, float]]]:
    folds = k_fold_indices(calibration.features.shape[0], args.crossfit_folds, seed)
    fold_ids = np.empty(calibration.features.shape[0], dtype=int)
    oof_propensity = np.empty(calibration.features.shape[0], dtype=float)
    fold_models = {}
    all_indices = np.arange(calibration.features.shape[0])
    for fold_id, audit_indices in enumerate(folds):
        nuisance_indices = np.setdiff1d(all_indices, audit_indices, assume_unique=False)
        nuisance = subset_data(calibration, nuisance_indices)
        nuisance_residuals = {
            arm: calibration_residuals[arm][nuisance_indices]
            for arm in (0, 1)
        }
        propensity_model = fit_propensity_model(nuisance.features, nuisance.treatment, mode="correct")
        oof_propensity[audit_indices] = propensity_model.predict(calibration.features[audit_indices])
        fold_ids[audit_indices] = fold_id
        fold_models[fold_id] = fit_coverage_cdf_models(
            nuisance.features,
            nuisance.treatment,
            nuisance_residuals,
            mode="flexible",
            seed=seed + 101 * fold_id,
        )

    base_radii = fit_weighted_base_radii(
        calibration_residuals,
        calibration.treatment,
        oof_propensity,
        args.alpha,
    )
    coverage_model = CrossFittedCoverageCDFModels(fold_ids=fold_ids, fold_models=fold_models)
    patched_radii, _ = fit_dr_causal_batch_mvp(
        calibration_residuals,
        calibration.treatment,
        oof_propensity,
        build_groups(calibration.features, calibration.feature_names),
        calibration.features,
        coverage_model,
        base_radii,
        alpha=args.alpha,
        n_buckets=args.buckets,
        max_rounds=120,
        learning_rate=0.06,
        tolerance=0.012,
    )
    return base_radii, group_radii_from_patched(patched_radii, build_groups(calibration.features, calibration.feature_names))


def prepare_data(args: argparse.Namespace, scenario: str, repeat: int, seed: int) -> SyntheticData:
    if args.dataset == "ihdp":
        return load_ihdp_replication(repeat)
    if args.dataset == "acic2016":
        return load_acic2016_replication(repeat)
    if args.dataset == "mimic_early_icu":
        return load_mimic_early_icu_cohort()
    return generate_synthetic_data(args.n_samples, args.n_features, scenario, seed)


def scenario_label(args: argparse.Namespace, scenario: str) -> str:
    if args.dataset != "synthetic":
        return args.dataset
    return scenario


def estimate_or_use_propensity(data: SyntheticData) -> np.ndarray:
    if np.all(np.isfinite(data.propensity)):
        return data.propensity
    return fit_propensity_model(data.features, data.treatment, mode="correct").predict(data.features)


def run_one_repeat(args: argparse.Namespace, scenario: str, repeat: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed = args.seed + 1000 * repeat + (0 if scenario == "rct" else 100000)
    data = prepare_data(args, scenario, repeat, seed)
    train, calibration, test = split_data(data, seed + 17)

    models = fit_outcome_models(
        train.features,
        train.treatment,
        train.outcome,
        seed=seed + 31,
        epochs=args.epochs,
    )

    calibration_residuals = residuals_by_arm(models, calibration.features, calibration.outcome)
    test_centers_0, test_centers_1 = centers_by_arm(models, test.features)
    calibration_groups = build_groups(calibration.features, calibration.feature_names)
    test_groups = build_groups(test.features, test.feature_names)
    calibration_propensity = estimate_or_use_propensity(calibration)

    unweighted_radii = fit_split_radii(
        calibration_residuals,
        calibration.treatment,
        calibration_propensity,
        args.alpha,
        weighted=False,
    )
    weighted_radii = fit_split_radii(
        calibration_residuals,
        calibration.treatment,
        calibration_propensity,
        args.alpha,
        weighted=True,
    )
    intervals: dict[str, CalibratedIntervals] = {
        "split": make_split_intervals(test_centers_0, test_centers_1, unweighted_radii),
        "weighted_split": make_split_intervals(test_centers_0, test_centers_1, weighted_radii),
    }
    baseline_group_radii = fit_group_radii(
        calibration_residuals,
        calibration.treatment,
        calibration_propensity,
        calibration_groups,
        args.alpha,
    )
    intervals["group_conformal"] = make_group_intervals(
        test_centers_0,
        test_centers_1,
        weighted_radii,
        baseline_group_radii,
        test_groups,
    )

    base_radii, final_group_radii = fit_unified_crossfit_dr_group_radii(
        calibration,
        calibration_residuals,
        args,
        seed + 71,
    )
    intervals["dr_cf_unified_mvp"] = make_group_intervals(
        test_centers_0,
        test_centers_1,
        base_radii,
        final_group_radii,
        test_groups,
    )

    summaries = []
    cells = []
    for method, method_intervals in intervals.items():
        result = evaluate_method(method, scenario_label(args, scenario), repeat, test, method_intervals, args.alpha, args.buckets)
        summaries.append(result.summary)
        cells.append(result.cells)
    return pd.concat(summaries, ignore_index=True), pd.concat(cells, ignore_index=True)


def main() -> None:
    args = parse_args()
    scenarios = [args.dataset] if args.dataset != "synthetic" else (["rct", "observational"] if args.scenario == "both" else [args.scenario])

    summary_frames = []
    cell_frames = []
    for scenario in scenarios:
        for repeat in range(args.repeats):
            print(f"Running scenario={scenario} repeat={repeat}", flush=True)
            summary, cells = run_one_repeat(args, scenario, repeat)
            summary_frames.append(summary)
            cell_frames.append(cells)

    results_dir = Path("results")
    figures_dir = Path("figures")
    reports_dir = Path("reports")
    results_dir.mkdir(exist_ok=True)

    summary = pd.concat(summary_frames, ignore_index=True)
    cells = pd.concat(cell_frames, ignore_index=True)
    prefix = "" if args.dataset == "synthetic" else f"{args.dataset}_"
    summary.to_csv(results_dir / f"{prefix}summary.csv", index=False)
    cells.to_csv(results_dir / f"{prefix}cell_metrics.csv", index=False)
    summarize_results(summary).to_csv(results_dir / f"{prefix}summary_aggregate.csv", index=False)
    write_plots(summary, figures_dir)
    write_report(summary, reports_dir / f"{prefix}experiment_report.md")
    print(summarize_results(summary).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
