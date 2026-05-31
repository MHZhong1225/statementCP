from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.conformal import CalibratedIntervals, assign_buckets
from src.data import SyntheticData, build_groups


@dataclass(frozen=True)
class EvaluationResult:
    summary: pd.DataFrame
    cells: pd.DataFrame


def evaluate_method(
    method: str,
    scenario: str,
    repeat: int,
    data: SyntheticData,
    intervals: CalibratedIntervals,
    alpha: float,
    n_buckets: int,
) -> EvaluationResult:
    groups = build_groups(data.features, data.feature_names)
    rows = []
    cell_rows = []
    target = 1.0 - alpha

    for arm in (0, 1):
        potential_outcome = data.potential_outcome_1 if arm == 1 else data.potential_outcome_0
        observed_only = not np.all(np.isfinite(potential_outcome))
        lower = intervals.lower(arm)
        upper = intervals.upper(arm)
        evaluation_outcome = data.outcome if observed_only else potential_outcome
        eligible = data.treatment == arm if observed_only else np.ones(data.treatment.shape[0], dtype=bool)
        covered = (evaluation_outcome >= lower) & (evaluation_outcome <= upper)
        length = upper - lower
        buckets = assign_buckets(length / 2.0, n_buckets)

        if eligible.sum() == 0:
            continue
        overall_coverage = float(covered[eligible].mean())
        group_coverages = []
        cell_coverages = []
        effective_sizes = []

        for group_name, group_mask in groups.items():
            group_mask = group_mask & eligible
            if group_mask.sum() == 0:
                continue
            group_coverage = float(covered[group_mask].mean())
            group_coverages.append(group_coverage)
            for bucket in range(n_buckets):
                mask = group_mask & (buckets == bucket)
                if mask.sum() < 20:
                    continue
                coverage = float(covered[mask].mean())
                cell_coverages.append(coverage)
                effective_sizes.append(int(mask.sum()))
                cell_rows.append(
                    {
                        "scenario": scenario,
                        "repeat": repeat,
                        "method": method,
                        "arm": arm,
                        "group": group_name,
                        "bucket": bucket,
                        "n": int(mask.sum()),
                        "coverage": coverage,
                        "undercoverage": max(0.0, target - coverage),
                        "avg_length": float(length[mask].mean()),
                    }
                )

        rows.append(
            {
                "scenario": scenario,
                "repeat": repeat,
                "method": method,
                "arm": arm,
                "evaluation": "observed_arm" if observed_only else "potential_outcome",
                "overall_coverage": overall_coverage,
                "worst_group_coverage": float(min(group_coverages)),
                "worst_cell_coverage": float(min(cell_coverages)),
                "max_cell_undercoverage": float(max(0.0, target - min(cell_coverages))),
                "avg_length": float(length.mean()),
                "median_length": float(np.median(length)),
                "min_cell_n": int(min(effective_sizes)),
            }
        )

    return EvaluationResult(summary=pd.DataFrame(rows), cells=pd.DataFrame(cell_rows))


def summarize_results(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "overall_coverage",
        "worst_group_coverage",
        "worst_cell_coverage",
        "max_cell_undercoverage",
        "avg_length",
    ]
    grouped = summary.groupby(["scenario", "method", "arm"], as_index=False)[metrics].agg(["mean", "std"])
    grouped.columns = ["_".join(column).rstrip("_") for column in grouped.columns.to_flat_index()]
    return grouped.reset_index(drop=True)
