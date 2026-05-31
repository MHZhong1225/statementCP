from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def write_plots(summary: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    aggregate = (
        summary.groupby(["scenario", "method"], as_index=False)
        .agg(
            worst_cell_coverage=("worst_cell_coverage", "mean"),
            max_cell_undercoverage=("max_cell_undercoverage", "mean"),
            avg_length=("avg_length", "mean"),
        )
        .sort_values(["scenario", "method"])
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    sns.scatterplot(
        data=aggregate,
        x="avg_length",
        y="worst_cell_coverage",
        hue="method",
        style="scenario",
        s=110,
        ax=axes[0],
    )
    axes[0].axhline(0.9, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("Coverage-Length Tradeoff")
    axes[0].set_xlabel("Average interval length")
    axes[0].set_ylabel("Worst group-threshold coverage")

    sns.barplot(
        data=aggregate,
        x="method",
        y="max_cell_undercoverage",
        hue="scenario",
        ax=axes[1],
    )
    axes[1].set_title("Worst Cell Undercoverage")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Target minus worst coverage")
    axes[1].tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_dir / "coverage_tradeoff.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.8))
    sns.boxplot(
        data=summary,
        x="method",
        y="worst_cell_coverage",
        hue="scenario",
        ax=ax,
    )
    ax.axhline(0.9, color="black", linestyle="--", linewidth=1)
    ax.set_title("Worst Cell Coverage Across Repeats")
    ax.set_xlabel("")
    ax.set_ylabel("Worst group-threshold coverage")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_dir / "worst_cell_coverage.png", dpi=180)
    plt.close(fig)


def write_report(summary: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate = (
        summary.groupby(["scenario", "method"], as_index=False)
        .agg(
            overall_coverage=("overall_coverage", "mean"),
            worst_group_coverage=("worst_group_coverage", "mean"),
            worst_cell_coverage=("worst_cell_coverage", "mean"),
            max_cell_undercoverage=("max_cell_undercoverage", "mean"),
            avg_length=("avg_length", "mean"),
        )
        .sort_values(["scenario", "method"])
    )
    table = dataframe_to_markdown(aggregate)
    evaluation_mode = "potential_outcome"
    if "evaluation" in summary.columns and (summary["evaluation"] == "observed_arm").all():
        evaluation_mode = "observed_arm"
    if evaluation_mode == "observed_arm":
        evaluation_text = (
            "Target coverage is 0.90. Coverage is evaluated only on the observed "
            "treatment arm because true counterfactual outcomes are unavailable in "
            "this real-world dataset."
        )
    else:
        evaluation_text = (
            "Target coverage is 0.90. Coverage is evaluated against known simulated "
            "potential outcomes, which are not used by the algorithms during calibration."
        )
    output_path.write_text(
        "# Experiment Report\n\n"
        f"{evaluation_text}\n\n"
        f"{table}\n\n"
        "The final `dr_cf_unified_mvp` method uses out-of-fold doubly robust "
        "group-threshold audits and runs one unified patching procedure over "
        "all calibration samples.\n",
        encoding="utf-8",
    )


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)
