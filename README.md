# Causal Batch Multivalid Conformal Prediction

This repository contains the final experimental version of our method:
unified cross-fitted doubly robust Causal-BatchMVP for binary-treatment
potential outcome intervals.

The target is:

\\[
P\\{Y(a) \\in C_a(X) \\mid g(X)=1, b_a(X)=t\\} \\approx 1 - \\alpha
\\]

for treatment arm `a`, group indicator `g`, and threshold bucket `t`.

## What Is Implemented

- Synthetic RCT and observational data generators with known potential outcomes.
- GPU-trained PyTorch outcome models for `mu_0(x)` and `mu_1(x)`.
- Final method:
  - `dr_cf_unified_mvp`, a unified cross-fitted doubly robust group-threshold
    patching method.
- Main baselines:
  - `split`, arm-wise split conformal prediction,
  - `weighted_split`, IPW split conformal prediction,
  - `group_conformal`, group-wise conformal patching without the final unified
    DR group-threshold audit.
- Metrics:
  - overall coverage,
  - worst group coverage,
  - worst group-threshold coverage,
  - interval length,
  - effective sample size.

## Run On The Remote Server

Synthetic benchmark:

```bash
ssh api
cd /home/ubuntu/zmh/causalCP
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ucp
CUDA_VISIBLE_DEVICES=0 python experiments/run_experiments.py --scenario both --repeats 8 --crossfit-folds 3
```

IHDP benchmark:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/run_experiments.py --dataset ihdp --repeats 100 --epochs 90 --crossfit-folds 3
```

ACIC 2016 benchmark:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/run_experiments.py --dataset acic2016 --repeats 10 --epochs 90 --crossfit-folds 3
```

MIMIC-IV early ICU case study:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/run_experiments.py --dataset mimic_early_icu --repeats 5 --epochs 30 --crossfit-folds 3
```

Outputs are written to:

- `results/summary.csv`
- `results/cell_metrics.csv`
- `results/summary_aggregate.csv`
- `results/ihdp_summary.csv`
- `results/ihdp_cell_metrics.csv`
- `results/ihdp_summary_aggregate.csv`
- `results/acic2016_summary.csv`
- `results/acic2016_cell_metrics.csv`
- `results/acic2016_summary_aggregate.csv`
- `results/mimic_early_icu_summary.csv`
- `results/mimic_early_icu_cell_metrics.csv`
- `results/mimic_early_icu_summary_aggregate.csv`
- `figures/coverage_tradeoff.png`
- `figures/worst_cell_coverage.png`
- `reports/experiment_report.md`
- `reports/ihdp_experiment_report.md`
- `reports/acic2016_experiment_report.md`
- `reports/mimic_early_icu_experiment_report.md`
