from __future__ import annotations

from dataclasses import dataclass
import gzip
from pathlib import Path
from urllib.request import urlretrieve
import zipfile

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticData:
    features: np.ndarray
    treatment: np.ndarray
    outcome: np.ndarray
    potential_outcome_0: np.ndarray
    potential_outcome_1: np.ndarray
    propensity: np.ndarray
    feature_names: tuple[str, ...] | None = None


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def generate_synthetic_data(
    n_samples: int,
    n_features: int,
    scenario: str,
    seed: int,
) -> SyntheticData:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n_samples, n_features)).astype(np.float32)

    baseline = (
        0.8 * np.sin(features[:, 0])
        + 0.5 * features[:, 1]
        - 0.35 * features[:, 2] ** 2
        + 0.25 * features[:, 3] * features[:, 4]
    )
    treatment_effect = (
        0.8
        + 0.55 * (features[:, 0] > 0)
        - 0.45 * (features[:, 1] < -0.4)
        + 0.35 * np.sin(features[:, 5])
    )

    heteroskedastic_noise = 0.35 + 0.45 * sigmoid(1.6 * features[:, 0] - features[:, 2])
    group_noise = 0.45 * ((features[:, 0] > 0) & (features[:, 1] > 0)).astype(float)

    noise_0 = rng.normal(scale=heteroskedastic_noise + group_noise)
    noise_1 = rng.normal(scale=1.15 * heteroskedastic_noise + 0.5 * group_noise)

    potential_outcome_0 = baseline + noise_0
    potential_outcome_1 = baseline + treatment_effect + noise_1

    if scenario == "rct":
        propensity = np.full(n_samples, 0.5)
    elif scenario == "observational":
        logits = 0.9 * features[:, 0] - 0.7 * features[:, 1] + 0.45 * features[:, 2]
        propensity = np.clip(sigmoid(logits), 0.08, 0.92)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    treatment = rng.binomial(1, propensity).astype(int)
    outcome = np.where(treatment == 1, potential_outcome_1, potential_outcome_0)

    return SyntheticData(
        features=features,
        treatment=treatment,
        outcome=outcome,
        potential_outcome_0=potential_outcome_0,
        potential_outcome_1=potential_outcome_1,
        propensity=propensity,
        feature_names=None,
    )


def split_data(data: SyntheticData, seed: int) -> tuple[SyntheticData, SyntheticData, SyntheticData]:
    rng = np.random.default_rng(seed)
    n_samples = data.features.shape[0]
    indices = rng.permutation(n_samples)
    train_end = int(0.45 * n_samples)
    calibration_end = int(0.7 * n_samples)
    return (
        subset_data(data, indices[:train_end]),
        subset_data(data, indices[train_end:calibration_end]),
        subset_data(data, indices[calibration_end:]),
    )


def load_ihdp_replication(replication: int, data_dir: Path | str = "data/ihdp") -> SyntheticData:
    data_path = Path(data_dir)
    train = load_ihdp_npz(data_path / "ihdp_npci_1-100.train.npz", "train")
    test = load_ihdp_npz(data_path / "ihdp_npci_1-100.test.npz", "test")
    rep_index = replication % train["t"].shape[1]
    features = np.concatenate(
        [
            np.swapaxes(train["x"], 1, 2)[:, rep_index, :],
            np.swapaxes(test["x"], 1, 2)[:, rep_index, :],
        ],
        axis=0,
    ).astype(np.float32)
    treatment = np.concatenate([train["t"][:, rep_index], test["t"][:, rep_index]], axis=0).astype(int)
    factual = np.concatenate([train["yf"][:, rep_index], test["yf"][:, rep_index]], axis=0)
    counterfactual = np.concatenate([train["ycf"][:, rep_index], test["ycf"][:, rep_index]], axis=0)
    potential_outcome_0 = np.where(treatment == 0, factual, counterfactual)
    potential_outcome_1 = np.where(treatment == 1, factual, counterfactual)
    return SyntheticData(
        features=features,
        treatment=treatment,
        outcome=factual,
        potential_outcome_0=potential_outcome_0,
        potential_outcome_1=potential_outcome_1,
        propensity=np.full(features.shape[0], np.nan),
        feature_names=None,
    )


def load_ihdp_npz(path: Path, split_name: str) -> dict[str, np.ndarray]:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"http://www.fredjo.com/files/ihdp_npci_1-100.{split_name}.npz"
        urlretrieve(url, path)
    return dict(np.load(path))


def load_mimic_early_icu_cohort(
    mimic_zip_path: Path | str = "/home/ubuntu/zmh/sgcp/dataset/mimic-iv-3.1.zip",
    cache_path: Path | str = "data/mimic/mimic_early_icu_los.csv",
) -> SyntheticData:
    cache = Path(cache_path)
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        cohort = build_mimic_early_icu_cohort(Path(mimic_zip_path))
        cohort.to_csv(cache, index=False)
    cohort = pd.read_csv(cache)
    treatment = cohort["early_icu"].astype(int).to_numpy()
    outcome = cohort["log_los_days"].astype(float).to_numpy()
    feature_cols = [column for column in cohort.columns if column.startswith("x_")]
    features = cohort[feature_cols].astype(np.float32).to_numpy()
    missing_potential_outcomes = np.full(cohort.shape[0], np.nan)
    return SyntheticData(
        features=features,
        treatment=treatment,
        outcome=outcome,
        potential_outcome_0=missing_potential_outcomes.copy(),
        potential_outcome_1=missing_potential_outcomes.copy(),
        propensity=np.full(cohort.shape[0], np.nan),
        feature_names=tuple(feature_cols),
    )


def load_acic2016_replication(
    replication: int,
    data_dir: Path | str = "data/acic2016",
) -> SyntheticData:
    data_path = Path(data_dir)
    ensure_acic2016_files(data_path)
    features_frame = pd.read_csv(data_path / "x.csv")
    features_frame = pd.get_dummies(features_frame, drop_first=True)
    features = features_frame.astype(float).to_numpy()
    features = standardize_features(features)
    outcome_path = data_path / f"zymu_{(replication % 10) + 1}.csv"
    outcomes = pd.read_csv(outcome_path)
    treatment = outcomes["z"].astype(int).to_numpy()
    potential_outcome_0 = outcomes["y0"].astype(float).to_numpy()
    potential_outcome_1 = outcomes["y1"].astype(float).to_numpy()
    factual = np.where(treatment == 1, potential_outcome_1, potential_outcome_0)
    return SyntheticData(
        features=features.astype(np.float32),
        treatment=treatment,
        outcome=factual,
        potential_outcome_0=potential_outcome_0,
        potential_outcome_1=potential_outcome_1,
        propensity=np.full(features.shape[0], np.nan),
        feature_names=tuple(features_frame.columns),
    )


def ensure_acic2016_files(data_path: Path) -> None:
    base_url = "https://raw.githubusercontent.com/BiomedSciAI/causallib/master/causallib/datasets/data/acic_challenge_2016"
    data_path.mkdir(parents=True, exist_ok=True)
    names = ["x.csv"] + [f"zymu_{index}.csv" for index in range(1, 11)]
    for name in names:
        path = data_path / name
        if not path.exists():
            urlretrieve(f"{base_url}/{name}", path)


def standardize_features(features: np.ndarray) -> np.ndarray:
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (features - mean) / std


def build_mimic_early_icu_cohort(mimic_zip_path: Path) -> pd.DataFrame:
    admissions = read_mimic_member(mimic_zip_path, "mimic-iv-3.1/hosp/admissions.csv.gz")
    patients = read_mimic_member(mimic_zip_path, "mimic-iv-3.1/hosp/patients.csv.gz")
    icustays = read_mimic_member(mimic_zip_path, "mimic-iv-3.1/icu/icustays.csv.gz")

    admissions["admittime"] = pd.to_datetime(admissions["admittime"], errors="coerce")
    admissions["dischtime"] = pd.to_datetime(admissions["dischtime"], errors="coerce")
    icustays["intime"] = pd.to_datetime(icustays["intime"], errors="coerce")

    icu_first = (
        icustays.dropna(subset=["intime"])
        .groupby("hadm_id", as_index=False)
        .agg(first_icu_intime=("intime", "min"))
    )
    cohort = admissions.merge(
        patients[["subject_id", "gender", "anchor_age", "anchor_year"]],
        on="subject_id",
        how="left",
    ).merge(icu_first, on="hadm_id", how="left")
    cohort = cohort.dropna(subset=["admittime", "dischtime", "anchor_age", "anchor_year"]).copy()
    cohort = cohort[cohort["dischtime"] > cohort["admittime"]].copy()

    cohort["age"] = cohort["anchor_age"] + (cohort["admittime"].dt.year - cohort["anchor_year"])
    cohort["age"] = cohort["age"].clip(lower=18.0, upper=89.0)
    cohort = cohort[cohort["age"] >= 18.0].copy()
    los_days = (cohort["dischtime"] - cohort["admittime"]).dt.total_seconds() / 86400.0
    cohort["log_los_days"] = np.log1p(los_days.clip(lower=0.01, upper=365.0))
    time_to_icu_hours = (cohort["first_icu_intime"] - cohort["admittime"]).dt.total_seconds() / 3600.0
    cohort["early_icu"] = ((time_to_icu_hours >= 0.0) & (time_to_icu_hours <= 4.0)).astype(int)

    race = cohort["race"].fillna("UNKNOWN").astype(str).str.upper()
    gender = cohort["gender"].fillna("U").astype(str).str.upper()
    insurance = cohort["insurance"].fillna("UNKNOWN").astype(str).str.upper()
    admission_type = cohort["admission_type"].fillna("UNKNOWN").astype(str).str.upper()
    marital = cohort["marital_status"].fillna("UNKNOWN").astype(str).str.upper()

    cohort["x_age"] = cohort["age"].astype(float)
    cohort["x_gender_m"] = (gender == "M").astype(int)
    cohort["x_non_white"] = (~race.str.contains("WHITE", regex=False)).astype(int)
    cohort["x_ins_private"] = (insurance == "PRIVATE").astype(int)
    cohort["x_ins_medicare"] = (insurance == "MEDICARE").astype(int)
    cohort["x_ins_medicaid"] = (insurance == "MEDICAID").astype(int)
    cohort["x_adm_elective"] = admission_type.str.contains("ELECTIVE", regex=False).astype(int)
    cohort["x_adm_urgent"] = admission_type.str.contains("URGENT", regex=False).astype(int)
    cohort["x_adm_observation"] = admission_type.str.contains("OBSERVATION", regex=False).astype(int)
    cohort["x_marital_married"] = (marital == "MARRIED").astype(int)
    cohort["x_marital_single"] = (marital == "SINGLE").astype(int)

    keep_cols = [
        "subject_id",
        "hadm_id",
        "early_icu",
        "log_los_days",
        "x_age",
        "x_gender_m",
        "x_non_white",
        "x_ins_private",
        "x_ins_medicare",
        "x_ins_medicaid",
        "x_adm_elective",
        "x_adm_urgent",
        "x_adm_observation",
        "x_marital_married",
        "x_marital_single",
    ]
    return cohort[keep_cols].dropna().reset_index(drop=True)


def read_mimic_member(mimic_zip_path: Path, member_name: str) -> pd.DataFrame:
    with zipfile.ZipFile(mimic_zip_path) as archive:
        with archive.open(member_name) as compressed:
            with gzip.GzipFile(fileobj=compressed) as uncompressed:
                return pd.read_csv(uncompressed)


def subset_data(data: SyntheticData, indices: np.ndarray) -> SyntheticData:
    return SyntheticData(
        features=data.features[indices],
        treatment=data.treatment[indices],
        outcome=data.outcome[indices],
        potential_outcome_0=data.potential_outcome_0[indices],
        potential_outcome_1=data.potential_outcome_1[indices],
        propensity=data.propensity[indices],
        feature_names=data.feature_names,
    )


def build_groups(features: np.ndarray, feature_names: tuple[str, ...] | None = None) -> dict[str, np.ndarray]:
    if feature_names is not None and "x_age" in feature_names:
        return build_named_clinical_groups(features, feature_names)
    groups = {
        "all": np.ones(features.shape[0], dtype=bool),
        "x0_pos": features[:, 0] > 0,
        "x1_neg": features[:, 1] < 0,
        "x2_high": features[:, 2] > np.quantile(features[:, 2], 0.65),
        "x3_low": features[:, 3] < np.quantile(features[:, 3], 0.35),
    }
    groups["x0_pos_x1_neg"] = groups["x0_pos"] & groups["x1_neg"]
    groups["x0_pos_x2_high"] = groups["x0_pos"] & groups["x2_high"]
    groups["x1_neg_x3_low"] = groups["x1_neg"] & groups["x3_low"]
    return groups


def build_named_clinical_groups(features: np.ndarray, feature_names: tuple[str, ...]) -> dict[str, np.ndarray]:
    columns = {name: features[:, index] for index, name in enumerate(feature_names)}
    age = columns["x_age"]
    groups = {
        "all": np.ones(features.shape[0], dtype=bool),
        "age_ge_65": age >= 65.0,
        "age_lt_50": age < 50.0,
    }
    for name in [
        "x_gender_m",
        "x_non_white",
        "x_ins_medicare",
        "x_ins_medicaid",
        "x_ins_private",
        "x_adm_elective",
        "x_adm_urgent",
    ]:
        if name in columns:
            groups[name.removeprefix("x_")] = columns[name] > 0.5
    if "x_non_white" in columns and "x_ins_medicaid" in columns:
        groups["non_white_medicaid"] = (columns["x_non_white"] > 0.5) & (columns["x_ins_medicaid"] > 0.5)
    if "x_gender_m" in columns:
        groups["age_ge_65_male"] = groups["age_ge_65"] & (columns["x_gender_m"] > 0.5)
    if "x_ins_medicare" in columns:
        groups["age_ge_65_medicare"] = groups["age_ge_65"] & (columns["x_ins_medicare"] > 0.5)
    return groups
