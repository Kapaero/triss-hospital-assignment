from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OLD_FEATURES = ROOT / "results" / "mimic_dynamic_triss_features.csv"
OLD_BINS = ROOT / "results" / "mimic_patient_time_bins.csv"
OUT = ROOT / "results"


def vital_score(df: pd.DataFrame) -> pd.Series:
    gcs = (df["GCS"] / 15.0).clip(0, 1)
    sbp = ((df["SBP"] - 60.0) / 80.0).clip(0, 1)
    rr = (1.0 - ((df["RR"] - 20.0).abs() / 22.0)).clip(0, 1)
    shock = (1.0 - ((df["shock_index"] - 0.55) / 1.45)).clip(0, 1)
    return (0.35 * gcs + 0.25 * sbp + 0.20 * rr + 0.20 * shock).clip(0, 1)


def add_generic_risk(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "injury_dx_count" not in df:
        df["injury_dx_count"] = 0
    if "anchor_age" not in df:
        df["anchor_age"] = 55
    df["vital_score"] = vital_score(df)
    df["injury_proxy"] = np.clip(1 + 4 * df["injury_dx_count"], 1, 35)
    z = (
        0.20
        + 4.00 * df["vital_score"]
        - 0.025 * df["injury_proxy"]
        - 0.012 * np.maximum(df["anchor_age"] - 55, 0)
        - 0.20 * df["shock_index"]
    )
    df["survival_t0"] = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
    df["hazard_proxy"] = np.clip(
        0.010
        + 0.003 * df["injury_proxy"]
        + 0.08 * np.maximum(df["shock_index"] - 0.75, 0)
        + 0.04 * (1 - df["vital_score"]),
        0.015,
        1.0,
    )
    df["survival_6h"] = df["survival_t0"] * np.exp(-df["hazard_proxy"] * 6)
    df["survival_12h"] = df["survival_t0"] * np.exp(-df["hazard_proxy"] * 12)
    df["red_zone"] = (df["survival_6h"] < 0.50).astype(int)
    return df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(OLD_FEATURES)
    features = add_generic_risk(features)
    keep = [
        "subject_id",
        "hadm_id",
        "stay_id",
        "charttime",
        "time_from_icu_h",
        "anchor_age",
        "gender",
        "HR",
        "SBP",
        "RR",
        "GCS",
        "shock_index",
        "vital_score",
        "injury_dx_count",
        "injury_proxy",
        "hospital_expire_flag",
        "survival_t0",
        "hazard_proxy",
        "survival_6h",
        "survival_12h",
        "red_zone",
    ]
    features[keep].to_csv(OUT / "mimic_dynamic_risk_features.csv", index=False)

    bins = pd.read_csv(OLD_BINS)
    static = (
        features.sort_values("time_from_icu_h")
        .groupby("stay_id")
        .last()[["anchor_age", "injury_dx_count"]]
        .reset_index()
    )
    bins = bins.drop(columns=[c for c in ["survival_t0", "survival_6h", "red_zone", "RTS"] if c in bins], errors="ignore")
    bins = bins.merge(static, on="stay_id", how="left")
    bins = add_generic_risk(bins)
    bin_keep = [
        "subject_id",
        "hadm_id",
        "stay_id",
        "time_bin",
        "HR",
        "SBP",
        "RR",
        "GCS",
        "shock_index",
        "vital_score",
        "survival_t0",
        "survival_6h",
        "red_zone",
    ]
    bins[bin_keep].to_csv(OUT / "mimic_patient_risk_time_bins.csv", index=False)

    summary = pd.DataFrame(
        [
            {"metric": "rows", "value": len(features)},
            {"metric": "icu_stays", "value": features["stay_id"].nunique()},
            {"metric": "patients", "value": features["subject_id"].nunique()},
            {"metric": "hospital_mortality_events", "value": features.groupby("hadm_id")["hospital_expire_flag"].max().sum()},
            {"metric": "mean_vital_score", "value": features["vital_score"].mean()},
            {"metric": "mean_shock_index", "value": features["shock_index"].mean()},
            {"metric": "mean_survival_6h", "value": features["survival_6h"].mean()},
            {"metric": "red_zone_rate", "value": features["red_zone"].mean()},
        ]
    )
    summary.to_csv(OUT / "mimic_risk_feature_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
