"""
Build dynamic RTS/TRISS-inspired features from MIMIC-IV Clinical Demo v2.2.

Expected input directory:
  data/mimic-iv-demo/2.2/

Outputs:
  results/mimic_dynamic_triss_features.csv
  results/mimic_patient_time_bins.csv
  results/mimic_feature_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ITEMIDS = {
    "HR": [220045],
    "SBP": [220179, 220050],  # non-invasive and arterial systolic BP
    "RR": [220210],
    "GCS_EYE": [220739],
    "GCS_VERBAL": [223900],
    "GCS_MOTOR": [223901],
}


def feature_name(itemid: int) -> str | None:
    for name, ids in ITEMIDS.items():
        if itemid in ids:
            return name
    return None


def code_gcs(gcs: float) -> int:
    if gcs >= 13:
        return 4
    if gcs >= 9:
        return 3
    if gcs >= 6:
        return 2
    if gcs >= 4:
        return 1
    return 0


def code_sbp(sbp: float) -> int:
    if sbp > 89:
        return 4
    if sbp >= 76:
        return 3
    if sbp >= 50:
        return 2
    if sbp >= 1:
        return 1
    return 0


def code_rr(rr: float) -> int:
    if 10 <= rr <= 29:
        return 4
    if rr > 29:
        return 3
    if rr >= 6:
        return 2
    if rr >= 1:
        return 1
    return 0


def calc_rts(row: pd.Series) -> float:
    return (
        0.9368 * code_gcs(row["GCS"])
        + 0.7326 * code_sbp(row["SBP"])
        + 0.2908 * code_rr(row["RR"])
    )


def is_injury_icd(code: str, version: int) -> bool:
    code = str(code).replace(".", "").upper()
    if not code or code == "NAN":
        return False
    if version == 9:
        digits = "".join(ch for ch in code if ch.isdigit())
        if not digits:
            return False
        prefix = int(digits[:3])
        return 800 <= prefix <= 959 and not (905 <= prefix <= 909) and not (930 <= prefix <= 939)
    return code.startswith(("S", "T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9"))


def build_iss_proxy(base: Path) -> pd.DataFrame:
    diag_path = base / "hosp" / "diagnoses_icd.csv.gz"
    if not diag_path.exists():
        return pd.DataFrame(columns=["subject_id", "hadm_id", "injury_dx_count", "iss_proxy"])
    diag = pd.read_csv(diag_path)
    diag["is_injury"] = [is_injury_icd(c, int(v)) for c, v in zip(diag["icd_code"], diag["icd_version"])]
    injury = (
        diag.groupby(["subject_id", "hadm_id"])["is_injury"]
        .sum()
        .reset_index(name="injury_dx_count")
    )
    injury["iss_proxy"] = np.clip(1 + 4 * injury["injury_dx_count"], 1, 35)
    return injury


def build_features(base: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ce_path = base / "icu" / "chartevents.csv.gz"
    icu_path = base / "icu" / "icustays.csv.gz"
    patients_path = base / "hosp" / "patients.csv.gz"
    admissions_path = base / "hosp" / "admissions.csv.gz"

    wanted = sorted({item for ids in ITEMIDS.values() for item in ids})
    chartevents = pd.read_csv(
        ce_path,
        usecols=["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "valuenum"],
        low_memory=False,
    )
    chartevents = chartevents[chartevents["itemid"].isin(wanted)].copy()
    chartevents = chartevents.dropna(subset=["valuenum"])
    chartevents["feature"] = chartevents["itemid"].map(feature_name)
    chartevents["charttime"] = pd.to_datetime(chartevents["charttime"])

    pivot = (
        chartevents.pivot_table(
            index=["subject_id", "hadm_id", "stay_id", "charttime"],
            columns="feature",
            values="valuenum",
            aggfunc="mean",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    icu = pd.read_csv(icu_path)
    icu["intime"] = pd.to_datetime(icu["intime"])
    icu["outtime"] = pd.to_datetime(icu["outtime"])
    data = pivot.merge(icu[["subject_id", "hadm_id", "stay_id", "intime", "outtime"]], on=["subject_id", "hadm_id", "stay_id"], how="inner")
    data["time_from_icu_h"] = (data["charttime"] - data["intime"]).dt.total_seconds() / 3600.0
    data = data[(data["time_from_icu_h"] >= 0) & (data["time_from_icu_h"] <= 24)].copy()
    data = data.sort_values(["stay_id", "charttime"])

    for col in ["HR", "SBP", "RR", "GCS_EYE", "GCS_VERBAL", "GCS_MOTOR"]:
        if col not in data:
            data[col] = np.nan
        data[col] = data.groupby("stay_id")[col].ffill()
        data[col] = data.groupby("stay_id")[col].bfill()

    data["GCS"] = data[["GCS_EYE", "GCS_VERBAL", "GCS_MOTOR"]].sum(axis=1, min_count=3)
    for col in ["HR", "SBP", "RR", "GCS"]:
        data[col] = data[col].fillna(data[col].median())

    data["RTS"] = data.apply(calc_rts, axis=1)
    data["shock_index"] = data["HR"] / data["SBP"].clip(lower=1)

    patients = pd.read_csv(patients_path)
    admissions = pd.read_csv(admissions_path)
    admissions["hospital_expire_flag"] = admissions["hospital_expire_flag"].fillna(0).astype(int)

    data = data.merge(patients[["subject_id", "gender", "anchor_age"]], on="subject_id", how="left")
    data = data.merge(admissions[["subject_id", "hadm_id", "hospital_expire_flag"]], on=["subject_id", "hadm_id"], how="left")
    data = data.merge(build_iss_proxy(base), on=["subject_id", "hadm_id"], how="left")
    data["injury_dx_count"] = data["injury_dx_count"].fillna(0)
    data["iss_proxy"] = data["iss_proxy"].fillna(1)

    # TRISS-inspired dynamic score. It is not presented as canonical TRISS,
    # because MIMIC demo does not provide AIS-derived ISS directly.
    z = (
        -0.75
        + 0.82 * data["RTS"]
        - 0.055 * data["iss_proxy"]
        - 0.025 * np.maximum(data["anchor_age"] - 55, 0)
        - 0.65 * data["shock_index"]
    )
    data["survival_t0"] = 1.0 / (1.0 + np.exp(-z))
    data["hazard_proxy"] = np.clip(
        0.025
        + 0.010 * data["iss_proxy"]
        + 0.18 * np.maximum(data["shock_index"] - 0.75, 0)
        - 0.012 * data["RTS"],
        0.015,
        1.0,
    )
    data["survival_6h"] = data["survival_t0"] * np.exp(-data["hazard_proxy"] * 6)
    data["survival_12h"] = data["survival_t0"] * np.exp(-data["hazard_proxy"] * 12)
    data["red_zone"] = (data["survival_6h"] < 0.50).astype(int)

    bins = [0, 2, 4, 6, 8, 12, 18, 24]
    labels = ["0-2", "2-4", "4-6", "6-8", "8-12", "12-18", "18-24"]
    data["time_bin"] = pd.cut(data["time_from_icu_h"], bins=bins, labels=labels, include_lowest=True, right=False)
    binned = (
        data.dropna(subset=["time_bin"])
        .groupby(["subject_id", "hadm_id", "stay_id", "time_bin"], observed=True)
        [["HR", "SBP", "RR", "GCS", "RTS", "shock_index", "survival_t0", "survival_6h", "red_zone"]]
        .mean()
        .reset_index()
    )

    keep_cols = [
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
        "RTS",
        "shock_index",
        "injury_dx_count",
        "iss_proxy",
        "hospital_expire_flag",
        "survival_t0",
        "hazard_proxy",
        "survival_6h",
        "survival_12h",
        "red_zone",
    ]
    return data[keep_cols], binned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("data/mimic-iv-demo/2.2"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    features, binned = build_features(args.base)
    features.to_csv(args.out / "mimic_dynamic_triss_features.csv", index=False)
    binned.to_csv(args.out / "mimic_patient_time_bins.csv", index=False)

    summary = pd.DataFrame(
        [
            {"metric": "rows", "value": len(features)},
            {"metric": "icu_stays", "value": features["stay_id"].nunique()},
            {"metric": "patients", "value": features["subject_id"].nunique()},
            {"metric": "hospital_mortality_events", "value": features.groupby("hadm_id")["hospital_expire_flag"].max().sum()},
            {"metric": "mean_rts", "value": features["RTS"].mean()},
            {"metric": "mean_shock_index", "value": features["shock_index"].mean()},
            {"metric": "mean_survival_6h", "value": features["survival_6h"].mean()},
            {"metric": "red_zone_rate", "value": features["red_zone"].mean()},
        ]
    )
    summary.to_csv(args.out / "mimic_feature_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved: {args.out / 'mimic_dynamic_triss_features.csv'}")
    print(f"Saved: {args.out / 'mimic_patient_time_bins.csv'}")


if __name__ == "__main__":
    main()

