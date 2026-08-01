from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
EICU_ROOT = ROOT / "data" / "eicu-crd-demo" / "eicu-collaborative-research-database-demo-2.0.1"
OUT = ROOT / "results_external"

INJURY_PATTERN = re.compile(r"trauma|fracture|burn|laceration|injur|accident|fall|wound|crush|contusion", re.I)
BURN_PATTERN = re.compile(r"burn", re.I)


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(EICU_ROOT / name)


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def parse_age(value: object) -> float:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return np.nan
    if ">" in text:
        return 90.0
    return pd.to_numeric(text, errors="coerce")


def vital_score(df: pd.DataFrame) -> pd.Series:
    gcs = (df["GCS"] / 15.0).clip(0, 1)
    sbp = ((df["SBP"] - 60.0) / 80.0).clip(0, 1)
    rr = (1.0 - ((df["RR"] - 20.0).abs() / 22.0)).clip(0, 1)
    shock = (1.0 - ((df["shock_index"] - 0.55) / 1.45)).clip(0, 1)
    return (0.35 * gcs + 0.25 * sbp + 0.20 * rr + 0.20 * shock).clip(0, 1)


def build_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    patient = read_csv("patient.csv.gz")
    hospital = read_csv("hospital.csv.gz")
    apache = read_csv("apachePatientResult.csv.gz")
    aps = read_csv("apacheApsVar.csv.gz")
    adm_dx = read_csv("admissionDx.csv.gz")
    vital = read_csv("vitalPeriodic.csv.gz")

    apache = (
        apache.sort_values(["patientunitstayid", "apacheversion"])
        .groupby("patientunitstayid")
        .tail(1)
        .reset_index(drop=True)
    )

    vital_early = vital[(numeric(vital["observationoffset"]) >= 0) & (numeric(vital["observationoffset"]) <= 360)].copy()
    vital_agg = (
        vital_early.groupby("patientunitstayid")
        .agg(
            HR_vital=("heartrate", "median"),
            RR_vital=("respiration", "median"),
            SBP_vital=("systemicsystolic", "median"),
            DBP_vital=("systemicdiastolic", "median"),
            meanbp_vital=("systemicmean", "median"),
            sao2_vital=("sao2", "median"),
            temp_vital=("temperature", "median"),
        )
        .reset_index()
    )

    dx_text = adm_dx.copy()
    for col in ["admitdxpath", "admitdxname", "admitdxtext"]:
        dx_text[col] = dx_text[col].fillna("")
    dx_text["dx_joined"] = dx_text["admitdxpath"] + " " + dx_text["admitdxname"] + " " + dx_text["admitdxtext"]
    dx_text["is_injury"] = dx_text["dx_joined"].str.contains(INJURY_PATTERN)
    dx_text["is_burn"] = dx_text["dx_joined"].str.contains(BURN_PATTERN)
    dx_agg = (
        dx_text.groupby("patientunitstayid")
        .agg(injury_dx_count=("is_injury", "sum"), burn_dx_count=("is_burn", "sum"))
        .reset_index()
    )

    df = (
        patient.merge(hospital, on="hospitalid", how="left")
        .merge(apache, on="patientunitstayid", how="left")
        .merge(aps, on="patientunitstayid", how="left")
        .merge(vital_agg, on="patientunitstayid", how="left")
        .merge(dx_agg, on="patientunitstayid", how="left")
    )

    df["anchor_age"] = df["age"].map(parse_age)
    df["HR"] = numeric(df["heartrate"]).fillna(numeric(df["HR_vital"]))
    df["RR"] = numeric(df["respiratoryrate"]).fillna(numeric(df["RR_vital"]))
    df["SBP"] = numeric(df["SBP_vital"])
    estimated_sbp = numeric(df["meanbp"]).fillna(numeric(df["meanbp_vital"])) * 1.35
    df["SBP"] = df["SBP"].fillna(estimated_sbp)
    df["DBP"] = numeric(df["DBP_vital"])
    df["GCS"] = (
        numeric(df["eyes"]).clip(1, 4).fillna(4)
        + numeric(df["motor"]).clip(1, 6).fillna(6)
        + numeric(df["verbal"]).clip(1, 5).fillna(5)
    )

    df["anchor_age"] = df["anchor_age"].fillna(df["anchor_age"].median()).clip(18, 91)
    df["HR"] = df["HR"].fillna(df["HR"].median()).clip(35, 220)
    df["RR"] = df["RR"].fillna(df["RR"].median()).clip(6, 60)
    df["SBP"] = df["SBP"].fillna(df["SBP"].median()).clip(50, 240)
    df["DBP"] = df["DBP"].fillna(df["DBP"].median()).clip(20, 160)
    df["GCS"] = df["GCS"].fillna(15).clip(3, 15)
    df["shock_index"] = (df["HR"] / df["SBP"]).replace([np.inf, -np.inf], np.nan).fillna(0.75).clip(0.25, 2.5)
    df["vital_score"] = vital_score(df)

    df["injury_dx_count"] = numeric(df["injury_dx_count"]).fillna(0).clip(0, 12)
    df["burn_dx_count"] = numeric(df["burn_dx_count"]).fillna(0).clip(0, 8)
    apache_score = numeric(df["apachescore"]).fillna(numeric(df["apachescore"]).median()).clip(0, 200)
    df["iss_proxy"] = np.clip(1 + 4.0 * df["injury_dx_count"] + 5.0 * df["burn_dx_count"] + apache_score / 12.0, 1, 50)

    predicted_mortality = numeric(df["predictedhospitalmortality"]).clip(0.001, 0.95)
    fallback_logit = (
        -4.0
        + 0.035 * apache_score
        + 0.55 * np.maximum(df["shock_index"] - 0.85, 0)
        + 0.03 * np.maximum(df["anchor_age"] - 65, 0)
        + 0.60 * (1 - df["vital_score"])
    )
    fallback_mortality = 1.0 / (1.0 + np.exp(-np.clip(fallback_logit, -50, 50)))
    predicted_mortality = predicted_mortality.fillna(fallback_mortality).clip(0.001, 0.95)
    df["survival_t0"] = (1.0 - predicted_mortality).clip(0.02, 0.995)
    df["hazard_proxy"] = np.clip(
        0.012
        + 0.0025 * apache_score
        + 0.06 * np.maximum(df["shock_index"] - 0.75, 0)
        + 0.045 * (1 - df["vital_score"])
        + 0.0035 * df["injury_dx_count"],
        0.015,
        1.0,
    )
    df["survival_6h"] = df["survival_t0"] * np.exp(-df["hazard_proxy"] * 6)
    df["red_zone"] = (df["survival_6h"] < 0.50).astype(int)
    df["hospital_expire_flag"] = (df["actualhospitalmortality"].astype(str).str.lower() == "expired").astype(int)
    df["time_from_icu_h"] = 0.0
    df["charttime"] = "0"

    keep = [
        "patientunitstayid",
        "patienthealthsystemstayid",
        "hospitalid",
        "wardid",
        "charttime",
        "time_from_icu_h",
        "anchor_age",
        "gender",
        "ethnicity",
        "unittype",
        "apacheadmissiondx",
        "HR",
        "SBP",
        "DBP",
        "RR",
        "GCS",
        "shock_index",
        "vital_score",
        "injury_dx_count",
        "burn_dx_count",
        "iss_proxy",
        "hospital_expire_flag",
        "survival_t0",
        "hazard_proxy",
        "survival_6h",
        "red_zone",
    ]
    features = df[keep].rename(
        columns={
            "patientunitstayid": "stay_id",
            "patienthealthsystemstayid": "hadm_id",
            "hospitalid": "subject_id",
        }
    )
    return features.sort_values("stay_id").reset_index(drop=True), hospital


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    features, hospital = build_features()
    features.to_csv(OUT / "eicu_demo_features.csv", index=False)
    hospital.to_csv(OUT / "eicu_demo_hospitals.csv", index=False)
    summary = pd.DataFrame(
        [
            {"metric": "rows", "value": len(features)},
            {"metric": "icu_stays", "value": features["stay_id"].nunique()},
            {"metric": "hospitals", "value": features["subject_id"].nunique()},
            {"metric": "expired_hospital", "value": features["hospital_expire_flag"].sum()},
            {"metric": "injury_dx_rows", "value": (features["injury_dx_count"] > 0).sum()},
            {"metric": "mean_vital_score", "value": features["vital_score"].mean()},
            {"metric": "mean_survival_6h", "value": features["survival_6h"].mean()},
            {"metric": "red_zone_rate", "value": features["red_zone"].mean()},
        ]
    )
    summary.to_csv(OUT / "eicu_demo_feature_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
