from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
ED_ROOT = ROOT / "data" / "mimic-iv-ed-demo" / "mimic-iv-ed-demo-2.2" / "ed"
HOSP_ROOT = ROOT / "data" / "mimic-iv-demo" / "mimic-iv-clinical-database-demo-2.2" / "hosp"
OUT = ROOT / "results_external"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def is_injury_icd(code: object, version: object) -> bool:
    text = str(code).replace(".", "").upper()
    if not text or text == "NAN":
        return False
    try:
        ver = int(version)
    except (TypeError, ValueError):
        ver = 10 if text[0].isalpha() else 9
    if ver == 10:
        return text.startswith("S") or text.startswith("T")
    if not text[:3].isdigit():
        return False
    stem = int(text[:3])
    return 800 <= stem <= 959


def is_burn_icd(code: object, version: object) -> bool:
    text = str(code).replace(".", "").upper()
    if not text or text == "NAN":
        return False
    try:
        ver = int(version)
    except (TypeError, ValueError):
        ver = 10 if text[0].isalpha() else 9
    if ver == 10:
        if not text.startswith("T") or len(text) < 3 or not text[1:3].isdigit():
            return False
        return 20 <= int(text[1:3]) <= 32
    if not text[:3].isdigit():
        return False
    stem = int(text[:3])
    return 940 <= stem <= 949


def gcs_proxy_from_acuity(acuity: pd.Series) -> pd.Series:
    acuity = numeric(acuity).clip(1, 5)
    return acuity.map({1.0: 11.0, 2.0: 13.0, 3.0: 14.0, 4.0: 15.0, 5.0: 15.0}).fillna(15.0)


def vital_score(df: pd.DataFrame) -> pd.Series:
    gcs = (df["GCS"] / 15.0).clip(0, 1)
    sbp = ((df["SBP"] - 60.0) / 80.0).clip(0, 1)
    rr = (1.0 - ((df["RR"] - 20.0).abs() / 22.0)).clip(0, 1)
    shock = (1.0 - ((df["shock_index"] - 0.55) / 1.45)).clip(0, 1)
    return (0.35 * gcs + 0.25 * sbp + 0.20 * rr + 0.20 * shock).clip(0, 1)


def build_features() -> pd.DataFrame:
    edstays = read_csv(ED_ROOT / "edstays.csv.gz")
    triage = read_csv(ED_ROOT / "triage.csv.gz")
    vitals = read_csv(ED_ROOT / "vitalsign.csv.gz")
    ed_dx = read_csv(ED_ROOT / "diagnosis.csv.gz")
    patients = read_csv(HOSP_ROOT / "patients.csv.gz")[["subject_id", "anchor_age"]]
    admissions = read_csv(HOSP_ROOT / "admissions.csv.gz")[["subject_id", "hadm_id", "hospital_expire_flag"]]
    hosp_dx = read_csv(HOSP_ROOT / "diagnoses_icd.csv.gz")

    vitals_agg = (
        vitals.assign(charttime=pd.to_datetime(vitals["charttime"], errors="coerce"))
        .sort_values("charttime")
        .groupby("stay_id")
        .agg(
            HR_vitals=("heartrate", "median"),
            RR_vitals=("resprate", "median"),
            SBP_vitals=("sbp", "median"),
            DBP_vitals=("dbp", "median"),
            O2_vitals=("o2sat", "median"),
        )
        .reset_index()
    )

    ed_dx = ed_dx.copy()
    ed_dx["is_injury"] = [is_injury_icd(c, v) for c, v in zip(ed_dx["icd_code"], ed_dx["icd_version"])]
    ed_dx["is_burn"] = [is_burn_icd(c, v) for c, v in zip(ed_dx["icd_code"], ed_dx["icd_version"])]
    ed_dx_agg = (
        ed_dx.groupby("stay_id")
        .agg(injury_dx_count_ed=("is_injury", "sum"), burn_dx_count_ed=("is_burn", "sum"))
        .reset_index()
    )

    hosp_dx = hosp_dx.copy()
    hosp_dx["is_injury"] = [is_injury_icd(c, v) for c, v in zip(hosp_dx["icd_code"], hosp_dx["icd_version"])]
    hosp_dx["is_burn"] = [is_burn_icd(c, v) for c, v in zip(hosp_dx["icd_code"], hosp_dx["icd_version"])]
    hosp_dx_agg = (
        hosp_dx.groupby("hadm_id")
        .agg(injury_dx_count_hosp=("is_injury", "sum"), burn_dx_count_hosp=("is_burn", "sum"))
        .reset_index()
    )

    df = (
        edstays.merge(triage, on=["subject_id", "stay_id"], how="left", suffixes=("", "_triage"))
        .merge(vitals_agg, on="stay_id", how="left")
        .merge(ed_dx_agg, on="stay_id", how="left")
        .merge(hosp_dx_agg, on="hadm_id", how="left")
        .merge(patients, on="subject_id", how="left")
        .merge(admissions, on=["subject_id", "hadm_id"], how="left")
    )

    df["HR"] = numeric(df["heartrate"]).fillna(numeric(df["HR_vitals"]))
    df["RR"] = numeric(df["resprate"]).fillna(numeric(df["RR_vitals"]))
    df["SBP"] = numeric(df["sbp"]).fillna(numeric(df["SBP_vitals"]))
    df["DBP"] = numeric(df["dbp"]).fillna(numeric(df["DBP_vitals"]))
    df["O2Sat"] = numeric(df["o2sat"]).fillna(numeric(df["O2_vitals"]))
    df["GCS"] = gcs_proxy_from_acuity(df["acuity"])

    df["HR"] = df["HR"].fillna(df["HR"].median()).clip(35, 220)
    df["RR"] = df["RR"].fillna(df["RR"].median()).clip(6, 60)
    df["SBP"] = df["SBP"].fillna(df["SBP"].median()).clip(50, 240)
    df["DBP"] = df["DBP"].fillna(df["DBP"].median()).clip(20, 160)
    df["O2Sat"] = df["O2Sat"].fillna(df["O2Sat"].median()).clip(50, 100)
    df["anchor_age"] = numeric(df["anchor_age"]).fillna(numeric(df["anchor_age"]).median()).clip(18, 91)
    df["acuity"] = numeric(df["acuity"]).fillna(3).clip(1, 5)

    df["shock_index"] = (df["HR"] / df["SBP"]).replace([np.inf, -np.inf], np.nan).fillna(0.75).clip(0.25, 2.5)
    df["vital_score"] = vital_score(df)
    df["injury_dx_count"] = (
        numeric(df["injury_dx_count_ed"]).fillna(0) + numeric(df["injury_dx_count_hosp"]).fillna(0)
    ).clip(0, 12)
    df["burn_dx_count"] = (
        numeric(df["burn_dx_count_ed"]).fillna(0) + numeric(df["burn_dx_count_hosp"]).fillna(0)
    ).clip(0, 8)
    df["iss_proxy"] = np.clip(1 + 4.5 * df["injury_dx_count"] + 5.0 * df["burn_dx_count"], 1, 50)

    acuity_severity = ((6 - df["acuity"]) / 5.0).clip(0, 1)
    z = (
        0.60
        + 3.70 * df["vital_score"]
        - 0.035 * df["iss_proxy"]
        - 0.014 * np.maximum(df["anchor_age"] - 55, 0)
        - 0.30 * df["shock_index"]
        - 0.35 * acuity_severity
    )
    df["survival_t0"] = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
    df["hazard_proxy"] = np.clip(
        0.012
        + 0.0045 * df["iss_proxy"]
        + 0.08 * np.maximum(df["shock_index"] - 0.75, 0)
        + 0.05 * (1 - df["vital_score"])
        + 0.025 * acuity_severity,
        0.015,
        1.0,
    )
    df["survival_6h"] = df["survival_t0"] * np.exp(-df["hazard_proxy"] * 6)
    df["red_zone"] = (df["survival_6h"] < 0.50).astype(int)
    df["time_from_icu_h"] = 0.0
    df["charttime"] = df["intime"]
    df["hospital_expire_flag"] = numeric(df["hospital_expire_flag"]).fillna((df["disposition"] == "EXPIRED").astype(int))

    keep = [
        "subject_id",
        "hadm_id",
        "stay_id",
        "charttime",
        "time_from_icu_h",
        "anchor_age",
        "gender",
        "race",
        "arrival_transport",
        "disposition",
        "acuity",
        "HR",
        "SBP",
        "DBP",
        "RR",
        "O2Sat",
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
    return df[keep].sort_values("stay_id").reset_index(drop=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    features = build_features()
    features.to_csv(OUT / "mimic_iv_ed_demo_features.csv", index=False)
    summary = pd.DataFrame(
        [
            {"metric": "rows", "value": len(features)},
            {"metric": "ed_stays", "value": features["stay_id"].nunique()},
            {"metric": "patients", "value": features["subject_id"].nunique()},
            {"metric": "admitted_or_transferred", "value": features["disposition"].isin(["ADMITTED", "TRANSFER"]).sum()},
            {"metric": "expired_in_ed_or_hospital", "value": features["hospital_expire_flag"].sum()},
            {"metric": "injury_dx_rows", "value": (features["injury_dx_count"] > 0).sum()},
            {"metric": "mean_vital_score", "value": features["vital_score"].mean()},
            {"metric": "mean_survival_6h", "value": features["survival_6h"].mean()},
            {"metric": "red_zone_rate", "value": features["red_zone"].mean()},
        ]
    )
    summary.to_csv(OUT / "mimic_iv_ed_demo_feature_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
