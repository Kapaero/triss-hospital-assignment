"""
Discrete-choice experiment for probabilistic patient-to-hospital routing on eICU.

The model learns P(hospital | patient, candidate hospitals) from observed eICU
hospital assignments.  For each patient, the observed hospital is the positive
alternative and all other hospitals form the competing alternatives.

The main model is a conditional logit:

    p_ij = exp(theta^T z_ij) / sum_k exp(theta^T z_ik)

where z_ij contains pairwise clinical/profile/transport/capacity features.

Run:
  python eicu_discrete_choice_experiment.py
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_FEATURES = ROOT / "results_external" / "eicu_demo_features.csv"
DEFAULT_HOSPITALS = ROOT / "results_external" / "eicu_demo_hospitals.csv"
OUT = ROOT / "results_eicu_choice"
BASE_SEED = 20260628

NEED_CATEGORIES = (
    "trauma",
    "burn",
    "cardiac",
    "neuro",
    "respiratory",
    "sepsis",
    "gi",
    "surgical",
    "toxicology",
    "general",
)

UNIT_TYPES = (
    "Med-Surg ICU",
    "MICU",
    "SICU",
    "Cardiac ICU",
    "CCU-CTICU",
    "Neuro ICU",
    "CSICU",
    "CTICU",
)

FEATURE_NAMES = (
    "profile_match",
    "unit_profile_match",
    "transport_score",
    "travel_minutes_scaled",
    "capacity_norm",
    "free_capacity_proxy",
    "load_proxy",
    "quality_proxy",
    "teaching_status",
    "historical_volume_norm",
    "severe_profile_match",
    "severe_transport_score",
    "red_transport_score",
    "red_quality_proxy",
    "burn_profile_match",
    "trauma_profile_match",
    "neuro_profile_match",
    "cardiac_profile_match",
)


@dataclass
class ChoiceData:
    patients: pd.DataFrame
    hospitals: pd.DataFrame
    hospital_ids: np.ndarray
    actual_index: np.ndarray
    need_matrix: np.ndarray
    unit_matrix: np.ndarray
    travel_minutes: np.ndarray
    pair_features: np.ndarray
    feature_names: tuple[str, ...]


def softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    ex = np.exp(np.clip(shifted, -60, 60))
    return ex / ex.sum(axis=1, keepdims=True)


def normalize01(values: np.ndarray) -> np.ndarray:
    values = values.astype(float)
    lo = np.nanmin(values)
    hi = np.nanmax(values)
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - lo) / (hi - lo)


def infer_need_matrix(df: pd.DataFrame) -> np.ndarray:
    text = df["apacheadmissiondx"].fillna("").astype(str).str.lower()
    need = pd.DataFrame(index=df.index)
    need["trauma"] = (
        (pd.to_numeric(df["injury_dx_count"], errors="coerce").fillna(0) > 0)
        | text.str.contains(r"trauma|fracture|fall|injur|laceration|contusion|crush", regex=True)
    )
    need["burn"] = (
        (pd.to_numeric(df["burn_dx_count"], errors="coerce").fillna(0) > 0)
        | text.str.contains(r"burn", regex=True)
    )
    need["cardiac"] = text.str.contains(
        r"cardiac|heart|myocard|infarction|rhythm|coronary|angina|cardiogenic|arrhythm", regex=True
    )
    need["neuro"] = text.str.contains(
        r"coma|consciousness|stroke|seizure|neuro|intracranial|head|cerebral", regex=True
    )
    need["respiratory"] = text.str.contains(
        r"respiratory|pulmonary|pneumonia|asthma|copd|ventilat|hypox|airway", regex=True
    )
    need["sepsis"] = text.str.contains(r"sepsis|septic", regex=True)
    need["gi"] = text.str.contains(
        r"\bgi\b|gastro|abdomen|abdominal|bleeding|perforation|rupture|pancrea|liver", regex=True
    )
    need["surgical"] = text.str.contains(
        r"surgery|surgical|post[- ]?op|operative|replacement|transplant|bypass|resection", regex=True
    )
    need["toxicology"] = text.str.contains(r"overdose|toxin|poison|drug", regex=True)
    matched = need.any(axis=1)
    need["general"] = ~matched
    arr = need[list(NEED_CATEGORIES)].astype(float).to_numpy()
    return arr


def infer_unit_matrix(df: pd.DataFrame) -> np.ndarray:
    unit = df["unittype"].fillna("Med-Surg ICU").astype(str)
    mat = np.zeros((len(df), len(UNIT_TYPES)), dtype=float)
    unit_to_idx = {name: idx for idx, name in enumerate(UNIT_TYPES)}
    for row, value in enumerate(unit):
        idx = unit_to_idx.get(value)
        if idx is not None:
            mat[row, idx] = 1.0
    empty = mat.sum(axis=1) == 0
    mat[empty, unit_to_idx["Med-Surg ICU"]] = 1.0
    return mat


def bed_midpoint(category: object) -> float:
    text = str(category)
    if text == "<100":
        return 75.0
    if text == "100 - 249":
        return 175.0
    if text == "250 - 499":
        return 375.0
    if text == ">= 500":
        return 650.0
    return np.nan


def hospital_coordinates(hospitals: pd.DataFrame, rng: np.random.Generator) -> dict[int, tuple[float, float]]:
    region_centers = {
        "Midwest": (0.0, 0.0),
        "South": (380.0, -160.0),
        "West": (-520.0, 70.0),
        "Northeast": (460.0, 260.0),
    }
    coords: dict[int, tuple[float, float]] = {}
    for row in hospitals.itertuples(index=False):
        hospital_id = int(row.hospitalid)
        center = region_centers.get(str(row.region), (40.0, 230.0))
        x = center[0] + rng.normal(0, 85)
        y = center[1] + rng.normal(0, 85)
        coords[hospital_id] = (float(x), float(y))
    return coords


def patient_coordinates(
    patients: pd.DataFrame,
    hospital_coords: dict[int, tuple[float, float]],
    rng: np.random.Generator,
    catchment_sigma: float,
) -> np.ndarray:
    coords = np.zeros((len(patients), 2), dtype=float)
    for row_idx, hospital_id in enumerate(patients["subject_id"].astype(int).to_numpy()):
        hx, hy = hospital_coords[hospital_id]
        coords[row_idx, 0] = hx + rng.normal(0, catchment_sigma)
        coords[row_idx, 1] = hy + rng.normal(0, catchment_sigma)
    return coords


def stratified_split(df: pd.DataFrame, label_col: str, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for _, idx in df.groupby(label_col).groups.items():
        idx_arr = np.array(list(idx), dtype=int)
        rng.shuffle(idx_arr)
        n_test = max(1, int(round(len(idx_arr) * test_fraction)))
        n_test = min(n_test, len(idx_arr) - 1)
        test_parts.append(idx_arr[:n_test])
        train_parts.append(idx_arr[n_test:])
    train_idx = np.concatenate(train_parts)
    test_idx = np.concatenate(test_parts)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def build_hospital_profiles(
    train: pd.DataFrame,
    hospital_meta: pd.DataFrame,
    hospital_ids: np.ndarray,
    train_need: np.ndarray,
    train_unit: np.ndarray,
    alpha: float = 8.0,
) -> pd.DataFrame:
    id_to_row = {int(hid): idx for idx, hid in enumerate(hospital_ids)}
    rows: list[dict[str, float | int | str]] = []
    global_need = train_need.mean(axis=0)
    global_unit = train_unit.mean(axis=0)
    global_death = train["hospital_expire_flag"].astype(float).mean()
    global_survival = 1.0 - global_death
    hospital_meta = hospital_meta.copy()
    hospital_meta["bed_midpoint"] = hospital_meta["numbedscategory"].map(bed_midpoint)
    median_beds = float(hospital_meta["bed_midpoint"].median())
    hospital_meta["bed_midpoint"] = hospital_meta["bed_midpoint"].fillna(median_beds)
    meta_map = hospital_meta.set_index("hospitalid")
    train_counts = train["subject_id"].astype(int).value_counts().to_dict()
    max_count = max(train_counts.values()) if train_counts else 1
    max_beds = float(hospital_meta["bed_midpoint"].max())

    for hospital_id in hospital_ids:
        mask = train["subject_id"].astype(int).to_numpy() == int(hospital_id)
        count = int(mask.sum())
        need_sum = train_need[mask].sum(axis=0) if count else np.zeros(len(NEED_CATEGORIES))
        unit_sum = train_unit[mask].sum(axis=0) if count else np.zeros(len(UNIT_TYPES))
        death_sum = float(train.loc[mask, "hospital_expire_flag"].sum()) if count else 0.0
        need_profile = (need_sum + alpha * global_need) / max(count + alpha, 1e-9)
        unit_profile = (unit_sum + alpha * global_unit) / max(count + alpha, 1e-9)
        survival_quality = 1.0 - ((death_sum + alpha * (1.0 - global_survival)) / max(count + alpha, 1e-9))
        meta = meta_map.loc[int(hospital_id)] if int(hospital_id) in meta_map.index else None
        beds = float(meta["bed_midpoint"]) if meta is not None else median_beds
        teaching = 1.0 if meta is not None and str(meta["teachingstatus"]).lower() == "t" else 0.0
        region = str(meta["region"]) if meta is not None else "unknown"
        volume_norm = math.log1p(count) / math.log1p(max_count)
        capacity_norm = beds / max_beds
        load_raw = count / max(beds, 1.0)
        rows.append(
            {
                "hospital_id": int(hospital_id),
                "train_count": count,
                "bed_midpoint": beds,
                "capacity_norm": capacity_norm,
                "teaching_status": teaching,
                "region": region,
                "historical_volume_norm": volume_norm,
                "load_raw": load_raw,
                "quality_proxy": survival_quality,
                **{f"need_{name}": float(need_profile[idx]) for idx, name in enumerate(NEED_CATEGORIES)},
                **{f"unit_{name}": float(unit_profile[idx]) for idx, name in enumerate(UNIT_TYPES)},
            }
        )

    profiles = pd.DataFrame(rows)
    profiles["load_proxy"] = normalize01(profiles["load_raw"].to_numpy())
    profiles["free_capacity_proxy"] = (profiles["capacity_norm"] * (1.0 - profiles["load_proxy"])).clip(0, 1)
    return profiles


def build_choice_data(
    patients: pd.DataFrame,
    hospital_meta: pd.DataFrame,
    profile_train: pd.DataFrame,
    full_need: np.ndarray,
    full_unit: np.ndarray,
    hospital_ids: np.ndarray,
    hospital_coords: dict[int, tuple[float, float]],
    seed: int,
    catchment_sigma: float,
) -> ChoiceData:
    rng = np.random.default_rng(seed)
    patients = patients.reset_index(drop=True).copy()
    actual_ids = patients["subject_id"].astype(int).to_numpy()
    id_to_alt = {int(hid): idx for idx, hid in enumerate(hospital_ids)}
    actual_index = np.array([id_to_alt[int(hid)] for hid in actual_ids], dtype=int)
    need = full_need[patients["_source_index"].to_numpy()]
    unit = full_unit[patients["_source_index"].to_numpy()]
    patient_xy = patient_coordinates(patients, hospital_coords, rng, catchment_sigma)
    hospital_xy = np.array([hospital_coords[int(hid)] for hid in hospital_ids], dtype=float)
    distance_km = np.sqrt(((patient_xy[:, None, :] - hospital_xy[None, :, :]) ** 2).sum(axis=2))
    travel_minutes = np.clip(8.0 + distance_km / 0.85 + rng.normal(0, 2.5, size=distance_km.shape), 3.0, None)

    hp = profile_train.set_index("hospital_id").loc[hospital_ids].reset_index()
    hospital_need = hp[[f"need_{name}" for name in NEED_CATEGORIES]].to_numpy(float)
    hospital_unit = hp[[f"unit_{name}" for name in UNIT_TYPES]].to_numpy(float)
    profile_match = need @ hospital_need.T
    unit_match = unit @ hospital_unit.T
    transport_score = np.exp(-travel_minutes / 55.0)
    travel_scaled = travel_minutes / 120.0
    capacity = np.repeat(hp["capacity_norm"].to_numpy(float)[None, :], len(patients), axis=0)
    free_capacity = np.repeat(hp["free_capacity_proxy"].to_numpy(float)[None, :], len(patients), axis=0)
    load = np.repeat(hp["load_proxy"].to_numpy(float)[None, :], len(patients), axis=0)
    quality = np.repeat(hp["quality_proxy"].to_numpy(float)[None, :], len(patients), axis=0)
    teaching = np.repeat(hp["teaching_status"].to_numpy(float)[None, :], len(patients), axis=0)
    volume = np.repeat(hp["historical_volume_norm"].to_numpy(float)[None, :], len(patients), axis=0)

    severity = (
        0.55 * (1.0 - patients["survival_t0"].astype(float).to_numpy())
        + 0.30 * (patients["iss_proxy"].astype(float).to_numpy() / 50.0)
        + 0.15 * normalize01(patients["shock_index"].astype(float).to_numpy())
    ).clip(0, 1)
    severity = severity[:, None]
    red = patients["red_zone"].astype(float).to_numpy()[:, None]

    cat_index = {name: idx for idx, name in enumerate(NEED_CATEGORIES)}
    burn_profile = np.repeat(need[:, cat_index["burn"]][:, None], len(hospital_ids), axis=1) * hospital_need[:, cat_index["burn"]]
    trauma_profile = np.repeat(need[:, cat_index["trauma"]][:, None], len(hospital_ids), axis=1) * hospital_need[:, cat_index["trauma"]]
    neuro_profile = np.repeat(need[:, cat_index["neuro"]][:, None], len(hospital_ids), axis=1) * hospital_need[:, cat_index["neuro"]]
    cardiac_profile = np.repeat(need[:, cat_index["cardiac"]][:, None], len(hospital_ids), axis=1) * hospital_need[:, cat_index["cardiac"]]

    features = np.stack(
        [
            profile_match,
            unit_match,
            transport_score,
            travel_scaled,
            capacity,
            free_capacity,
            load,
            quality,
            teaching,
            volume,
            severity * profile_match,
            severity * transport_score,
            red * transport_score,
            red * quality,
            burn_profile,
            trauma_profile,
            neuro_profile,
            cardiac_profile,
        ],
        axis=2,
    ).astype(np.float64)

    return ChoiceData(
        patients=patients,
        hospitals=hp,
        hospital_ids=hospital_ids,
        actual_index=actual_index,
        need_matrix=need,
        unit_matrix=unit,
        travel_minutes=travel_minutes,
        pair_features=features,
        feature_names=FEATURE_NAMES,
    )


def standardize(train_x: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    mean = train_x.reshape(-1, train_x.shape[-1]).mean(axis=0)
    std = train_x.reshape(-1, train_x.shape[-1]).std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    train_z = (train_x - mean) / std
    other_z = [(x - mean) / std for x in others]
    return train_z, other_z, mean, std


def fit_conditional_logit(
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    learning_rate: float,
    l2: float,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    rng = np.random.default_rng(seed)
    theta = rng.normal(0, 0.01, size=x.shape[2])
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    history: list[dict[str, float]] = []
    rows = np.arange(len(y))
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    for epoch in range(1, epochs + 1):
        scores = np.einsum("ijf,f->ij", x, theta)
        probs = softmax(scores)
        chosen_prob = np.clip(probs[rows, y], 1e-12, 1.0)
        nll = float(-np.log(chosen_prob).mean() + 0.5 * l2 * float(theta @ theta))
        expected = np.einsum("ij,ijf->f", probs, x) / len(y)
        chosen = x[rows, y, :].mean(axis=0)
        grad = expected - chosen + l2 * theta
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad * grad)
        m_hat = m / (1 - beta1**epoch)
        v_hat = v / (1 - beta2**epoch)
        theta -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)
        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            history.append({"epoch": epoch, "train_nll": nll, "grad_norm": float(np.linalg.norm(grad))})
    return theta, history


def metrics_from_scores(scores: np.ndarray, y: np.ndarray, name: str) -> dict[str, float | str]:
    probs = softmax(scores)
    rows = np.arange(len(y))
    chosen_prob = np.clip(probs[rows, y], 1e-12, 1.0)
    order = np.argsort(-scores, axis=1)
    ranks = np.empty(len(y), dtype=int)
    for i in range(len(y)):
        ranks[i] = int(np.where(order[i] == y[i])[0][0]) + 1
    chosen_scores = scores[rows, y]
    pair_auc = ((scores < chosen_scores[:, None]).sum(axis=1) - 1) / np.maximum(scores.shape[1] - 1, 1)
    ties = ((scores == chosen_scores[:, None]).sum(axis=1) - 1) / np.maximum(scores.shape[1] - 1, 1)
    pair_auc = pair_auc + 0.5 * ties
    return {
        "model": name,
        "n_patients": int(len(y)),
        "n_hospitals": int(scores.shape[1]),
        "cross_entropy": float(-np.log(chosen_prob).mean()),
        "mean_true_hospital_probability": float(chosen_prob.mean()),
        "top1_accuracy": float((ranks <= 1).mean()),
        "top3_accuracy": float((ranks <= 3).mean()),
        "top5_accuracy": float((ranks <= 5).mean()),
        "top10_accuracy": float((ranks <= 10).mean()),
        "mean_reciprocal_rank": float((1.0 / ranks).mean()),
        "mean_rank": float(ranks.mean()),
        "pairwise_auc": float(pair_auc.mean()),
    }


def acceptable_sets(choice: ChoiceData) -> dict[str, np.ndarray]:
    profile = choice.pair_features[:, :, 0]
    unit = choice.pair_features[:, :, 1]
    travel = choice.travel_minutes
    free_capacity = choice.pair_features[:, :, 5]

    profile_norm = (profile - profile.min(axis=1, keepdims=True)) / np.maximum(
        profile.max(axis=1, keepdims=True) - profile.min(axis=1, keepdims=True),
        1e-9,
    )
    unit_norm = (unit - unit.min(axis=1, keepdims=True)) / np.maximum(
        unit.max(axis=1, keepdims=True) - unit.min(axis=1, keepdims=True),
        1e-9,
    )
    clinical_score = 0.5 * profile_norm + 0.5 * unit_norm
    clinical_threshold = np.quantile(clinical_score, 0.90, axis=1, keepdims=True)
    clinical_core = clinical_score >= clinical_threshold

    travel_threshold = np.quantile(travel, 0.40, axis=1, keepdims=True)
    capacity_threshold = np.quantile(free_capacity, 0.20, axis=1, keepdims=True)
    feasible = clinical_core & (travel <= travel_threshold) & (free_capacity >= capacity_threshold)

    # Guard against empty feasible sets in rare edge cases.
    empty = feasible.sum(axis=1) == 0
    if np.any(empty):
        feasible[empty] = clinical_core[empty]

    return {
        "clinical_core_top10pct": clinical_core,
        "clinical_transport_capacity_core": feasible,
    }


def probability_list_metrics(choice: ChoiceData, scores: np.ndarray, model_name: str) -> pd.DataFrame:
    probs = softmax(scores)
    order = np.argsort(-scores, axis=1)
    rows = np.arange(len(choice.actual_index))
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)
    normalized_entropy = entropy / math.log(scores.shape[1])
    effective_hospitals = np.exp(entropy)
    profile = choice.pair_features[:, :, 0]
    unit = choice.pair_features[:, :, 1]
    travel = choice.travel_minutes
    top1 = order[:, 0]

    records: list[dict[str, float | str]] = [
        {
            "model": model_name,
            "metric_group": "probability_shape",
            "mean_max_probability": float(probs.max(axis=1).mean()),
            "mean_normalized_entropy": float(normalized_entropy.mean()),
            "mean_effective_hospitals": float(effective_hospitals.mean()),
            "mean_hospitals_probability_ge_001": float((probs >= 0.01).sum(axis=1).mean()),
            "mean_hospitals_probability_ge_005": float((probs >= 0.05).sum(axis=1).mean()),
            "mean_expected_profile_match": float((probs * profile).sum(axis=1).mean()),
            "mean_expected_unit_match": float((probs * unit).sum(axis=1).mean()),
            "mean_expected_travel_minutes": float((probs * travel).sum(axis=1).mean()),
            "mean_top1_profile_match": float(profile[rows, top1].mean()),
            "mean_top1_unit_match": float(unit[rows, top1].mean()),
            "mean_top1_travel_minutes": float(travel[rows, top1].mean()),
            "mean_profile_regret_top1": float((profile.max(axis=1) - profile[rows, top1]).mean()),
            "mean_unit_regret_top1": float((unit.max(axis=1) - unit[rows, top1]).mean()),
            "mean_travel_regret_top1": float((travel[rows, top1] - travel.min(axis=1)).mean()),
        }
    ]

    for set_name, mask in acceptable_sets(choice).items():
        set_size = mask.sum(axis=1)
        first_ranks = []
        for i in range(len(mask)):
            hit_positions = np.flatnonzero(mask[i, order[i]])
            first_ranks.append(int(hit_positions[0]) + 1 if len(hit_positions) else scores.shape[1] + 1)
        first_ranks_arr = np.array(first_ranks, dtype=float)
        base: dict[str, float | str] = {
            "model": model_name,
            "metric_group": set_name,
            "mean_acceptable_set_size": float(set_size.mean()),
            "mean_probability_mass_in_set": float((probs * mask).sum(axis=1).mean()),
            "mean_first_acceptable_rank": float(first_ranks_arr.mean()),
            "top1_contains_acceptable": float(mask[rows, top1].mean()),
            "top3_contains_acceptable": float(mask[rows[:, None], order[:, :3]].any(axis=1).mean()),
            "top5_contains_acceptable": float(mask[rows[:, None], order[:, :5]].any(axis=1).mean()),
            "top10_contains_acceptable": float(mask[rows[:, None], order[:, :10]].any(axis=1).mean()),
            "top5_acceptable_precision": float(mask[rows[:, None], order[:, :5]].sum(axis=1).mean() / 5.0),
            "top10_acceptable_precision": float(mask[rows[:, None], order[:, :10]].sum(axis=1).mean() / 10.0),
        }
        records.append(base)

    return pd.DataFrame(records)


def baseline_scores(choice: ChoiceData, train_counts: pd.Series) -> dict[str, np.ndarray]:
    hp = choice.hospitals
    popularity = np.log1p(np.array([train_counts.get(int(hid), 0) for hid in choice.hospital_ids], dtype=float))
    popularity = np.repeat(popularity[None, :], len(choice.patients), axis=0)
    profile = choice.pair_features[:, :, 0] + 0.65 * choice.pair_features[:, :, 1]
    nearest = -choice.travel_minutes
    quality_capacity = (
        0.8 * choice.pair_features[:, :, 7]
        + 0.35 * choice.pair_features[:, :, 4]
        - 0.25 * choice.pair_features[:, :, 6]
    )
    simple_rule = (
        1.15 * choice.pair_features[:, :, 0]
        + 0.70 * choice.pair_features[:, :, 1]
        + 0.90 * choice.pair_features[:, :, 2]
        + 0.25 * choice.pair_features[:, :, 5]
        + 0.20 * choice.pair_features[:, :, 7]
        - 0.20 * choice.pair_features[:, :, 6]
    )
    return {
        "baseline_popularity": popularity,
        "baseline_nearest_synthetic": nearest,
        "baseline_profile": profile,
        "baseline_quality_capacity": quality_capacity,
        "baseline_simple_rule": simple_rule,
    }


def save_svg_bar(metrics: pd.DataFrame, path: Path) -> None:
    selected = metrics[["model", "top1_accuracy", "top5_accuracy", "mean_reciprocal_rank"]].copy()
    width = 920
    height = 380
    left = 190
    top = 45
    row_h = 44
    bar_w = 520
    colors = {"top1_accuracy": "#2f6db3", "top5_accuracy": "#3d9b6d", "mean_reciprocal_rank": "#c76b3a"}
    labels = {"top1_accuracy": "Top-1", "top5_accuracy": "Top-5", "mean_reciprocal_rank": "MRR"}
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="28" font-family="Arial" font-size="20" font-weight="700">eICU: quality of probabilistic hospital choice</text>',
    ]
    for x_tick in np.linspace(0, 1, 6):
        x = left + x_tick * bar_w
        lines.append(f'<line x1="{x:.1f}" y1="{top-8}" x2="{x:.1f}" y2="{height-45}" stroke="#dddddd" stroke-width="1"/>')
        lines.append(f'<text x="{x-10:.1f}" y="{height-20}" font-family="Arial" font-size="12" fill="#555">{x_tick:.1f}</text>')
    y = top
    for _, row in selected.iterrows():
        lines.append(f'<text x="24" y="{y+19}" font-family="Arial" font-size="13" fill="#222">{row["model"]}</text>')
        offset = 0
        for metric in ["top1_accuracy", "top5_accuracy", "mean_reciprocal_rank"]:
            value = float(row[metric])
            bw = value * bar_w
            lines.append(
                f'<rect x="{left}" y="{y+offset}" width="{bw:.1f}" height="10" fill="{colors[metric]}" opacity="0.88"/>'
            )
            lines.append(
                f'<text x="{left+bw+6:.1f}" y="{y+offset+9}" font-family="Arial" font-size="11" fill="#333">{value:.3f}</text>'
            )
            offset += 13
        y += row_h
    legend_x = width - 235
    legend_y = 58
    for idx, metric in enumerate(["top1_accuracy", "top5_accuracy", "mean_reciprocal_rank"]):
        yy = legend_y + idx * 22
        lines.append(f'<rect x="{legend_x}" y="{yy-10}" width="14" height="10" fill="{colors[metric]}"/>')
        lines.append(f'<text x="{legend_x+22}" y="{yy}" font-family="Arial" font-size="12" fill="#333">{labels[metric]}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def save_probability_examples(
    choice: ChoiceData,
    scores: np.ndarray,
    out_path: Path,
    max_patients: int,
) -> pd.DataFrame:
    probs = softmax(scores)
    rows: list[dict[str, float | int | str]] = []
    interesting = np.argsort(np.max(probs, axis=1))[::-1][:max_patients]
    for local_i in interesting:
        p = probs[local_i]
        order = np.argsort(-p)[:8]
        actual = int(choice.hospital_ids[choice.actual_index[local_i]])
        patient = choice.patients.iloc[local_i]
        for rank, alt in enumerate(order, start=1):
            hospital = choice.hospitals.iloc[alt]
            rows.append(
                {
                    "patient_stay_id": int(patient["stay_id"]),
                    "actual_hospital_id": actual,
                    "rank": rank,
                    "candidate_hospital_id": int(choice.hospital_ids[alt]),
                    "probability": float(p[alt]),
                    "is_actual": int(alt == choice.actual_index[local_i]),
                    "travel_minutes": float(choice.travel_minutes[local_i, alt]),
                    "profile_match": float(choice.pair_features[local_i, alt, 0]),
                    "unit_profile_match": float(choice.pair_features[local_i, alt, 1]),
                    "capacity_norm": float(choice.pair_features[local_i, alt, 4]),
                    "free_capacity_proxy": float(choice.pair_features[local_i, alt, 5]),
                    "load_proxy": float(choice.pair_features[local_i, alt, 6]),
                    "quality_proxy": float(choice.pair_features[local_i, alt, 7]),
                    "patient_unittype": str(patient["unittype"]),
                    "patient_dx": str(patient["apacheadmissiondx"]),
                    "patient_iss_proxy": float(patient["iss_proxy"]),
                    "patient_survival_t0": float(patient["survival_t0"]),
                }
            )
    examples = pd.DataFrame(rows)
    examples.to_csv(out_path, index=False)
    return examples


def run(args: argparse.Namespace) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(args.features).reset_index(drop=True)
    hospitals = pd.read_csv(args.hospitals)
    features = features.dropna(subset=["subject_id"]).copy()
    features["subject_id"] = features["subject_id"].astype(int)
    features["_source_index"] = np.arange(len(features))

    counts = features["subject_id"].value_counts()
    eligible_ids = counts[counts >= args.min_cases_per_hospital].index.astype(int).to_numpy()
    if args.max_hospitals and args.max_hospitals > 0:
        eligible_ids = counts.loc[eligible_ids].sort_values(ascending=False).head(args.max_hospitals).index.astype(int).to_numpy()
    eligible_ids = np.array(sorted(eligible_ids), dtype=int)
    data = features[features["subject_id"].isin(eligible_ids)].reset_index(drop=True).copy()
    data["_source_index"] = data["_source_index"].astype(int)
    train_idx, test_idx = stratified_split(data, "subject_id", args.test_fraction, args.seed)
    train = data.iloc[train_idx].reset_index(drop=True).copy()
    test = data.iloc[test_idx].reset_index(drop=True).copy()

    full_need = infer_need_matrix(features)
    full_unit = infer_unit_matrix(features)
    train_need = full_need[train["_source_index"].to_numpy()]
    train_unit = full_unit[train["_source_index"].to_numpy()]

    hospital_ids = np.array(sorted(data["subject_id"].unique()), dtype=int)
    profile_train = build_hospital_profiles(train, hospitals, hospital_ids, train_need, train_unit, alpha=args.profile_smoothing)
    rng = np.random.default_rng(args.seed)
    coords = hospital_coordinates(hospitals[hospitals["hospitalid"].isin(hospital_ids)], rng)

    train_choice = build_choice_data(
        train,
        hospitals,
        profile_train,
        full_need,
        full_unit,
        hospital_ids,
        coords,
        args.seed + 11,
        args.catchment_sigma,
    )
    test_choice = build_choice_data(
        test,
        hospitals,
        profile_train,
        full_need,
        full_unit,
        hospital_ids,
        coords,
        args.seed + 29,
        args.catchment_sigma,
    )

    x_train, other, mean, std = standardize(train_choice.pair_features, test_choice.pair_features)
    x_test = other[0]
    theta, history = fit_conditional_logit(
        x_train,
        train_choice.actual_index,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        seed=args.seed + 101,
    )

    train_scores = np.einsum("ijf,f->ij", x_train, theta)
    test_scores = np.einsum("ijf,f->ij", x_test, theta)
    test_score_outputs: dict[str, np.ndarray] = {"conditional_logit_test": test_scores}
    metrics = [
        metrics_from_scores(train_scores, train_choice.actual_index, "conditional_logit_train"),
        metrics_from_scores(test_scores, test_choice.actual_index, "conditional_logit_test"),
    ]

    if not args.skip_ablations:
        feature_index = {name: idx for idx, name in enumerate(FEATURE_NAMES)}
        transport_features = {
            "transport_score",
            "travel_minutes_scaled",
            "severe_transport_score",
            "red_transport_score",
        }
        profile_features = {
            "profile_match",
            "unit_profile_match",
            "severe_profile_match",
            "burn_profile_match",
            "trauma_profile_match",
            "neuro_profile_match",
            "cardiac_profile_match",
        }
        all_names = set(FEATURE_NAMES)
        ablations = {
            "conditional_logit_no_transport": sorted(feature_index[name] for name in all_names - transport_features),
            "conditional_logit_transport_only": sorted(feature_index[name] for name in transport_features),
            "conditional_logit_profile_only": sorted(feature_index[name] for name in profile_features),
            "conditional_logit_no_profile": sorted(feature_index[name] for name in all_names - profile_features),
        }
        for model_name, cols in ablations.items():
            sub_theta, _ = fit_conditional_logit(
                x_train[:, :, cols],
                train_choice.actual_index,
                epochs=max(180, args.epochs // 2),
                learning_rate=args.learning_rate,
                l2=args.l2,
                seed=args.seed + 200 + len(metrics),
            )
            sub_scores = np.einsum("ijf,f->ij", x_test[:, :, cols], sub_theta)
            metrics.append(metrics_from_scores(sub_scores, test_choice.actual_index, model_name))
            test_score_outputs[model_name] = sub_scores

    train_counts = train["subject_id"].value_counts()
    for name, scores in baseline_scores(test_choice, train_counts).items():
        metrics.append(metrics_from_scores(scores, test_choice.actual_index, name))
        test_score_outputs[name] = scores
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(OUT / "choice_model_metrics.csv", index=False)

    list_metrics = pd.concat(
        [probability_list_metrics(test_choice, scores, name) for name, scores in test_score_outputs.items()],
        ignore_index=True,
    )
    list_metrics.to_csv(OUT / "choice_probability_list_metrics.csv", index=False)

    coef = pd.DataFrame(
        {
            "feature": FEATURE_NAMES,
            "standardized_coefficient": theta,
            "abs_coefficient": np.abs(theta),
            "raw_feature_mean": mean,
            "raw_feature_std": std,
        }
    ).sort_values("abs_coefficient", ascending=False)
    coef.to_csv(OUT / "choice_model_coefficients.csv", index=False)
    pd.DataFrame(history).to_csv(OUT / "choice_model_training_history.csv", index=False)
    profile_train.to_csv(OUT / "choice_hospital_profiles.csv", index=False)
    save_probability_examples(test_choice, test_scores, OUT / "choice_probability_examples.csv", args.example_patients)
    save_svg_bar(metrics_df, OUT / "choice_topk_comparison.svg")

    protocol = {
        "features": str(Path(args.features).resolve()),
        "hospitals": str(Path(args.hospitals).resolve()),
        "seed": args.seed,
        "rows_total_after_filter": int(len(data)),
        "train_patients": int(len(train)),
        "test_patients": int(len(test)),
        "candidate_hospitals": int(len(hospital_ids)),
        "min_cases_per_hospital": int(args.min_cases_per_hospital),
        "max_hospitals": int(args.max_hospitals),
        "test_fraction": float(args.test_fraction),
        "catchment_sigma_km": float(args.catchment_sigma),
        "epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "l2": float(args.l2),
        "skip_ablations": bool(args.skip_ablations),
        "death_rows_total": int(data["hospital_expire_flag"].sum()),
        "death_rate_total": float(data["hospital_expire_flag"].mean()),
        "need_categories": NEED_CATEGORIES,
        "unit_types": UNIT_TYPES,
        "transport_note": "Synthetic coordinates are reproducibly generated around observed admitting hospitals because eICU demo has no patient geocoordinates.",
    }
    (OUT / "choice_model_protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")

    print(metrics_df.to_string(index=False))
    print()
    print("Probability-list metrics:")
    print(list_metrics.to_string(index=False))
    print()
    print("Top coefficients:")
    print(coef.head(10).to_string(index=False))
    print()
    print(f"Saved outputs to {OUT}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--hospitals", type=Path, default=DEFAULT_HOSPITALS)
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--min-cases-per-hospital", type=int, default=10)
    parser.add_argument("--max-hospitals", type=int, default=0)
    parser.add_argument("--test-fraction", type=float, default=0.30)
    parser.add_argument("--profile-smoothing", type=float, default=8.0)
    parser.add_argument("--catchment-sigma", type=float, default=95.0)
    parser.add_argument("--epochs", type=int, default=550)
    parser.add_argument("--learning-rate", type=float, default=0.045)
    parser.add_argument("--l2", type=float, default=0.002)
    parser.add_argument("--example-patients", type=int, default=12)
    parser.add_argument("--skip-ablations", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
