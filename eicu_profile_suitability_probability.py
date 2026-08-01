from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from eicu_discrete_choice_experiment import (
    BASE_SEED,
    DEFAULT_FEATURES,
    NEED_CATEGORIES,
    UNIT_TYPES,
    infer_need_matrix,
    infer_unit_matrix,
    stratified_split,
)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_eicu_profile_suitability"
THRESHOLDS = (0.50, 0.70, 0.90)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    paths = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
    ]
    for path in paths:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: object, size: int, fill: str = "#222", bold: bool = False) -> None:
    draw.text(xy, str(value), font=font(size, bold), fill=fill)


def geometric_mean(values: np.ndarray, axis: int) -> np.ndarray:
    clipped = np.clip(values, 1e-6, 1.0)
    return np.exp(np.mean(np.log(clipped), axis=axis))


def capability_probability(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    kappa = np.zeros(counts.shape[1], dtype=float)
    for c in range(counts.shape[1]):
        positive = counts[:, c][counts[:, c] > 0]
        kappa[c] = float(np.median(positive)) if len(positive) else 1.0
    kappa = np.maximum(kappa, 1.0)
    probability = 1.0 - np.exp(-counts / kappa[None, :])
    return probability.clip(0, 1), kappa


def build_capabilities(
    train: pd.DataFrame,
    hospital_ids: np.ndarray,
    full_need: np.ndarray,
    full_unit: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    id_to_col = {int(hid): idx for idx, hid in enumerate(hospital_ids)}
    need_counts = np.zeros((len(hospital_ids), len(NEED_CATEGORIES)), dtype=float)
    unit_counts = np.zeros((len(hospital_ids), len(UNIT_TYPES)), dtype=float)
    source = train["_source_index"].to_numpy()
    hids = train["subject_id"].astype(int).to_numpy()
    for row, hid in enumerate(hids):
        j = id_to_col[int(hid)]
        need_counts[j] += full_need[source[row]]
        unit_counts[j] += full_unit[source[row]]
    need_capability, need_kappa = capability_probability(need_counts)
    unit_capability, unit_kappa = capability_probability(unit_counts)
    return need_capability, unit_capability, need_kappa, unit_kappa


def profile_probability_matrix(
    patients: pd.DataFrame,
    hospital_ids: np.ndarray,
    full_need: np.ndarray,
    full_unit: np.ndarray,
    need_capability: np.ndarray,
    unit_capability: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = patients["_source_index"].to_numpy()
    patient_need = full_need[source]
    patient_unit = full_unit[source]
    clinical = np.zeros((len(patients), len(hospital_ids)), dtype=float)
    unit = np.zeros_like(clinical)
    for i in range(len(patients)):
        need_idx = np.flatnonzero(patient_need[i] > 0)
        unit_idx = np.flatnonzero(patient_unit[i] > 0)
        if len(need_idx) == 0:
            need_idx = np.array([len(NEED_CATEGORIES) - 1])
        if len(unit_idx) == 0:
            unit_idx = np.array([0])
        clinical[i] = geometric_mean(need_capability[:, need_idx], axis=1)
        unit[i] = geometric_mean(unit_capability[:, unit_idx], axis=1)
    profile = np.sqrt(np.clip(clinical, 1e-6, 1.0) * np.clip(unit, 1e-6, 1.0))
    return profile, clinical, unit


def profile_labels(need_row: np.ndarray) -> str:
    labels = [name for value, name in zip(need_row, NEED_CATEGORIES) if value > 0]
    return ", ".join(labels) if labels else "general"


def summarize_repeat(
    repeat: int,
    seed: int,
    test: pd.DataFrame,
    hospital_ids: np.ndarray,
    q: np.ndarray,
    q_clinical: np.ndarray,
    q_unit: np.ndarray,
    full_need: np.ndarray,
) -> tuple[dict[str, float | int], pd.DataFrame, pd.DataFrame]:
    id_to_col = {int(hid): idx for idx, hid in enumerate(hospital_ids)}
    actual_cols = np.array([id_to_col[int(hid)] for hid in test["subject_id"].astype(int)], dtype=int)
    rows = np.arange(len(test))
    order = np.argsort(-q, axis=1)
    actual_rank = np.array([np.where(order[i] == actual_cols[i])[0][0] + 1 for i in range(len(test))])
    summary: dict[str, float | int] = {
        "repeat": repeat,
        "seed": seed,
        "test_patients": int(len(test)),
        "candidate_hospitals": int(len(hospital_ids)),
        "mean_pair_profile_probability": float(q.mean()),
        "mean_max_profile_probability": float(q.max(axis=1).mean()),
        "mean_top5_profile_probability": float(np.take_along_axis(q, order[:, :5], axis=1).mean()),
        "mean_actual_hospital_probability": float(q[rows, actual_cols].mean()),
        "median_actual_hospital_rank": float(np.median(actual_rank)),
        "top5_contains_actual_by_profile_probability": float((actual_rank <= 5).mean()),
    }
    for threshold in THRESHOLDS:
        summary[f"mean_hospitals_ge_{str(threshold).replace('.', '_')}"] = float((q >= threshold).sum(axis=1).mean())
        summary[f"share_patients_any_ge_{str(threshold).replace('.', '_')}"] = float(((q >= threshold).sum(axis=1) > 0).mean())

    examples = []
    chosen = np.unique(
        np.concatenate(
            [
                np.argsort(q.max(axis=1))[:4],
                np.argsort(q.max(axis=1))[-4:],
                np.argsort(np.abs((q >= 0.70).sum(axis=1) - np.median((q >= 0.70).sum(axis=1))))[:4],
            ]
        )
    )[:14]
    for i in chosen:
        top = order[i, :8]
        patient = test.iloc[i]
        for rank, alt in enumerate(top, start=1):
            examples.append(
                {
                    "repeat": repeat,
                    "seed": seed,
                    "patient_stay_id": int(patient["stay_id"]),
                    "patient_profile": profile_labels(full_need[int(patient["_source_index"])]),
                    "patient_unit": str(patient["unittype"]),
                    "actual_hospital_id": int(patient["subject_id"]),
                    "rank": rank,
                    "candidate_hospital_id": int(hospital_ids[alt]),
                    "profile_suitability_probability": float(q[i, alt]),
                    "clinical_capability_probability": float(q_clinical[i, alt]),
                    "unit_capability_probability": float(q_unit[i, alt]),
                    "is_actual_hospital": int(alt == actual_cols[i]),
                }
            )

    patient_stats = pd.DataFrame(
        {
            "repeat": repeat,
            "seed": seed,
            "patient_stay_id": test["stay_id"].astype(int).to_numpy(),
            "patient_profile": [profile_labels(full_need[idx]) for idx in test["_source_index"].to_numpy()],
            "patient_unit": test["unittype"].astype(str).to_numpy(),
            "max_profile_probability": q.max(axis=1),
            "hospitals_ge_0_50": (q >= 0.50).sum(axis=1),
            "hospitals_ge_0_70": (q >= 0.70).sum(axis=1),
            "hospitals_ge_0_90": (q >= 0.90).sum(axis=1),
            "actual_hospital_probability": q[rows, actual_cols],
            "actual_hospital_rank": actual_rank,
        }
    )
    return summary, pd.DataFrame(examples), patient_stats


def aggregate_summary(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in raw.columns:
        if col in {"repeat", "seed"}:
            continue
        if pd.api.types.is_numeric_dtype(raw[col]):
            values = raw[col].astype(float)
            rows.append(
                {
                    "metric": col,
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "ci95": float(1.96 * values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def save_distribution(patient_stats: pd.DataFrame, path: Path) -> None:
    data = patient_stats["max_profile_probability"].to_numpy(float)
    bins = np.linspace(0, 1, 21)
    counts, edges = np.histogram(data, bins=bins)
    scale = 2
    width, height = 1200, 720
    img = Image.new("RGB", (width * scale, height * scale), "white")
    draw = ImageDraw.Draw(img)
    left, top, plot_w, plot_h = 110 * scale, 110 * scale, 900 * scale, 430 * scale
    draw_text(draw, (34 * scale, 28 * scale), "Distribution of maximum profile-match probability", 28 * scale, bold=True)
    draw_text(draw, (34 * scale, 66 * scale), "For each patient, the best hospital by independent probability q_ij is used; q_ij is not normalized across hospitals", 17 * scale, fill="#555")
    draw.rectangle([left, top, left + plot_w, top + plot_h], outline="#cccccc", fill="#fbfbfb", width=2 * scale)
    max_count = max(counts.max(), 1)
    bar_w = plot_w / len(counts)
    for i, count in enumerate(counts):
        h = count / max_count * plot_h
        x = left + i * bar_w
        y = top + plot_h - h
        draw.rectangle([x + 2 * scale, y, x + bar_w - 2 * scale, top + plot_h], fill="#2d6cdf")
    for tick in np.linspace(0, 1, 6):
        x = left + tick * plot_w
        draw.line([(x, top), (x, top + plot_h)], fill="#e5e5e5", width=1 * scale)
        draw_text(draw, (int(x - 12 * scale), int(top + plot_h + 20 * scale)), f"{tick:.1f}", 15 * scale, fill="#555")
    draw_text(draw, (left + 235 * scale, top + plot_h + 58 * scale), "Maximum profile-match probability", 18 * scale)
    draw_text(draw, (1030 * scale, 160 * scale), f"Mean: {data.mean():.3f}", 18 * scale, bold=True)
    draw_text(draw, (1030 * scale, 194 * scale), f"Median: {np.median(data):.3f}", 18 * scale)
    draw_text(draw, (1030 * scale, 228 * scale), f"n = {len(data)}", 18 * scale)
    img = img.resize((width, height), Image.Resampling.LANCZOS)
    img.save(path)


def save_thresholds(patient_stats: pd.DataFrame, path: Path) -> None:
    labels = ["q ≥ 0.50", "q ≥ 0.70", "q ≥ 0.90"]
    values = [
        patient_stats["hospitals_ge_0_50"].mean(),
        patient_stats["hospitals_ge_0_70"].mean(),
        patient_stats["hospitals_ge_0_90"].mean(),
    ]
    scale = 2
    width, height = 1050, 560
    img = Image.new("RGB", (width * scale, height * scale), "white")
    draw = ImageDraw.Draw(img)
    left, top, plot_w, row_h = 330 * scale, 130 * scale, 540 * scale, 90 * scale
    draw_text(draw, (34 * scale, 30 * scale), "How many hospitals are a profile match for a single patient", 28 * scale, bold=True)
    draw_text(draw, (34 * scale, 68 * scale), "Mean number of hospitals above a given threshold of the independent probability q_ij", 17 * scale, fill="#555")
    xmax = max(max(values) * 1.15, 1.0)
    for tick in np.linspace(0, xmax, 6):
        x = left + tick / xmax * plot_w
        draw.line([(x, top - 28 * scale), (x, top + row_h * len(values))], fill="#e6e6e6", width=1 * scale)
        draw_text(draw, (int(x - 12 * scale), int(top + row_h * len(values) + 20 * scale)), f"{tick:.1f}", 15 * scale, fill="#555")
    colors = ["#2d6cdf", "#208b68", "#c74b50"]
    for i, (label, value) in enumerate(zip(labels, values)):
        y = top + i * row_h
        draw_text(draw, (34 * scale, int(y + 8 * scale)), label, 20 * scale, bold=True)
        bw = value / xmax * plot_w
        draw.rounded_rectangle([left, y, left + bw, y + 34 * scale], radius=6 * scale, fill=colors[i])
        draw_text(draw, (int(left + bw + 14 * scale), int(y + 4 * scale)), f"{value:.1f}", 19 * scale)
    draw_text(draw, (left + 125 * scale, top + row_h * len(values) + 58 * scale), "Mean number of hospitals", 18 * scale)
    img = img.resize((width, height), Image.Resampling.LANCZOS)
    img.save(path)


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(DEFAULT_FEATURES).reset_index(drop=True)
    features = features.dropna(subset=["subject_id"]).copy()
    features["subject_id"] = features["subject_id"].astype(int)
    features["_source_index"] = np.arange(len(features))
    full_need = infer_need_matrix(features)
    full_unit = infer_unit_matrix(features)
    counts = features["subject_id"].value_counts()
    eligible = np.array(sorted(counts[counts >= 10].index.astype(int)), dtype=int)
    data = features[features["subject_id"].isin(eligible)].reset_index(drop=True).copy()

    summaries = []
    examples = []
    patient_stats = []
    kappas = []
    for repeat in range(1, 6):
        seed = BASE_SEED + (repeat - 1) * 101
        train_idx, test_idx = stratified_split(data, "subject_id", 0.30, seed)
        train = data.iloc[train_idx].reset_index(drop=True).copy()
        test = data.iloc[test_idx].reset_index(drop=True).copy()
        hospital_ids = np.array(sorted(data["subject_id"].unique()), dtype=int)
        need_cap, unit_cap, need_kappa, unit_kappa = build_capabilities(train, hospital_ids, full_need, full_unit)
        q, q_clinical, q_unit = profile_probability_matrix(test, hospital_ids, full_need, full_unit, need_cap, unit_cap)
        summary, example_df, patient_df = summarize_repeat(
            repeat,
            seed,
            test,
            hospital_ids,
            q,
            q_clinical,
            q_unit,
            full_need,
        )
        summaries.append(summary)
        examples.append(example_df)
        patient_stats.append(patient_df)
        kappas.append(
            pd.DataFrame(
                {
                    "repeat": repeat,
                    "seed": seed,
                    "need_category": NEED_CATEGORIES,
                    "need_kappa": need_kappa,
                }
            )
        )
        kappas.append(
            pd.DataFrame(
                {
                    "repeat": repeat,
                    "seed": seed,
                    "unit_type": UNIT_TYPES,
                    "unit_kappa": unit_kappa,
                }
            )
        )

    raw_summary = pd.DataFrame(summaries)
    examples_df = pd.concat(examples, ignore_index=True)
    patient_df = pd.concat(patient_stats, ignore_index=True)
    raw_summary.to_csv(OUT / "profile_suitability_summary_raw.csv", index=False)
    aggregate_summary(raw_summary).to_csv(OUT / "profile_suitability_summary.csv", index=False)
    examples_df.to_csv(OUT / "profile_suitability_patient_examples.csv", index=False, encoding="utf-8-sig")
    patient_df.to_csv(OUT / "profile_suitability_patient_stats.csv", index=False, encoding="utf-8-sig")
    pd.concat(kappas, ignore_index=True).to_csv(OUT / "profile_suitability_kappa.csv", index=False, encoding="utf-8-sig")
    save_distribution(patient_df, OUT / "figure_ru_profile_probability_distribution.png")
    save_thresholds(patient_df, OUT / "figure_ru_profile_probability_thresholds.png")
    print(aggregate_summary(raw_summary).to_string(index=False))


if __name__ == "__main__":
    run()
