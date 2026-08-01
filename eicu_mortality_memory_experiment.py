from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from eicu_discrete_choice_experiment import (
    BASE_SEED,
    DEFAULT_FEATURES,
    DEFAULT_HOSPITALS,
    FEATURE_NAMES,
    ROOT,
    acceptable_sets,
    baseline_scores,
    build_choice_data,
    build_hospital_profiles,
    fit_conditional_logit,
    hospital_coordinates,
    infer_need_matrix,
    infer_unit_matrix,
    metrics_from_scores,
    softmax,
    standardize,
    stratified_split,
)
from eicu_choice_model_comparison import (
    build_vital_sequences,
    fit_pair_mlp,
    fit_temporal_lstm_choice,
    predict_pair_mlp,
    predict_temporal_lstm_choice,
)
from eicu_choice_repeated_runs import aggregate_metrics
from eicu_profile_suitability_probability import build_capabilities, profile_probability_matrix


OUT = ROOT / "results_eicu_mortality_memory"

MODEL_RU = {
    "conditional_logit": "Conditional logit model",
    "conditional_logit_memory": "Logit model + mortality memory",
    "pair_mlp": "MLP",
    "pair_mlp_memory": "MLP + soft mortality memory",
    "pair_mlp_filter": "MLP + risk cutoff",
    "pair_mlp_intersection": "MLP + model intersection",
    "pair_mlp_hospital_bias": "MLP + hospital bias",
    "pair_mlp_hospital_bias_memory": "MLP with bias + memory",
    "temporal_lstm_choice": "LSTM",
    "temporal_lstm_choice_memory": "LSTM + mortality memory",
    "baseline_nearest_synthetic": "Nearest hospital",
    "baseline_simple_rule": "Suitability rule",
    "baseline_profile": "Profile only",
    "baseline_quality_capacity": "Readiness and capacity",
    "baseline_popularity": "Hospital popularity",
}

PLOT_ORDER = [
    "baseline_nearest_synthetic",
    "baseline_simple_rule",
    "conditional_logit",
    "conditional_logit_memory",
    "pair_mlp",
    "pair_mlp_memory",
    "pair_mlp_filter",
    "pair_mlp_intersection",
    "temporal_lstm_choice",
    "temporal_lstm_choice_memory",
]

COLORS = {
    "conditional_logit": "#2d6cdf",
    "conditional_logit_memory": "#18498f",
    "pair_mlp": "#c74b50",
    "pair_mlp_memory": "#8f2730",
    "pair_mlp_filter": "#a85a1e",
    "pair_mlp_intersection": "#7b3f95",
    "pair_mlp_hospital_bias": "#7b59c0",
    "pair_mlp_hospital_bias_memory": "#563a8f",
    "temporal_lstm_choice": "#208b68",
    "temporal_lstm_choice_memory": "#126142",
    "baseline_nearest_synthetic": "#d28a1e",
    "baseline_simple_rule": "#777777",
    "baseline_profile": "#a06b35",
    "baseline_quality_capacity": "#b0b0b0",
    "baseline_popularity": "#9a9a9a",
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def draw_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: object,
    size: int,
    fill: str = "#222",
    bold: bool = False,
) -> None:
    draw.text(xy, str(text), font=font(size, bold), fill=fill)


def choose_kernel_sigma(chosen_features: np.ndarray, seed: int, sample_size: int = 512, neighbor: int = 25) -> float:
    rng = np.random.default_rng(seed)
    n = len(chosen_features)
    if n <= 2:
        return 1.0
    sample_idx = rng.choice(n, size=min(sample_size, n), replace=False)
    sample = chosen_features[sample_idx]
    a2 = np.sum(sample * sample, axis=1, keepdims=True)
    b2 = np.sum(chosen_features * chosen_features, axis=1)[None, :]
    dist2 = np.maximum(a2 + b2 - 2.0 * sample @ chosen_features.T, 0.0)
    sorted_dist = np.sort(np.sqrt(dist2), axis=1)
    k = min(neighbor, sorted_dist.shape[1] - 1)
    sigma = float(np.median(sorted_dist[:, k]))
    return max(sigma, 0.25)


def mortality_memory_risk(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_death: np.ndarray,
    test_x: np.ndarray,
    sigma: float,
    prior_strength: float,
    chunk_size: int,
) -> np.ndarray:
    rows = np.arange(len(train_y))
    chosen = train_x[rows, train_y, :]
    death = train_death.astype(float)
    base_rate = float(death.mean())
    chosen_norm = np.sum(chosen * chosen, axis=1)[None, :]
    flat = test_x.reshape(-1, test_x.shape[-1])
    out = np.empty(len(flat), dtype=float)
    denom_scale = 2.0 * sigma * sigma
    eps = 1e-12
    for start in range(0, len(flat), chunk_size):
        chunk = flat[start : start + chunk_size]
        chunk_norm = np.sum(chunk * chunk, axis=1, keepdims=True)
        dist2 = np.maximum(chunk_norm + chosen_norm - 2.0 * chunk @ chosen.T, 0.0)
        kernel = np.exp(-dist2 / denom_scale)
        total = kernel.sum(axis=1)
        bad = kernel @ death
        out[start : start + chunk_size] = (bad + prior_strength * base_rate) / (total + prior_strength + eps)
    return out.reshape(test_x.shape[:2]).clip(0.0, 1.0)


def death_excess_log_odds(risk: np.ndarray, base_rate: float) -> np.ndarray:
    risk = np.clip(risk, 1e-4, 1.0 - 1e-4)
    base = float(np.clip(base_rate, 1e-4, 1.0 - 1e-4))
    logit_risk = np.log(risk / (1.0 - risk))
    logit_base = math.log(base / (1.0 - base))
    return np.maximum(logit_risk - logit_base, 0.0)


def apply_memory_soft(scores: np.ndarray, risk: np.ndarray, base_rate: float, gamma: float) -> np.ndarray:
    return scores - gamma * death_excess_log_odds(risk, base_rate)


def apply_low_risk_filter(scores: np.ndarray, risk: np.ndarray, base_rate: float) -> np.ndarray:
    masked = scores.copy()
    low_risk = risk <= base_rate
    empty = low_risk.sum(axis=1) == 0
    masked[~low_risk] = -1e9
    masked[empty] = scores[empty]
    return masked


def apply_intersection_filter(scores: np.ndarray, risk: np.ndarray, base_rate: float) -> np.ndarray:
    probs = softmax(scores)
    threshold = 1.0 / scores.shape[1]
    mask = (probs >= threshold) & (risk <= base_rate)
    fallback = mask.sum(axis=1) == 0
    masked = scores.copy()
    masked[~mask] = -1e9
    masked[fallback] = apply_memory_soft(scores[fallback], risk[fallback], base_rate, gamma=1.0)
    return masked


def extended_metrics(
    choice,
    scores: np.ndarray,
    model: str,
    risk: np.ndarray,
    q_profile: np.ndarray,
    test_death: np.ndarray,
    base_rate: float,
) -> dict[str, float | str]:
    base = metrics_from_scores(scores, choice.actual_index, model)
    probs = softmax(scores)
    order = np.argsort(-scores, axis=1)
    rows = np.arange(len(choice.actual_index))
    top1 = order[:, 0]
    top5 = order[:, :5]
    core = acceptable_sets(choice)["clinical_transport_capacity_core"]
    profile = choice.pair_features[:, :, 0]
    unit = choice.pair_features[:, :, 1]
    travel = choice.travel_minutes
    high_risk = risk > base_rate
    death_mask = test_death.astype(bool)

    base.update(
        {
            "mean_expected_travel_minutes": float((probs * travel).sum(axis=1).mean()),
            "mean_top1_travel_minutes": float(travel[rows, top1].mean()),
            "mean_expected_profile_match": float((probs * profile).sum(axis=1).mean()),
            "mean_top1_profile_match": float(profile[rows, top1].mean()),
            "mean_top1_unit_match": float(unit[rows, top1].mean()),
            "mean_expected_profile_probability": float((probs * q_profile).sum(axis=1).mean()),
            "mean_top1_profile_probability": float(q_profile[rows, top1].mean()),
            "mean_acceptable_core_mass": float((probs * core).sum(axis=1).mean()),
            "top1_in_acceptable_core": float(core[rows, top1].mean()),
            "top5_contains_acceptable_core": float(core[rows[:, None], top5].any(axis=1).mean()),
            "mean_expected_death_memory": float((probs * risk).sum(axis=1).mean()),
            "mean_top1_death_memory": float(risk[rows, top1].mean()),
            "mean_high_risk_probability_mass": float((probs * high_risk).sum(axis=1).mean()),
            "top1_high_risk_share": float(high_risk[rows, top1].mean()),
            "mean_actual_death_memory": float(risk[rows, choice.actual_index].mean()),
            "test_death_cases": int(death_mask.sum()),
        }
    )
    if death_mask.any():
        death_rows = np.flatnonzero(death_mask)
        ranks = np.empty(len(choice.actual_index), dtype=int)
        for i in range(len(ranks)):
            ranks[i] = int(np.where(order[i] == choice.actual_index[i])[0][0]) + 1
        base.update(
            {
                "death_actual_hospital_probability": float(probs[death_rows, choice.actual_index[death_rows]].mean()),
                "death_actual_hospital_rank": float(ranks[death_rows].mean()),
                "death_actual_hospital_memory": float(risk[death_rows, choice.actual_index[death_rows]].mean()),
                "death_high_risk_probability_mass": float((probs[death_rows] * high_risk[death_rows]).sum(axis=1).mean()),
            }
        )
    else:
        base.update(
            {
                "death_actual_hospital_probability": float("nan"),
                "death_actual_hospital_rank": float("nan"),
                "death_actual_hospital_memory": float("nan"),
                "death_high_risk_probability_mass": float("nan"),
            }
        )
    return base


def run_one(
    features: pd.DataFrame,
    hospitals: pd.DataFrame,
    full_need: np.ndarray,
    full_unit: np.ndarray,
    seq_z: np.ndarray,
    args: argparse.Namespace,
    seed: int,
    repeat: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = features["subject_id"].value_counts()
    eligible = counts[counts >= args.min_cases_per_hospital].index.astype(int).to_numpy()
    if args.max_hospitals > 0:
        eligible = counts.loc[eligible].sort_values(ascending=False).head(args.max_hospitals).index.astype(int).to_numpy()
    eligible = np.array(sorted(eligible), dtype=int)
    data = features[features["subject_id"].isin(eligible)].reset_index(drop=True).copy()
    train_idx, test_idx = stratified_split(data, "subject_id", args.test_fraction, seed)
    train = data.iloc[train_idx].reset_index(drop=True).copy()
    test = data.iloc[test_idx].reset_index(drop=True).copy()
    hospital_ids = np.array(sorted(data["subject_id"].unique()), dtype=int)

    train_need = full_need[train["_source_index"].to_numpy()]
    train_unit = full_unit[train["_source_index"].to_numpy()]
    profiles = build_hospital_profiles(train, hospitals, hospital_ids, train_need, train_unit, alpha=args.profile_smoothing)
    need_cap, unit_cap, _, _ = build_capabilities(train, hospital_ids, full_need, full_unit)
    q_profile, _, _ = profile_probability_matrix(test, hospital_ids, full_need, full_unit, need_cap, unit_cap)

    rng = np.random.default_rng(seed)
    coords = hospital_coordinates(hospitals[hospitals["hospitalid"].isin(hospital_ids)], rng)
    train_choice = build_choice_data(train, hospitals, profiles, full_need, full_unit, hospital_ids, coords, seed + 11, args.catchment_sigma)
    test_choice = build_choice_data(test, hospitals, profiles, full_need, full_unit, hospital_ids, coords, seed + 29, args.catchment_sigma)
    x_train, other, _, _ = standardize(train_choice.pair_features, test_choice.pair_features)
    x_test = other[0]
    seq_train = seq_z[train["_source_index"].to_numpy()]
    seq_test = seq_z[test["_source_index"].to_numpy()]
    train_death = train["hospital_expire_flag"].astype(float).to_numpy()
    test_death = test["hospital_expire_flag"].astype(float).to_numpy()
    train_base_rate = float(train_death.mean())
    sigma = choose_kernel_sigma(x_train[np.arange(len(train_choice.actual_index)), train_choice.actual_index, :], seed + 901)
    risk = mortality_memory_risk(
        x_train,
        train_choice.actual_index,
        train_death,
        x_test,
        sigma=sigma,
        prior_strength=args.memory_prior_strength,
        chunk_size=args.memory_chunk_size,
    )

    score_outputs: dict[str, np.ndarray] = {}
    theta, _ = fit_conditional_logit(x_train, train_choice.actual_index, args.linear_epochs, args.linear_lr, args.l2, seed + 101)
    score_outputs["conditional_logit"] = np.einsum("ijf,f->ij", x_test, theta)

    mlp_params, _ = fit_pair_mlp(x_train, train_choice.actual_index, args.mlp_hidden, args.mlp_epochs, args.mlp_lr, args.l2, seed + 201, False)
    score_outputs["pair_mlp"] = predict_pair_mlp(x_test, mlp_params, False)

    bias_params, _ = fit_pair_mlp(x_train, train_choice.actual_index, args.mlp_hidden, args.mlp_epochs, args.mlp_lr, args.l2, seed + 301, True)
    score_outputs["pair_mlp_hospital_bias"] = predict_pair_mlp(x_test, bias_params, True)

    lstm_params, _ = fit_temporal_lstm_choice(
        x_train,
        seq_train,
        train_choice.actual_index,
        args.lstm_pair_hidden,
        args.lstm_hidden,
        args.lstm_epochs,
        args.lstm_lr,
        args.l2,
        seed + 501,
    )
    score_outputs["temporal_lstm_choice"] = predict_temporal_lstm_choice(x_test, seq_test, lstm_params)

    train_counts = train["subject_id"].value_counts()
    score_outputs.update(baseline_scores(test_choice, train_counts))

    variants: dict[str, np.ndarray] = {}
    keep_base = [
        "conditional_logit",
        "pair_mlp",
        "pair_mlp_hospital_bias",
        "temporal_lstm_choice",
        "baseline_nearest_synthetic",
        "baseline_simple_rule",
        "baseline_profile",
        "baseline_quality_capacity",
        "baseline_popularity",
    ]
    for name in keep_base:
        variants[name] = score_outputs[name]
    for name in ["conditional_logit", "pair_mlp", "pair_mlp_hospital_bias", "temporal_lstm_choice"]:
        variants[f"{name}_memory"] = apply_memory_soft(score_outputs[name], risk, train_base_rate, gamma=args.memory_gamma)
    variants["pair_mlp_filter"] = apply_low_risk_filter(score_outputs["pair_mlp"], risk, train_base_rate)
    variants["pair_mlp_intersection"] = apply_intersection_filter(score_outputs["pair_mlp"], risk, train_base_rate)

    metric_rows = []
    for name, scores in variants.items():
        row = extended_metrics(test_choice, scores, name, risk, q_profile, test_death, train_base_rate)
        row.update(
            {
                "repeat": repeat,
                "seed": seed,
                "train_death_rate": train_base_rate,
                "test_death_rate": float(test_death.mean()),
                "memory_sigma": sigma,
            }
        )
        metric_rows.append(row)

    examples = build_examples(test_choice, score_outputs["pair_mlp"], variants["pair_mlp_memory"], risk, q_profile, test_death, train_base_rate, repeat, seed)
    return pd.DataFrame(metric_rows), examples


def build_examples(
    choice,
    base_scores: np.ndarray,
    memory_scores: np.ndarray,
    risk: np.ndarray,
    q_profile: np.ndarray,
    test_death: np.ndarray,
    base_rate: float,
    repeat: int,
    seed: int,
    max_patients: int = 12,
) -> pd.DataFrame:
    death_idx = np.flatnonzero(test_death.astype(bool))
    if len(death_idx) == 0:
        return pd.DataFrame()
    rows = np.arange(len(choice.actual_index))
    actual_risk = risk[rows, choice.actual_index]
    selected = death_idx[np.argsort(-actual_risk[death_idx])[:max_patients]]
    base_prob = softmax(base_scores)
    mem_prob = softmax(memory_scores)
    records = []
    for i in selected:
        patient = choice.patients.iloc[i]
        actual = int(choice.hospital_ids[choice.actual_index[i]])
        base_order = np.argsort(-base_prob[i])[:5]
        mem_order = np.argsort(-mem_prob[i])[:5]
        for source, order, prob in [("base_mlp", base_order, base_prob), ("memory_mlp", mem_order, mem_prob)]:
            for rank, alt in enumerate(order, start=1):
                records.append(
                    {
                        "repeat": repeat,
                        "seed": seed,
                        "patient_stay_id": int(patient["stay_id"]),
                        "patient_dx": str(patient["apacheadmissiondx"]),
                        "actual_hospital_id": actual,
                        "list_type": source,
                        "rank": rank,
                        "candidate_hospital_id": int(choice.hospital_ids[alt]),
                        "probability": float(prob[i, alt]),
                        "death_memory_risk": float(risk[i, alt]),
                        "risk_above_train_rate": int(risk[i, alt] > base_rate),
                        "profile_probability": float(q_profile[i, alt]),
                        "travel_minutes": float(choice.travel_minutes[i, alt]),
                        "is_actual_hospital": int(alt == choice.actual_index[i]),
                    }
                )
    return pd.DataFrame(records)


def save_bar(
    summary: pd.DataFrame,
    value_col: str,
    ci_col: str,
    title: str,
    subtitle: str,
    xlabel: str,
    path: Path,
    xmax: float,
    lower_is_better: bool = False,
) -> None:
    plot = summary[summary["model"].isin(PLOT_ORDER)].copy()
    plot["order"] = plot["model"].map({name: i for i, name in enumerate(PLOT_ORDER)})
    plot = plot.sort_values("order")
    scale = 2
    width = 1620
    height = 190 + 68 * len(plot)
    img = Image.new("RGB", (width * scale, height * scale), "white")
    draw = ImageDraw.Draw(img)
    left, top, bar_w, row_h = 570 * scale, 122 * scale, 780 * scale, 68 * scale
    draw_text(draw, (34 * scale, 26 * scale), title, 28 * scale, bold=True)
    draw_text(draw, (34 * scale, 66 * scale), subtitle, 17 * scale, fill="#555")
    ticks = np.linspace(0, xmax, 6)
    for tick in ticks:
        x = left + tick / xmax * bar_w
        draw.line([(x, top - 28 * scale), (x, height * scale - 78 * scale)], fill="#e3e3e3", width=1 * scale)
        label = f"{tick:.2f}" if xmax <= 1.0 else f"{tick:.0f}"
        bbox = draw.textbbox((0, 0), label, font=font(16 * scale))
        draw.text((x - (bbox[2] - bbox[0]) // 2, height * scale - 54 * scale), label, font=font(16 * scale), fill="#555")
    for idx, row in enumerate(plot.itertuples(index=False)):
        y = top + idx * row_h
        model = str(row.model)
        mean = float(getattr(row, value_col))
        ci = float(getattr(row, ci_col))
        color = COLORS.get(model, "#4477aa")
        draw_text(draw, (34 * scale, int(y - 2 * scale)), MODEL_RU.get(model, model), 17 * scale)
        bw = min(max(mean, 0.0) / xmax, 1.0) * bar_w
        draw.rounded_rectangle([left, y, left + bw, y + 25 * scale], radius=5 * scale, fill=color)
        x1 = left + max(mean - ci, 0) / xmax * bar_w
        x2 = left + min(mean + ci, xmax) / xmax * bar_w
        xm = left + min(mean, xmax) / xmax * bar_w
        draw.line([(x1, y + 13 * scale), (x2, y + 13 * scale)], fill="#111", width=3 * scale)
        draw.line([(xm, y - 7 * scale), (xm, y + 33 * scale)], fill="#111", width=2 * scale)
        suffix = " lower is better" if lower_is_better else ""
        draw_text(draw, (int(left + bw + 12 * scale), int(y - 2 * scale)), f"{mean:.3f} ± {ci:.3f}", 16 * scale, fill="#333")
    bbox = draw.textbbox((0, 0), xlabel, font=font(18 * scale))
    draw.text((left + bar_w // 2 - (bbox[2] - bbox[0]) // 2, height * scale - 22 * scale), xlabel, fill="#222", font=font(18 * scale))
    if lower_is_better:
        draw_text(draw, (1360 * scale, 31 * scale), "lower is better", 17 * scale, fill="#666", bold=True)
    img = img.resize((width, height), Image.Resampling.LANCZOS)
    img.save(path)


def save_tradeoff(summary: pd.DataFrame, path: Path) -> None:
    plot = summary[summary["model"].isin(PLOT_ORDER)].copy()
    plot["order"] = plot["model"].map({name: i for i, name in enumerate(PLOT_ORDER)})
    plot = plot.sort_values("order").reset_index(drop=True)
    scale = 2
    width, height = 1360, 820
    img = Image.new("RGB", (width * scale, height * scale), "white")
    draw = ImageDraw.Draw(img)
    left, top, plot_w, plot_h = 110 * scale, 115 * scale, 760 * scale, 500 * scale
    draw_text(draw, (34 * scale, 28 * scale), "Trade-off: travel time vs. mortality memory", 28 * scale, bold=True)
    draw_text(draw, (34 * scale, 68 * scale), "Point number corresponds to the model in the legend; further left and lower means less transport and historical risk", 17 * scale, fill="#555")
    x = plot["mean_top1_travel_minutes_mean"].to_numpy(float)
    y = plot["mean_top1_death_memory_mean"].to_numpy(float)
    xmin, xmax = max(0.0, x.min() * 0.85), x.max() * 1.08
    ymin, ymax = 0.0, max(y.max() * 1.15, 0.10)
    draw.rectangle([left, top, left + plot_w, top + plot_h], outline="#cfcfcf", fill="#fbfbfb", width=2 * scale)
    for tick in np.linspace(xmin, xmax, 6):
        px = left + (tick - xmin) / (xmax - xmin) * plot_w
        draw.line([(px, top), (px, top + plot_h)], fill="#e4e4e4", width=1 * scale)
        draw_text(draw, (int(px - 24 * scale), int(top + plot_h + 18 * scale)), f"{tick:.0f}", 15 * scale, fill="#555")
    for tick in np.linspace(ymin, ymax, 6):
        py = top + plot_h - (tick - ymin) / (ymax - ymin) * plot_h
        draw.line([(left, py), (left + plot_w, py)], fill="#e4e4e4", width=1 * scale)
        draw_text(draw, (30 * scale, int(py - 9 * scale)), f"{tick:.2f}", 15 * scale, fill="#555")
    for idx, row in enumerate(plot.itertuples(index=False), start=1):
        model = str(row.model)
        px = left + (float(row.mean_top1_travel_minutes_mean) - xmin) / (xmax - xmin) * plot_w
        py = top + plot_h - (float(row.mean_top1_death_memory_mean) - ymin) / (ymax - ymin) * plot_h
        color = COLORS.get(model, "#4477aa")
        draw.ellipse([px - 12 * scale, py - 12 * scale, px + 12 * scale, py + 12 * scale], fill=color, outline="#222", width=1 * scale)
        label = str(idx)
        bbox = draw.textbbox((0, 0), label, font=font(14 * scale, bold=True))
        draw.text((px - (bbox[2] - bbox[0]) / 2, py - (bbox[3] - bbox[1]) / 2 - 1 * scale), label, font=font(14 * scale, bold=True), fill="white")
    draw_text(draw, (left + 210 * scale, top + plot_h + 58 * scale), "Travel time to top candidate, min", 18 * scale)
    draw_text(draw, (895 * scale, 122 * scale), "Legend", 20 * scale, bold=True)
    for idx, row in enumerate(plot.itertuples(index=False), start=1):
        y0 = 160 * scale + (idx - 1) * 48 * scale
        model = str(row.model)
        color = COLORS.get(model, "#4477aa")
        draw.ellipse([895 * scale, y0 - 13 * scale, 921 * scale, y0 + 13 * scale], fill=color, outline="#222", width=1 * scale)
        label = str(idx)
        bbox = draw.textbbox((0, 0), label, font=font(13 * scale, bold=True))
        draw.text((908 * scale - (bbox[2] - bbox[0]) / 2, y0 - (bbox[3] - bbox[1]) / 2 - 1 * scale), label, font=font(13 * scale, bold=True), fill="white")
        draw_text(draw, (934 * scale, y0 - 13 * scale), MODEL_RU.get(model, model), 15 * scale)
        draw_text(
            draw,
            (934 * scale, y0 + 8 * scale),
            f"{float(row.mean_top1_travel_minutes_mean):.1f} min; risk {float(row.mean_top1_death_memory_mean):.3f}",
            12 * scale,
            fill="#555",
        )
    draw_text(draw, (884 * scale, 690 * scale), "Y axis: mortality memory risk of the top candidate", 15 * scale, fill="#333")
    img = img.resize((width, height), Image.Resampling.LANCZOS)
    img.save(path)


def write_article_tables(summary: pd.DataFrame) -> None:
    table = summary[summary["model"].isin(PLOT_ORDER)].copy()
    table["Model"] = table["model"].map(MODEL_RU)
    cols = [
        "Model",
        "top5_accuracy_mean",
        "top5_accuracy_ci95",
        "mean_acceptable_core_mass_mean",
        "mean_acceptable_core_mass_ci95",
        "mean_top1_profile_probability_mean",
        "mean_top1_profile_probability_ci95",
        "mean_top1_travel_minutes_mean",
        "mean_top1_travel_minutes_ci95",
        "mean_top1_death_memory_mean",
        "mean_top1_death_memory_ci95",
        "mean_high_risk_probability_mass_mean",
        "mean_high_risk_probability_mass_ci95",
        "death_actual_hospital_probability_mean",
        "death_actual_hospital_probability_ci95",
    ]
    table[cols].to_csv(OUT / "article_mortality_memory_metrics_ru.csv", index=False, encoding="utf-8-sig")


def run(args: argparse.Namespace) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(args.features).reset_index(drop=True)
    hospitals = pd.read_csv(args.hospitals)
    features = features.dropna(subset=["subject_id"]).copy()
    features["subject_id"] = features["subject_id"].astype(int)
    features["_source_index"] = np.arange(len(features))
    full_need = infer_need_matrix(features)
    full_unit = infer_unit_matrix(features)
    sequence, seq_mean, seq_std = build_vital_sequences(features, args.sequence_bins, args.seed + 777)
    seq_z = (sequence - seq_mean) / seq_std

    all_metrics = []
    all_examples = []
    seeds = [args.seed + i * 101 for i in range(args.repeats)]
    for repeat, seed in enumerate(seeds, start=1):
        print(f"repeat {repeat}/{args.repeats}, seed={seed}", flush=True)
        metrics_df, examples_df = run_one(features, hospitals, full_need, full_unit, seq_z, args, seed, repeat)
        all_metrics.append(metrics_df)
        all_examples.append(examples_df)

    raw = pd.concat(all_metrics, ignore_index=True)
    examples = pd.concat(all_examples, ignore_index=True) if all_examples else pd.DataFrame()
    summary = aggregate_metrics(raw, ["model"])
    raw.to_csv(OUT / "mortality_memory_metrics_raw.csv", index=False)
    summary.to_csv(OUT / "mortality_memory_metrics_summary.csv", index=False)
    examples.to_csv(OUT / "mortality_memory_patient_examples.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"pair_feature": FEATURE_NAMES}).to_csv(OUT / "feature_names.csv", index=False)
    write_article_tables(summary)

    save_bar(
        summary,
        "mean_top1_death_memory_mean",
        "mean_top1_death_memory_ci95",
        "Mortality memory of the top candidate",
        "Risk is computed from the similarity of the patient-hospital pair to past fatal cases; lower is better",
        "Mean mortality memory risk",
        OUT / "figure_ru_top1_death_memory.png",
        xmax=max(0.12, float(summary["mean_top1_death_memory_mean"].max()) * 1.15),
        lower_is_better=True,
    )
    save_bar(
        summary,
        "mean_high_risk_probability_mass_mean",
        "mean_high_risk_probability_mass_ci95",
        "Probability mass on routes with above-average risk",
        "High risk: local mortality memory exceeds the baseline death rate in the training set",
        "Probability mass",
        OUT / "figure_ru_high_risk_probability_mass.png",
        xmax=1.0,
        lower_is_better=True,
    )
    save_bar(
        summary,
        "mean_acceptable_core_mass_mean",
        "mean_acceptable_core_mass_ci95",
        "Probability mass in the acceptable core",
        "The acceptable core accounts for profile match, transport accessibility, and free capacity",
        "Probability mass",
        OUT / "figure_ru_acceptable_core_mass_memory.png",
        xmax=0.60,
    )
    save_bar(
        summary,
        "death_actual_hospital_probability_mean",
        "death_actual_hospital_probability_ci95",
        "Probability of the actual hospital for patients with a fatal outcome",
        "After adding memory, the model should favor routes resembling past fatal cases less",
        "Mean probability",
        OUT / "figure_ru_death_actual_probability.png",
        xmax=0.25,
        lower_is_better=True,
    )
    save_tradeoff(summary, OUT / "figure_ru_memory_risk_transport_tradeoff.png")

    protocol = {
        "features": str(Path(args.features).resolve()),
        "hospitals": str(Path(args.hospitals).resolve()),
        "repeats": args.repeats,
        "seeds": seeds,
        "death_rows_total": int(features["hospital_expire_flag"].sum()),
        "death_rate_total": float(features["hospital_expire_flag"].astype(float).mean()),
        "memory_formula": "kernel-smoothed local mortality risk over observed death cases in train set",
        "memory_gamma": args.memory_gamma,
        "memory_prior_strength": args.memory_prior_strength,
        "risk_threshold": "train death rate for each split",
        "transport_note": "Synthetic coordinates are reproducibly generated because eICU Demo has hospital region but no patient geocoordinates.",
    }
    (OUT / "mortality_memory_protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")

    cols = [
        "model",
        "top5_accuracy_mean",
        "mean_acceptable_core_mass_mean",
        "mean_top1_travel_minutes_mean",
        "mean_top1_profile_probability_mean",
        "mean_top1_death_memory_mean",
        "mean_high_risk_probability_mass_mean",
        "death_actual_hospital_probability_mean",
    ]
    print(summary[cols].sort_values("mean_top1_death_memory_mean").to_string(index=False))
    print(f"Saved outputs to {OUT}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--hospitals", type=Path, default=DEFAULT_HOSPITALS)
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--min-cases-per-hospital", type=int, default=10)
    parser.add_argument("--max-hospitals", type=int, default=0)
    parser.add_argument("--test-fraction", type=float, default=0.30)
    parser.add_argument("--profile-smoothing", type=float, default=8.0)
    parser.add_argument("--catchment-sigma", type=float, default=95.0)
    parser.add_argument("--l2", type=float, default=0.002)
    parser.add_argument("--linear-epochs", type=int, default=420)
    parser.add_argument("--linear-lr", type=float, default=0.045)
    parser.add_argument("--mlp-hidden", type=int, default=28)
    parser.add_argument("--mlp-epochs", type=int, default=230)
    parser.add_argument("--mlp-lr", type=float, default=0.025)
    parser.add_argument("--sequence-bins", type=int, default=6)
    parser.add_argument("--lstm-hidden", type=int, default=10)
    parser.add_argument("--lstm-pair-hidden", type=int, default=22)
    parser.add_argument("--lstm-epochs", type=int, default=90)
    parser.add_argument("--lstm-lr", type=float, default=0.018)
    parser.add_argument("--memory-gamma", type=float, default=1.0)
    parser.add_argument("--memory-prior-strength", type=float, default=1.0)
    parser.add_argument("--memory-chunk-size", type=int, default=4096)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
