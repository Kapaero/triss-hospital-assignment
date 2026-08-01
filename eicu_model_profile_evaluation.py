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
    ROOT,
    baseline_scores,
    build_choice_data,
    build_hospital_profiles,
    fit_conditional_logit,
    hospital_coordinates,
    infer_need_matrix,
    infer_unit_matrix,
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


OUT = ROOT / "results_eicu_model_profile_eval"

MODEL_RU = {
    "conditional_logit": "Conditional logit model",
    "pair_mlp": "MLP for patient-hospital pair",
    "pair_mlp_hospital_bias": "MLP + hospital bias",
    "temporal_lstm_choice": "LSTM + pair model",
    "baseline_nearest_synthetic": "Nearest hospital",
    "baseline_simple_rule": "Suitability rule",
    "baseline_profile": "Profile only",
    "baseline_popularity": "Hospital popularity",
    "baseline_quality_capacity": "Readiness and capacity",
}

ORDER = [
    "baseline_profile",
    "baseline_simple_rule",
    "pair_mlp",
    "pair_mlp_hospital_bias",
    "temporal_lstm_choice",
    "conditional_logit",
    "baseline_nearest_synthetic",
    "baseline_popularity",
    "baseline_quality_capacity",
]

COLOR = {
    "conditional_logit": "#2d6cdf",
    "pair_mlp": "#c74b50",
    "pair_mlp_hospital_bias": "#7b59c0",
    "temporal_lstm_choice": "#208b68",
    "baseline_nearest_synthetic": "#d28a1e",
    "baseline_simple_rule": "#777777",
    "baseline_profile": "#a06b35",
    "baseline_popularity": "#8d8d8d",
    "baseline_quality_capacity": "#b0b0b0",
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


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: object, size: int, fill: str = "#222", bold: bool = False) -> None:
    draw.text(xy, str(value), font=font(size, bold), fill=fill)


def profile_metrics(scores: np.ndarray, q_profile: np.ndarray, model: str) -> dict[str, float | str]:
    probs = softmax(scores)
    order = np.argsort(-scores, axis=1)
    rows = np.arange(scores.shape[0])
    top1 = order[:, 0]
    top3 = order[:, :3]
    top5 = order[:, :5]
    top10 = order[:, :10]
    best_profile = np.argmax(q_profile, axis=1)
    best_profile_rank = np.array([np.where(order[i] == best_profile[i])[0][0] + 1 for i in range(len(order))])
    top1_q = q_profile[rows, top1]
    best_q = q_profile[rows, best_profile]
    high70 = q_profile >= 0.70
    high90 = q_profile >= 0.90
    return {
        "model": model,
        "top1_profile_probability": float(top1_q.mean()),
        "top3_profile_probability_mean": float(q_profile[rows[:, None], top3].mean()),
        "top5_profile_probability_mean": float(q_profile[rows[:, None], top5].mean()),
        "top10_profile_probability_mean": float(q_profile[rows[:, None], top10].mean()),
        "expected_profile_probability": float((probs * q_profile).sum(axis=1).mean()),
        "profile_mass_ge_0_70": float((probs * high70).sum(axis=1).mean()),
        "profile_mass_ge_0_90": float((probs * high90).sum(axis=1).mean()),
        "top1_share_ge_0_70": float((top1_q >= 0.70).mean()),
        "top1_share_ge_0_90": float((top1_q >= 0.90).mean()),
        "top5_contains_ge_0_90": float(high90[rows[:, None], top5].any(axis=1).mean()),
        "top10_contains_ge_0_90": float(high90[rows[:, None], top10].any(axis=1).mean()),
        "mean_best_profile_probability": float(best_q.mean()),
        "top1_profile_regret": float((best_q - top1_q).mean()),
        "best_profile_mean_rank": float(best_profile_rank.mean()),
        "best_profile_median_rank": float(np.median(best_profile_rank)),
    }


def run_one(
    features: pd.DataFrame,
    hospitals: pd.DataFrame,
    full_need: np.ndarray,
    full_unit: np.ndarray,
    seq_z: np.ndarray,
    args: argparse.Namespace,
    seed: int,
    repeat: int,
) -> pd.DataFrame:
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

    rows = []
    for model, scores in score_outputs.items():
        row = profile_metrics(scores, q_profile, model)
        row["repeat"] = repeat
        row["seed"] = seed
        rows.append(row)
    return pd.DataFrame(rows)


def save_ci_bar(summary: pd.DataFrame, value_col: str, ci_col: str, title: str, subtitle: str, x_label: str, path: Path, xmax: float, fmt: str) -> None:
    plot = summary[summary["model"].isin(ORDER)].copy()
    plot["order"] = plot["model"].map({m: i for i, m in enumerate(ORDER)})
    plot = plot.sort_values("order")
    scale = 2
    width = 1550
    height = 190 + 70 * len(plot)
    img = Image.new("RGB", (width * scale, height * scale), "white")
    draw = ImageDraw.Draw(img)
    left, top, bar_w, row_h = 520 * scale, 122 * scale, 790 * scale, 70 * scale
    draw_text(draw, (34 * scale, 28 * scale), title, 28 * scale, bold=True)
    draw_text(draw, (34 * scale, 68 * scale), subtitle, 17 * scale, fill="#555")
    for tick in np.linspace(0, xmax, 6):
        x = left + tick / xmax * bar_w
        draw.line([(x, top - 28 * scale), (x, height * scale - 78 * scale)], fill="#e2e2e2", width=1 * scale)
        label = f"{tick:.2f}" if xmax <= 1 else f"{tick:.0f}"
        bbox = draw.textbbox((0, 0), label, font=font(16 * scale))
        draw.text((x - (bbox[2] - bbox[0]) // 2, height * scale - 55 * scale), label, font=font(16 * scale), fill="#555")
    for idx, row in enumerate(plot.itertuples(index=False)):
        y = top + idx * row_h
        model = str(row.model)
        draw_text(draw, (34 * scale, int(y - 2 * scale)), MODEL_RU.get(model, model), 17 * scale)
        mean = float(getattr(row, value_col))
        ci = float(getattr(row, ci_col))
        bw = min(mean / xmax, 1.0) * bar_w
        color = COLOR.get(model, "#4477aa")
        draw.rounded_rectangle([left, y, left + bw, y + 25 * scale], radius=5 * scale, fill=color)
        x1 = left + max(mean - ci, 0) / xmax * bar_w
        x2 = left + min(mean + ci, xmax) / xmax * bar_w
        xm = left + mean / xmax * bar_w
        draw.line([(x1, y + 13 * scale), (x2, y + 13 * scale)], fill="#111", width=3 * scale)
        draw.line([(xm, y - 7 * scale), (xm, y + 33 * scale)], fill="#111", width=2 * scale)
        draw_text(draw, (int(left + bw + 12 * scale), int(y - 2 * scale)), fmt.format(mean=mean, ci=ci), 16 * scale, fill="#333")
    bbox = draw.textbbox((0, 0), x_label, font=font(18 * scale))
    draw.text((left + bar_w // 2 - (bbox[2] - bbox[0]) // 2, height * scale - 20 * scale), x_label, fill="#222", font=font(18 * scale))
    img = img.resize((width, height), Image.Resampling.LANCZOS)
    img.save(path)


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

    outputs = []
    seeds = [args.seed + i * 101 for i in range(args.repeats)]
    for repeat, seed in enumerate(seeds, start=1):
        print(f"repeat {repeat}/{args.repeats}, seed={seed}", flush=True)
        outputs.append(run_one(features, hospitals, full_need, full_unit, seq_z, args, seed, repeat))
    raw = pd.concat(outputs, ignore_index=True)
    summary = aggregate_metrics(raw, ["model"])
    raw.to_csv(OUT / "model_profile_metrics_raw.csv", index=False)
    summary.to_csv(OUT / "model_profile_metrics_summary.csv", index=False)

    save_ci_bar(
        summary,
        "top1_profile_probability_mean",
        "top1_profile_probability_ci95",
        "Model top-1 candidate profile suitability",
        "Evaluated via the independent probability q_ij; higher means a better profile match",
        "Mean q_ij of the top-1 candidate",
        OUT / "figure_ru_model_top1_profile_probability.png",
        xmax=1.0,
        fmt="{mean:.3f} ± {ci:.3f}",
    )
    save_ci_bar(
        summary,
        "expected_profile_probability_mean",
        "expected_profile_probability_ci95",
        "Expected profile suitability under model probabilities",
        "The metric accounts for all candidate hospitals and the probability p_ij assigned by the model",
        "E_p[q_ij]",
        OUT / "figure_ru_model_expected_profile_probability.png",
        xmax=1.0,
        fmt="{mean:.3f} ± {ci:.3f}",
    )
    save_ci_bar(
        summary,
        "profile_mass_ge_0_70_mean",
        "profile_mass_ge_0_70_ci95",
        "Probability mass on profile-suitable hospitals",
        "Computed as the sum of p_ij over hospitals where q_ij ≥ 0.70",
        "Probability mass",
        OUT / "figure_ru_model_profile_mass_ge_070.png",
        xmax=1.0,
        fmt="{mean:.3f} ± {ci:.3f}",
    )
    save_ci_bar(
        summary,
        "top1_profile_regret_mean",
        "top1_profile_regret_ci95",
        "Profile suitability loss relative to the best-matching hospital",
        "Lower is better; 0 means the model chose the most profile-suitable hospital",
        "Mean q_ij loss",
        OUT / "figure_ru_model_profile_regret.png",
        xmax=0.75,
        fmt="{mean:.3f} ± {ci:.3f}",
    )
    table = summary[summary["model"].isin(ORDER)].copy()
    table["Model"] = table["model"].map(MODEL_RU)
    table = table[
        [
            "Model",
            "top1_profile_probability_mean",
            "top1_profile_probability_ci95",
            "top5_profile_probability_mean_mean",
            "top5_profile_probability_mean_ci95",
            "expected_profile_probability_mean",
            "expected_profile_probability_ci95",
            "profile_mass_ge_0_70_mean",
            "profile_mass_ge_0_70_ci95",
            "top1_profile_regret_mean",
            "top1_profile_regret_ci95",
            "best_profile_mean_rank_mean",
            "best_profile_mean_rank_ci95",
        ]
    ]
    table.to_csv(OUT / "article_model_profile_metrics_ru.csv", index=False, encoding="utf-8-sig")
    protocol = {
        "repeats": args.repeats,
        "seeds": seeds,
        "profile_probability": "Independent q_ij profile suitability from eICU hospital capabilities; not softmax-normalized over hospitals.",
    }
    (OUT / "model_profile_eval_protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary.to_string(index=False))
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
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
