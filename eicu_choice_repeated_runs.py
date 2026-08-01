"""
Repeated eICU choice-model experiment.

Runs several train/test splits to estimate mean, standard deviation, and
approximate 95% confidence intervals for routing probability models.

Run:
  python eicu_choice_repeated_runs.py --repeats 5
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from eicu_discrete_choice_experiment import (
    BASE_SEED,
    DEFAULT_FEATURES,
    DEFAULT_HOSPITALS,
    FEATURE_NAMES,
    ROOT,
    baseline_scores,
    build_choice_data,
    build_hospital_profiles,
    fit_conditional_logit,
    hospital_coordinates,
    infer_need_matrix,
    infer_unit_matrix,
    metrics_from_scores,
    probability_list_metrics,
    standardize,
    stratified_split,
)
from eicu_choice_model_comparison import (
    ambiguity_frame,
    ambiguity_summary,
    build_vital_sequences,
    fit_pair_mlp,
    fit_temporal_lstm_choice,
    predict_pair_mlp,
    predict_temporal_lstm_choice,
)


OUT = ROOT / "results_eicu_choice_repeated"


def aggregate_metrics(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    metric_cols = [
        c
        for c in df.columns
        if c not in set(group_cols + ["seed", "repeat"])
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    rows = []
    for keys, part in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys)}
        row["repeats"] = int(part["repeat"].nunique())
        for col in metric_cols:
            values = part[col].astype(float)
            row[f"{col}_mean"] = float(values.mean())
            row[f"{col}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{col}_ci95"] = float(1.96 * values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def run_one(
    features: pd.DataFrame,
    hospitals: pd.DataFrame,
    full_need: np.ndarray,
    full_unit: np.ndarray,
    seq_z: np.ndarray,
    args: argparse.Namespace,
    seed: int,
    repeat: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    rng = np.random.default_rng(seed)
    coords = hospital_coordinates(hospitals[hospitals["hospitalid"].isin(hospital_ids)], rng)
    train_choice = build_choice_data(
        train,
        hospitals,
        profiles,
        full_need,
        full_unit,
        hospital_ids,
        coords,
        seed + 11,
        args.catchment_sigma,
    )
    test_choice = build_choice_data(
        test,
        hospitals,
        profiles,
        full_need,
        full_unit,
        hospital_ids,
        coords,
        seed + 29,
        args.catchment_sigma,
    )
    x_train, other, _, _ = standardize(train_choice.pair_features, test_choice.pair_features)
    x_test = other[0]
    seq_train = seq_z[train["_source_index"].to_numpy()]
    seq_test = seq_z[test["_source_index"].to_numpy()]

    metrics = []
    score_outputs: dict[str, np.ndarray] = {}

    theta, _ = fit_conditional_logit(
        x_train,
        train_choice.actual_index,
        args.linear_epochs,
        args.linear_lr,
        args.l2,
        seed + 101,
    )
    scores = np.einsum("ijf,f->ij", x_test, theta)
    metrics.append(metrics_from_scores(scores, test_choice.actual_index, "conditional_logit"))
    score_outputs["conditional_logit"] = scores

    mlp_params, _ = fit_pair_mlp(
        x_train,
        train_choice.actual_index,
        args.mlp_hidden,
        args.mlp_epochs,
        args.mlp_lr,
        args.l2,
        seed + 201,
        False,
    )
    scores = predict_pair_mlp(x_test, mlp_params, False)
    metrics.append(metrics_from_scores(scores, test_choice.actual_index, "pair_mlp"))
    score_outputs["pair_mlp"] = scores

    bias_params, _ = fit_pair_mlp(
        x_train,
        train_choice.actual_index,
        args.mlp_hidden,
        args.mlp_epochs,
        args.mlp_lr,
        args.l2,
        seed + 301,
        True,
    )
    scores = predict_pair_mlp(x_test, bias_params, True)
    metrics.append(metrics_from_scores(scores, test_choice.actual_index, "pair_mlp_hospital_bias"))
    score_outputs["pair_mlp_hospital_bias"] = scores

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
    scores = predict_temporal_lstm_choice(x_test, seq_test, lstm_params)
    metrics.append(metrics_from_scores(scores, test_choice.actual_index, "temporal_lstm_choice"))
    score_outputs["temporal_lstm_choice"] = scores

    train_counts = train["subject_id"].value_counts()
    for name, scores in baseline_scores(test_choice, train_counts).items():
        metrics.append(metrics_from_scores(scores, test_choice.actual_index, name))
        score_outputs[name] = scores

    metrics_df = pd.DataFrame(metrics)
    list_df = pd.concat(
        [probability_list_metrics(test_choice, scores, name) for name, scores in score_outputs.items()],
        ignore_index=True,
    )
    ambiguity = pd.concat(
        [ambiguity_frame(test_choice, scores, name) for name, scores in score_outputs.items()],
        ignore_index=True,
    )
    ambiguity_df = ambiguity_summary(ambiguity)

    for df in [metrics_df, list_df, ambiguity_df]:
        df.insert(0, "seed", seed)
        df.insert(0, "repeat", repeat)
    return metrics_df, list_df, ambiguity_df


def save_ci_svg(summary: pd.DataFrame, path: Path) -> None:
    plot = summary[~summary["model"].str.contains("train", case=False, na=False)].copy()
    if "top5_accuracy_mean" not in plot.columns:
        return
    plot = plot.sort_values("top5_accuracy_mean", ascending=True)
    width, height = 980, max(360, 48 + 34 * len(plot))
    left, top, bar_w, row_h = 260, 30, 560, 30
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="24" font-family="Arial" font-size="18" font-weight="700">Repeated splits: Top-5 accuracy with 95% CI</text>',
    ]
    for tick in np.linspace(0, 1, 6):
        x = left + tick * bar_w
        lines.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-35}" stroke="#e0e0e0"/>')
        lines.append(f'<text x="{x-10:.1f}" y="{height-12}" font-family="Arial" font-size="12">{tick:.1f}</text>')
    for idx, row in enumerate(plot.itertuples(index=False)):
        y = top + idx * row_h + 7
        mean = float(row.top5_accuracy_mean)
        ci = float(row.top5_accuracy_ci95)
        lines.append(f'<text x="24" y="{y+8}" font-family="Arial" font-size="12">{row.model}</text>')
        lines.append(f'<rect x="{left}" y="{y}" width="{mean * bar_w:.1f}" height="12" fill="#3b77bd"/>')
        x1 = left + max(mean - ci, 0) * bar_w
        x2 = left + min(mean + ci, 1) * bar_w
        xm = left + mean * bar_w
        lines.append(f'<line x1="{x1:.1f}" y1="{y+6}" x2="{x2:.1f}" y2="{y+6}" stroke="#222" stroke-width="1.5"/>')
        lines.append(f'<line x1="{xm:.1f}" y1="{y-2}" x2="{xm:.1f}" y2="{y+14}" stroke="#222" stroke-width="1"/>')
        lines.append(f'<text x="{left + mean * bar_w + 8:.1f}" y="{y+10}" font-family="Arial" font-size="11">{mean:.3f}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def save_core_mass_svg(summary: pd.DataFrame, path: Path) -> None:
    plot = summary[summary["metric_group"] == "clinical_transport_capacity_core"].copy()
    if plot.empty:
        return
    plot = plot.sort_values("mean_probability_mass_in_set_mean", ascending=True)
    width, height = 980, max(360, 48 + 34 * len(plot))
    left, top, bar_w, row_h = 280, 30, 530, 30
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="24" font-family="Arial" font-size="18" font-weight="700">Probability mass in acceptable core with 95% CI</text>',
    ]
    for tick in np.linspace(0, 0.6, 7):
        x = left + tick / 0.6 * bar_w
        lines.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-35}" stroke="#e0e0e0"/>')
        lines.append(f'<text x="{x-10:.1f}" y="{height-12}" font-family="Arial" font-size="12">{tick:.1f}</text>')
    for idx, row in enumerate(plot.itertuples(index=False)):
        y = top + idx * row_h + 7
        mean = float(row.mean_probability_mass_in_set_mean)
        ci = float(row.mean_probability_mass_in_set_ci95)
        lines.append(f'<text x="24" y="{y+8}" font-family="Arial" font-size="12">{row.model}</text>')
        lines.append(f'<rect x="{left}" y="{y}" width="{mean / 0.6 * bar_w:.1f}" height="12" fill="#3a9b63"/>')
        x1 = left + max(mean - ci, 0) / 0.6 * bar_w
        x2 = left + min(mean + ci, 0.6) / 0.6 * bar_w
        xm = left + mean / 0.6 * bar_w
        lines.append(f'<line x1="{x1:.1f}" y1="{y+6}" x2="{x2:.1f}" y2="{y+6}" stroke="#222" stroke-width="1.5"/>')
        lines.append(f'<line x1="{xm:.1f}" y1="{y-2}" x2="{xm:.1f}" y2="{y+14}" stroke="#222" stroke-width="1"/>')
        lines.append(f'<text x="{xm+8:.1f}" y="{y+10}" font-family="Arial" font-size="11">{mean:.3f}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


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
    all_list = []
    all_ambiguity = []
    seeds = [args.seed + i * 101 for i in range(args.repeats)]
    for repeat, seed in enumerate(seeds, start=1):
        print(f"repeat {repeat}/{args.repeats}, seed={seed}", flush=True)
        metrics_df, list_df, ambiguity_df = run_one(
            features,
            hospitals,
            full_need,
            full_unit,
            seq_z,
            args,
            seed,
            repeat,
        )
        all_metrics.append(metrics_df)
        all_list.append(list_df)
        all_ambiguity.append(ambiguity_df)

    raw_metrics = pd.concat(all_metrics, ignore_index=True)
    raw_list = pd.concat(all_list, ignore_index=True)
    raw_ambiguity = pd.concat(all_ambiguity, ignore_index=True)
    metrics_summary = aggregate_metrics(raw_metrics, ["model"])
    list_summary = aggregate_metrics(raw_list, ["model", "metric_group"])
    ambiguity_summary_df = aggregate_metrics(raw_ambiguity, ["model"])

    raw_metrics.to_csv(OUT / "repeated_model_metrics_raw.csv", index=False)
    raw_list.to_csv(OUT / "repeated_probability_list_metrics_raw.csv", index=False)
    raw_ambiguity.to_csv(OUT / "repeated_ambiguity_summary_raw.csv", index=False)
    metrics_summary.to_csv(OUT / "repeated_model_metrics_summary.csv", index=False)
    list_summary.to_csv(OUT / "repeated_probability_list_metrics_summary.csv", index=False)
    ambiguity_summary_df.to_csv(OUT / "repeated_ambiguity_summary.csv", index=False)
    pd.DataFrame({"pair_feature": FEATURE_NAMES}).to_csv(OUT / "feature_names.csv", index=False)
    save_ci_svg(metrics_summary, OUT / "figure_repeated_top5_ci.svg")
    save_core_mass_svg(list_summary, OUT / "figure_repeated_core_mass_ci.svg")

    protocol = {
        "features": str(Path(args.features).resolve()),
        "hospitals": str(Path(args.hospitals).resolve()),
        "repeats": args.repeats,
        "seeds": seeds,
        "sequence_bins": args.sequence_bins,
        "linear_epochs": args.linear_epochs,
        "mlp_epochs": args.mlp_epochs,
        "lstm_epochs": args.lstm_epochs,
        "note": "95% CI is computed as 1.96 * sample standard deviation / sqrt(number of repeats).",
    }
    (OUT / "repeated_protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("Rank metrics summary:")
    cols = ["model", "top1_accuracy_mean", "top1_accuracy_ci95", "top5_accuracy_mean", "top5_accuracy_ci95", "mean_reciprocal_rank_mean"]
    print(metrics_summary[cols].sort_values("top5_accuracy_mean", ascending=False).to_string(index=False))
    print()
    print("Acceptable-core summary:")
    core = list_summary[list_summary["metric_group"] == "clinical_transport_capacity_core"].copy()
    cols = ["model", "mean_probability_mass_in_set_mean", "mean_probability_mass_in_set_ci95", "top5_contains_acceptable_mean", "top5_contains_acceptable_ci95"]
    print(core[cols].sort_values("mean_probability_mass_in_set_mean", ascending=False).to_string(index=False))
    print()
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
