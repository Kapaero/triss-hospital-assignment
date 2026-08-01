"""
Extended eICU choice-model comparison.

This script broadens the first discrete-choice experiment:

* conditional logit;
* pairwise MLP;
* pairwise MLP with hospital-specific bias;
* temporal LSTM choice model using early vital-sign sequences;
* rule-based baselines;
* probability-list, ambiguity, and acceptable-core metrics;
* SVG figures for article drafts.

Run:
  python eicu_choice_model_comparison.py
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
    acceptable_sets,
    baseline_scores,
    build_choice_data,
    build_hospital_profiles,
    fit_conditional_logit,
    hospital_coordinates,
    infer_need_matrix,
    infer_unit_matrix,
    metrics_from_scores,
    probability_list_metrics,
    softmax,
    standardize,
    stratified_split,
)


OUT = ROOT / "results_eicu_choice_comparison"
EICU_ROOT = ROOT / "data" / "eicu-crd-demo" / "eicu-collaborative-research-database-demo-2.0.1"
VITAL_COLUMNS = (
    "heartrate",
    "respiration",
    "systemicsystolic",
    "systemicdiastolic",
    "systemicmean",
    "sao2",
    "temperature",
)


def adam_update(
    params: dict[str, np.ndarray],
    grads: dict[str, np.ndarray],
    state: dict[str, dict[str, np.ndarray]],
    step: int,
    lr: float,
) -> None:
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    for name, param in params.items():
        grad = grads[name]
        if name not in state:
            state[name] = {"m": np.zeros_like(param), "v": np.zeros_like(param)}
        item = state[name]
        item["m"] = beta1 * item["m"] + (1.0 - beta1) * grad
        item["v"] = beta2 * item["v"] + (1.0 - beta2) * (grad * grad)
        m_hat = item["m"] / (1.0 - beta1**step)
        v_hat = item["v"] / (1.0 - beta2**step)
        param -= lr * m_hat / (np.sqrt(v_hat) + eps)


def fit_pair_mlp(
    x: np.ndarray,
    y: np.ndarray,
    hidden: int,
    epochs: int,
    learning_rate: float,
    l2: float,
    seed: int,
    hospital_bias: bool,
) -> tuple[dict[str, np.ndarray], list[dict[str, float]]]:
    rng = np.random.default_rng(seed)
    n, m, d = x.shape
    params = {
        "W": rng.normal(0, math.sqrt(2.0 / d), size=(d, hidden)),
        "b": np.zeros(hidden),
        "v": rng.normal(0, math.sqrt(2.0 / hidden), size=hidden),
        "hb": np.zeros(m),
    }
    if not hospital_bias:
        params["hb"][:] = 0.0
    state: dict[str, dict[str, np.ndarray]] = {}
    rows = np.arange(n)
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        a = np.tanh(np.einsum("ijd,dh->ijh", x, params["W"]) + params["b"])
        scores = np.einsum("ijh,h->ij", a, params["v"])
        if hospital_bias:
            scores = scores + params["hb"]
        probs = softmax(scores)
        chosen = np.clip(probs[rows, y], 1e-12, 1.0)
        loss = float(-np.log(chosen).mean())
        loss += 0.5 * l2 * (float((params["W"] ** 2).sum()) + float((params["v"] ** 2).sum()))
        if hospital_bias:
            loss += 0.5 * l2 * float((params["hb"] ** 2).sum())

        ds = probs
        ds[rows, y] -= 1.0
        ds /= n
        grads: dict[str, np.ndarray] = {}
        grads["v"] = np.einsum("ij,ijh->h", ds, a) + l2 * params["v"]
        da = ds[:, :, None] * params["v"][None, None, :] * (1.0 - a * a)
        grads["W"] = np.einsum("ijd,ijh->dh", x, da) + l2 * params["W"]
        grads["b"] = da.sum(axis=(0, 1))
        grads["hb"] = ds.sum(axis=0) + (l2 * params["hb"] if hospital_bias else 0.0)
        if not hospital_bias:
            grads["hb"][:] = 0.0
        adam_update(params, grads, state, epoch, learning_rate)
        if not hospital_bias:
            params["hb"][:] = 0.0
        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            history.append({"epoch": epoch, "train_nll": loss})
    return params, history


def predict_pair_mlp(x: np.ndarray, params: dict[str, np.ndarray], hospital_bias: bool) -> np.ndarray:
    a = np.tanh(np.einsum("ijd,dh->ijh", x, params["W"]) + params["b"])
    scores = np.einsum("ijh,h->ij", a, params["v"])
    if hospital_bias:
        scores = scores + params["hb"]
    return scores


def build_vital_sequences(
    features: pd.DataFrame,
    bins: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vital_path = EICU_ROOT / "vitalPeriodic.csv.gz"
    stay_ids = features["stay_id"].astype(int).to_numpy()
    stay_to_pos = {int(stay): pos for pos, stay in enumerate(stay_ids)}
    usecols = ["patientunitstayid", "observationoffset", *VITAL_COLUMNS]
    vital = pd.read_csv(vital_path, usecols=usecols)
    vital = vital[vital["patientunitstayid"].isin(stay_to_pos)]
    vital["observationoffset"] = pd.to_numeric(vital["observationoffset"], errors="coerce")
    vital = vital[(vital["observationoffset"] >= 0) & (vital["observationoffset"] < bins * 60)]
    vital["bin"] = (vital["observationoffset"] // 60).astype(int).clip(0, bins - 1)
    agg = vital.groupby(["patientunitstayid", "bin"])[list(VITAL_COLUMNS)].median().reset_index()

    seq = np.full((len(features), bins, len(VITAL_COLUMNS)), np.nan, dtype=float)
    for row in agg.itertuples(index=False):
        pos = stay_to_pos[int(row.patientunitstayid)]
        b = int(row.bin)
        seq[pos, b, :] = np.array([getattr(row, col) for col in VITAL_COLUMNS], dtype=float)

    # Fill missing values with patient medians, then with global medians.
    global_median = np.nanmedian(seq.reshape(-1, len(VITAL_COLUMNS)), axis=0)
    global_median = np.where(np.isfinite(global_median), global_median, 0.0)
    for i in range(seq.shape[0]):
        patient_median = np.nanmedian(seq[i], axis=0)
        patient_median = np.where(np.isfinite(patient_median), patient_median, global_median)
        missing = ~np.isfinite(seq[i])
        seq[i][missing] = np.take(patient_median, np.where(missing)[1])

    rng = np.random.default_rng(seed)
    seq = seq + rng.normal(0, 1e-5, size=seq.shape)
    mean = seq.reshape(-1, len(VITAL_COLUMNS)).mean(axis=0)
    std = seq.reshape(-1, len(VITAL_COLUMNS)).std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return seq, mean, std


def lstm_forward(xseq: np.ndarray, params: dict[str, np.ndarray]) -> tuple[np.ndarray, list[tuple[np.ndarray, ...]]]:
    n, _, f = xseq.shape
    hdim = params["lstm_b"].shape[0] // 4
    h = np.zeros((n, hdim))
    c = np.zeros((n, hdim))
    caches: list[tuple[np.ndarray, ...]] = []
    for t in range(xseq.shape[1]):
        x_t = xseq[:, t, :]
        z = np.concatenate([x_t, h], axis=1)
        gates = z @ params["lstm_W"] + params["lstm_b"]
        i = 1.0 / (1.0 + np.exp(-np.clip(gates[:, :hdim], -50, 50)))
        ff = 1.0 / (1.0 + np.exp(-np.clip(gates[:, hdim : 2 * hdim], -50, 50)))
        o = 1.0 / (1.0 + np.exp(-np.clip(gates[:, 2 * hdim : 3 * hdim], -50, 50)))
        g = np.tanh(gates[:, 3 * hdim :])
        c_prev = c
        h_prev = h
        c = ff * c + i * g
        h = o * np.tanh(c)
        caches.append((z, i, ff, o, g, c, c_prev, h_prev))
    return h, caches


def lstm_backward(
    dh: np.ndarray,
    caches: list[tuple[np.ndarray, ...]],
    params: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    hdim = dh.shape[1]
    fdim = params["lstm_W"].shape[0] - hdim
    dW = np.zeros_like(params["lstm_W"])
    db = np.zeros_like(params["lstm_b"])
    dh_next = dh
    dc_next = np.zeros_like(dh)
    for z, i, ff, o, g, c, c_prev, _h_prev in reversed(caches):
        tanh_c = np.tanh(c)
        do = dh_next * tanh_c
        dc = dh_next * o * (1.0 - tanh_c * tanh_c) + dc_next
        df = dc * c_prev
        di = dc * g
        dg = dc * i
        dc_next = dc * ff
        di_raw = di * i * (1.0 - i)
        df_raw = df * ff * (1.0 - ff)
        do_raw = do * o * (1.0 - o)
        dg_raw = dg * (1.0 - g * g)
        dgate = np.concatenate([di_raw, df_raw, do_raw, dg_raw], axis=1)
        dW += z.T @ dgate
        db += dgate.sum(axis=0)
        dz = dgate @ params["lstm_W"].T
        dh_next = dz[:, fdim:]
    return {"lstm_W": dW, "lstm_b": db}


def fit_temporal_lstm_choice(
    xpair: np.ndarray,
    xseq: np.ndarray,
    y: np.ndarray,
    pair_hidden: int,
    lstm_hidden: int,
    epochs: int,
    learning_rate: float,
    l2: float,
    seed: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, float]]]:
    rng = np.random.default_rng(seed)
    n, m, d = xpair.shape
    f = xseq.shape[2]
    params = {
        "lstm_W": rng.normal(0, math.sqrt(1.0 / (f + lstm_hidden)), size=(f + lstm_hidden, 4 * lstm_hidden)),
        "lstm_b": np.zeros(4 * lstm_hidden),
        "Wz": rng.normal(0, math.sqrt(2.0 / d), size=(d, pair_hidden)),
        "Wh": rng.normal(0, math.sqrt(2.0 / lstm_hidden), size=(lstm_hidden, pair_hidden)),
        "b": np.zeros(pair_hidden),
        "v": rng.normal(0, math.sqrt(2.0 / pair_hidden), size=pair_hidden),
        "hb": np.zeros(m),
    }
    state: dict[str, dict[str, np.ndarray]] = {}
    rows = np.arange(n)
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        h, caches = lstm_forward(xseq, params)
        z_part = np.einsum("ijd,dh->ijh", xpair, params["Wz"])
        h_part = h @ params["Wh"]
        a = np.tanh(z_part + h_part[:, None, :] + params["b"])
        scores = np.einsum("ijh,h->ij", a, params["v"]) + params["hb"]
        probs = softmax(scores)
        chosen = np.clip(probs[rows, y], 1e-12, 1.0)
        loss = float(-np.log(chosen).mean())
        loss += 0.5 * l2 * (
            float((params["Wz"] ** 2).sum())
            + float((params["Wh"] ** 2).sum())
            + float((params["v"] ** 2).sum())
            + float((params["hb"] ** 2).sum())
            + float((params["lstm_W"] ** 2).sum())
        )

        ds = probs
        ds[rows, y] -= 1.0
        ds /= n
        grads: dict[str, np.ndarray] = {}
        grads["v"] = np.einsum("ij,ijh->h", ds, a) + l2 * params["v"]
        da = ds[:, :, None] * params["v"][None, None, :] * (1.0 - a * a)
        grads["Wz"] = np.einsum("ijd,ijh->dh", xpair, da) + l2 * params["Wz"]
        dh_embed = np.einsum("ijh,lh->il", da, params["Wh"])
        grads["Wh"] = h.T @ da.sum(axis=1) + l2 * params["Wh"]
        grads["b"] = da.sum(axis=(0, 1))
        grads["hb"] = ds.sum(axis=0) + l2 * params["hb"]
        lstm_grads = lstm_backward(dh_embed, caches, params)
        grads["lstm_W"] = lstm_grads["lstm_W"] + l2 * params["lstm_W"]
        grads["lstm_b"] = lstm_grads["lstm_b"]
        adam_update(params, grads, state, epoch, learning_rate)
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            history.append({"epoch": epoch, "train_nll": loss})
    return params, history


def predict_temporal_lstm_choice(xpair: np.ndarray, xseq: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
    h, _ = lstm_forward(xseq, params)
    a = np.tanh(
        np.einsum("ijd,dh->ijh", xpair, params["Wz"])
        + (h @ params["Wh"])[:, None, :]
        + params["b"]
    )
    return np.einsum("ijh,h->ij", a, params["v"]) + params["hb"]


def ambiguity_frame(choice, scores: np.ndarray, model: str) -> pd.DataFrame:
    probs = softmax(scores)
    order = np.argsort(-scores, axis=1)
    top = order[:, 0]
    second = order[:, 1]
    rows = np.arange(len(top))
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)
    core = acceptable_sets(choice)["clinical_transport_capacity_core"]
    near_prob = probs >= 0.8 * probs[rows, top][:, None]
    near_utility = scores >= scores[rows, top][:, None] - 0.25
    return pd.DataFrame(
        {
            "model": model,
            "patient_stay_id": choice.patients["stay_id"].astype(int).to_numpy(),
            "top1_probability": probs[rows, top],
            "top2_probability": probs[rows, second],
            "top1_top2_gap": probs[rows, top] - probs[rows, second],
            "top1_top2_ratio": probs[rows, top] / np.maximum(probs[rows, second], 1e-12),
            "top3_probability_mass": np.take_along_axis(probs, order[:, :3], axis=1).sum(axis=1),
            "top5_probability_mass": np.take_along_axis(probs, order[:, :5], axis=1).sum(axis=1),
            "top10_probability_mass": np.take_along_axis(probs, order[:, :10], axis=1).sum(axis=1),
            "effective_hospitals": np.exp(entropy),
            "near_equivalent_count_prob80": near_prob.sum(axis=1),
            "near_equivalent_count_utility025": near_utility.sum(axis=1),
            "acceptable_core_size": core.sum(axis=1),
            "acceptable_core_probability_mass": (probs * core).sum(axis=1),
            "top1_in_acceptable_core": core[rows, top].astype(int),
        }
    )


def ambiguity_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    numeric = [
        "top1_probability",
        "top1_top2_gap",
        "top3_probability_mass",
        "top5_probability_mass",
        "effective_hospitals",
        "near_equivalent_count_prob80",
        "near_equivalent_count_utility025",
        "acceptable_core_size",
        "acceptable_core_probability_mass",
        "top1_in_acceptable_core",
    ]
    for model, part in frame.groupby("model"):
        row: dict[str, float | str] = {"model": model}
        for col in numeric:
            row[f"{col}_mean"] = float(part[col].mean())
            row[f"{col}_median"] = float(part[col].median())
            row[f"{col}_p25"] = float(part[col].quantile(0.25))
            row[f"{col}_p75"] = float(part[col].quantile(0.75))
        rows.append(row)
    return pd.DataFrame(rows)


def coverage_curve(choice, scores_by_model: dict[str, np.ndarray], max_k: int = 20) -> pd.DataFrame:
    core = acceptable_sets(choice)["clinical_transport_capacity_core"]
    rows = np.arange(core.shape[0])
    records = []
    for model, scores in scores_by_model.items():
        order = np.argsort(-scores, axis=1)
        for k in range(1, max_k + 1):
            selected = order[:, :k]
            records.append(
                {
                    "model": model,
                    "k": k,
                    "coverage": float(core[rows[:, None], selected].any(axis=1).mean()),
                    "precision": float(core[rows[:, None], selected].sum(axis=1).mean() / k),
                }
            )
    return pd.DataFrame(records)


def svg_polyline(points: list[tuple[float, float]], color: str) -> str:
    text = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{text}" fill="none" stroke="{color}" stroke-width="2.5"/>'


def save_topk_svg(metrics: pd.DataFrame, path: Path) -> None:
    plot = metrics[metrics["model"].str.contains("test|baseline")].copy()
    plot = plot[["model", "top1_accuracy", "top5_accuracy", "top10_accuracy", "mean_reciprocal_rank"]]
    width, height = 1040, 520
    left, top, bar_w, row_h = 260, 50, 600, 38
    colors = ["#2d6cdf", "#3a9b63", "#c77730", "#6a57b8"]
    metric_names = ["top1_accuracy", "top5_accuracy", "top10_accuracy", "mean_reciprocal_rank"]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="30" font-family="Arial" font-size="20" font-weight="700">Model comparison by rank metrics</text>',
    ]
    for tick in np.linspace(0, 1, 6):
        x = left + tick * bar_w
        lines.append(f'<line x1="{x:.1f}" y1="{top-8}" x2="{x:.1f}" y2="{height-45}" stroke="#dddddd"/>')
        lines.append(f'<text x="{x-10:.1f}" y="{height-20}" font-family="Arial" font-size="12" fill="#555">{tick:.1f}</text>')
    for r, row in enumerate(plot.itertuples(index=False)):
        y = top + r * row_h
        lines.append(f'<text x="24" y="{y+18}" font-family="Arial" font-size="12" fill="#222">{row.model}</text>')
        for mi, metric in enumerate(metric_names):
            value = float(getattr(row, metric))
            yy = y + mi * 8
            lines.append(f'<rect x="{left}" y="{yy}" width="{value * bar_w:.1f}" height="6" fill="{colors[mi]}"/>')
    legend_x = width - 170
    for mi, metric in enumerate(metric_names):
        y = 56 + mi * 22
        lines.append(f'<rect x="{legend_x}" y="{y-11}" width="14" height="8" fill="{colors[mi]}"/>')
        lines.append(f'<text x="{legend_x+20}" y="{y}" font-family="Arial" font-size="12">{metric}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def save_coverage_svg(curve: pd.DataFrame, path: Path) -> None:
    selected_models = [
        "conditional_logit_test",
        "pair_mlp_test",
        "pair_mlp_hospital_bias_test",
        "temporal_lstm_choice_test",
        "baseline_nearest_synthetic",
        "baseline_simple_rule",
    ]
    colors = ["#2d6cdf", "#d24c4c", "#7b59c0", "#209070", "#d68a18", "#666666"]
    width, height = 920, 520
    left, top, plot_w, plot_h = 80, 55, 650, 380
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="30" font-family="Arial" font-size="20" font-weight="700">Acceptable-core coverage by top-k list</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fafafa" stroke="#cccccc"/>',
    ]
    for tick in np.linspace(0, 1, 6):
        y = top + plot_h - tick * plot_h
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#e0e0e0"/>')
        lines.append(f'<text x="38" y="{y+4:.1f}" font-family="Arial" font-size="12">{tick:.1f}</text>')
    for tick in [1, 5, 10, 15, 20]:
        x = left + (tick - 1) / 19 * plot_w
        lines.append(f'<text x="{x-7:.1f}" y="{top+plot_h+24}" font-family="Arial" font-size="12">{tick}</text>')
    for idx, model in enumerate(selected_models):
        part = curve[curve["model"] == model]
        if part.empty:
            continue
        points = [
            (left + (float(r.k) - 1.0) / 19.0 * plot_w, top + plot_h - float(r.coverage) * plot_h)
            for r in part.itertuples(index=False)
        ]
        lines.append(svg_polyline(points, colors[idx]))
        ly = 68 + idx * 24
        lines.append(f'<line x1="755" y1="{ly-4}" x2="780" y2="{ly-4}" stroke="{colors[idx]}" stroke-width="3"/>')
        lines.append(f'<text x="786" y="{ly}" font-family="Arial" font-size="12">{model}</text>')
    lines.append('<text x="365" y="500" font-family="Arial" font-size="13">k</text>')
    lines.append('<text x="18" y="245" font-family="Arial" font-size="13" transform="rotate(-90 18,245)">coverage</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def save_ambiguity_svg(frame: pd.DataFrame, model: str, path: Path) -> None:
    values = frame.loc[frame["model"] == model, "near_equivalent_count_utility025"].to_numpy(float)
    bins = np.arange(1, min(26, int(values.max()) + 3))
    counts = np.array([((values >= b) & (values < b + 1)).sum() for b in bins], dtype=float)
    if counts.max() <= 0:
        counts += 1
    width, height = 820, 420
    left, top, plot_w, plot_h = 70, 45, 640, 300
    bw = plot_w / len(bins)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="24" y="28" font-family="Arial" font-size="20" font-weight="700">Near-equivalent hospitals: {model}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fafafa" stroke="#cccccc"/>',
    ]
    for i, b in enumerate(bins):
        h = counts[i] / counts.max() * plot_h
        x = left + i * bw
        y = top + plot_h - h
        lines.append(f'<rect x="{x+1:.1f}" y="{y:.1f}" width="{max(bw-2,1):.1f}" height="{h:.1f}" fill="#3b77bd"/>')
        if i % 3 == 0:
            lines.append(f'<text x="{x+2:.1f}" y="{top+plot_h+20}" font-family="Arial" font-size="11">{int(b)}</text>')
    lines.append('<text x="260" y="395" font-family="Arial" font-size="13">number of near-equivalent hospitals</text>')
    lines.append('<text x="18" y="210" font-family="Arial" font-size="13" transform="rotate(-90 18,210)">patients</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(args.features).reset_index(drop=True)
    hospitals = pd.read_csv(args.hospitals)
    features = features.dropna(subset=["subject_id"]).copy()
    features["subject_id"] = features["subject_id"].astype(int)
    features["_source_index"] = np.arange(len(features))

    counts = features["subject_id"].value_counts()
    eligible = counts[counts >= args.min_cases_per_hospital].index.astype(int).to_numpy()
    if args.max_hospitals > 0:
        eligible = counts.loc[eligible].sort_values(ascending=False).head(args.max_hospitals).index.astype(int).to_numpy()
    eligible = np.array(sorted(eligible), dtype=int)
    data = features[features["subject_id"].isin(eligible)].reset_index(drop=True).copy()
    data["_source_index"] = data["_source_index"].astype(int)
    train_idx, test_idx = stratified_split(data, "subject_id", args.test_fraction, args.seed)
    train = data.iloc[train_idx].reset_index(drop=True).copy()
    test = data.iloc[test_idx].reset_index(drop=True).copy()

    full_need = infer_need_matrix(features)
    full_unit = infer_unit_matrix(features)
    train_need = full_need[train["_source_index"].to_numpy()]
    train_unit = full_unit[train["_source_index"].to_numpy()]
    hospital_ids = np.array(sorted(data["subject_id"].unique()), dtype=int)
    profiles = build_hospital_profiles(train, hospitals, hospital_ids, train_need, train_unit, alpha=args.profile_smoothing)
    rng = np.random.default_rng(args.seed)
    coords = hospital_coordinates(hospitals[hospitals["hospitalid"].isin(hospital_ids)], rng)

    train_choice = build_choice_data(train, hospitals, profiles, full_need, full_unit, hospital_ids, coords, args.seed + 11, args.catchment_sigma)
    test_choice = build_choice_data(test, hospitals, profiles, full_need, full_unit, hospital_ids, coords, args.seed + 29, args.catchment_sigma)
    x_train, other, feature_mean, feature_std = standardize(train_choice.pair_features, test_choice.pair_features)
    x_test = other[0]

    metrics = []
    score_outputs: dict[str, np.ndarray] = {}
    histories: dict[str, pd.DataFrame] = {}

    theta, hist = fit_conditional_logit(x_train, train_choice.actual_index, args.linear_epochs, args.linear_lr, args.l2, args.seed + 101)
    linear_train = np.einsum("ijf,f->ij", x_train, theta)
    linear_test = np.einsum("ijf,f->ij", x_test, theta)
    metrics.append(metrics_from_scores(linear_train, train_choice.actual_index, "conditional_logit_train"))
    metrics.append(metrics_from_scores(linear_test, test_choice.actual_index, "conditional_logit_test"))
    score_outputs["conditional_logit_test"] = linear_test
    histories["conditional_logit"] = pd.DataFrame(hist)

    mlp_params, hist = fit_pair_mlp(x_train, train_choice.actual_index, args.mlp_hidden, args.mlp_epochs, args.mlp_lr, args.l2, args.seed + 201, False)
    mlp_train = predict_pair_mlp(x_train, mlp_params, False)
    mlp_test = predict_pair_mlp(x_test, mlp_params, False)
    metrics.append(metrics_from_scores(mlp_train, train_choice.actual_index, "pair_mlp_train"))
    metrics.append(metrics_from_scores(mlp_test, test_choice.actual_index, "pair_mlp_test"))
    score_outputs["pair_mlp_test"] = mlp_test
    histories["pair_mlp"] = pd.DataFrame(hist)

    bias_params, hist = fit_pair_mlp(x_train, train_choice.actual_index, args.mlp_hidden, args.mlp_epochs, args.mlp_lr, args.l2, args.seed + 301, True)
    bias_train = predict_pair_mlp(x_train, bias_params, True)
    bias_test = predict_pair_mlp(x_test, bias_params, True)
    metrics.append(metrics_from_scores(bias_train, train_choice.actual_index, "pair_mlp_hospital_bias_train"))
    metrics.append(metrics_from_scores(bias_test, test_choice.actual_index, "pair_mlp_hospital_bias_test"))
    score_outputs["pair_mlp_hospital_bias_test"] = bias_test
    histories["pair_mlp_hospital_bias"] = pd.DataFrame(hist)

    sequence, seq_mean, seq_std = build_vital_sequences(features, args.sequence_bins, args.seed + 401)
    seq_z = (sequence - seq_mean) / seq_std
    seq_train = seq_z[train["_source_index"].to_numpy()]
    seq_test = seq_z[test["_source_index"].to_numpy()]
    lstm_params, hist = fit_temporal_lstm_choice(
        x_train,
        seq_train,
        train_choice.actual_index,
        args.lstm_pair_hidden,
        args.lstm_hidden,
        args.lstm_epochs,
        args.lstm_lr,
        args.l2,
        args.seed + 501,
    )
    lstm_train = predict_temporal_lstm_choice(x_train, seq_train, lstm_params)
    lstm_test = predict_temporal_lstm_choice(x_test, seq_test, lstm_params)
    metrics.append(metrics_from_scores(lstm_train, train_choice.actual_index, "temporal_lstm_choice_train"))
    metrics.append(metrics_from_scores(lstm_test, test_choice.actual_index, "temporal_lstm_choice_test"))
    score_outputs["temporal_lstm_choice_test"] = lstm_test
    histories["temporal_lstm_choice"] = pd.DataFrame(hist)

    train_counts = train["subject_id"].value_counts()
    for name, scores in baseline_scores(test_choice, train_counts).items():
        metrics.append(metrics_from_scores(scores, test_choice.actual_index, name))
        score_outputs[name] = scores

    metrics_df = pd.DataFrame(metrics)
    list_metrics = pd.concat(
        [probability_list_metrics(test_choice, scores, name) for name, scores in score_outputs.items()],
        ignore_index=True,
    )
    ambiguity = pd.concat(
        [ambiguity_frame(test_choice, scores, name) for name, scores in score_outputs.items()],
        ignore_index=True,
    )
    amb_summary = ambiguity_summary(ambiguity)
    curve = coverage_curve(test_choice, score_outputs, max_k=20)

    metrics_df.to_csv(OUT / "model_comparison_metrics.csv", index=False)
    list_metrics.to_csv(OUT / "probability_list_metrics.csv", index=False)
    ambiguity.to_csv(OUT / "ambiguity_by_patient.csv", index=False)
    amb_summary.to_csv(OUT / "ambiguity_summary.csv", index=False)
    curve.to_csv(OUT / "acceptable_core_coverage_curve.csv", index=False)
    profiles.to_csv(OUT / "hospital_profiles_used.csv", index=False)
    pd.DataFrame({"feature": FEATURE_NAMES, "mean": feature_mean, "std": feature_std}).to_csv(
        OUT / "pair_feature_standardization.csv",
        index=False,
    )
    pd.DataFrame({"vital_feature": VITAL_COLUMNS, "mean": seq_mean, "std": seq_std}).to_csv(
        OUT / "sequence_feature_standardization.csv",
        index=False,
    )
    for name, hist in histories.items():
        hist.to_csv(OUT / f"training_history_{name}.csv", index=False)

    save_topk_svg(metrics_df, OUT / "figure_model_topk_comparison.svg")
    save_coverage_svg(curve, OUT / "figure_acceptable_core_coverage.svg")
    core_rows = list_metrics[list_metrics["metric_group"] == "clinical_transport_capacity_core"].copy()
    if not core_rows.empty:
        best_model = str(core_rows.sort_values("mean_probability_mass_in_set", ascending=False).iloc[0]["model"])
    else:
        best_model = "conditional_logit_test"
    save_ambiguity_svg(ambiguity, best_model, OUT / "figure_near_equivalent_hospitals.svg")

    protocol = {
        "features": str(Path(args.features).resolve()),
        "hospitals": str(Path(args.hospitals).resolve()),
        "seed": args.seed,
        "train_patients": int(len(train)),
        "test_patients": int(len(test)),
        "candidate_hospitals": int(len(hospital_ids)),
        "death_rows_total": int(data["hospital_expire_flag"].sum()),
        "death_rate_total": float(data["hospital_expire_flag"].mean()),
        "models": list(score_outputs.keys()),
        "note": "Probabilities are softmax-normalized over candidate hospitals. Multiple near-equivalent hospitals are expected when pairwise utilities are close.",
    }
    (OUT / "comparison_protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")

    print(metrics_df.to_string(index=False))
    print()
    print("Acceptable-core probability mass:")
    print(core_rows[["model", "mean_probability_mass_in_set", "top5_contains_acceptable", "top10_contains_acceptable"]].to_string(index=False))
    print()
    print("Ambiguity summary:")
    print(amb_summary[["model", "effective_hospitals_mean", "near_equivalent_count_utility025_mean", "acceptable_core_probability_mass_mean"]].to_string(index=False))
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
    parser.add_argument("--l2", type=float, default=0.002)
    parser.add_argument("--linear-epochs", type=int, default=550)
    parser.add_argument("--linear-lr", type=float, default=0.045)
    parser.add_argument("--mlp-hidden", type=int, default=28)
    parser.add_argument("--mlp-epochs", type=int, default=320)
    parser.add_argument("--mlp-lr", type=float, default=0.025)
    parser.add_argument("--sequence-bins", type=int, default=6)
    parser.add_argument("--lstm-hidden", type=int, default=10)
    parser.add_argument("--lstm-pair-hidden", type=int, default=22)
    parser.add_argument("--lstm-epochs", type=int, default=140)
    parser.add_argument("--lstm-lr", type=float, default=0.018)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
