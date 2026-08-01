from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from patient_allocation_experiment import (
    INJURY_TYPES,
    greedy_assignment,
    make_hospitals,
    make_patients_from_features,
    nearest_assignment,
    optimize_assignment,
    survival_matrix,
    travel_minutes,
)


ROOT = Path(__file__).resolve().parent
FEATURES = ROOT / "results" / "mimic_dynamic_risk_features.csv"
OUT = ROOT / "results_probabilistic"
SCENARIOS = ["normal", "mass_casualty", "specialty_shortage"]
POLICIES = [
    "nearest",
    "greedy_utility",
    "teacher_optimizer",
    "neural_no_capacity",
    "neural_capacity_softmax",
    "neural_vector_optimizer",
    "oracle_true",
]
BASE_SEED = 20260624
TARGET_TEMPERATURE = 0.070
TARGET_TEACHER_BONUS = 0.35
TARGET_UNIFORM_MIX = 0.10
PROBABILITY_TEMPERATURE = 1.85


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def one_hot(values: np.ndarray, categories: np.ndarray = INJURY_TYPES) -> np.ndarray:
    return (values[:, None] == categories[None, :]).astype(float)


@dataclass
class Instance:
    scenario: str
    seed: int
    patients: pd.DataFrame
    hospitals: pd.DataFrame
    travel: np.ndarray
    pair_features: np.ndarray
    teacher_assignment: np.ndarray
    oracle_assignment: np.ndarray
    true_utility: np.ndarray
    pred_utility: np.ndarray


class PairwiseScorer(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 96),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, hospitals, dims = x.shape
        return self.net(x.reshape(batch * hospitals, dims)).reshape(batch, hospitals)


def build_pair_features(
    patients: pd.DataFrame,
    hospitals: pd.DataFrame,
    travel: np.ndarray,
    pred_utility: np.ndarray,
) -> np.ndarray:
    n, h = travel.shape
    p_type = patients["injury_type"].to_numpy()
    h_type = hospitals["hospital_type"].to_numpy()
    p_type_oh = one_hot(p_type)
    h_type_oh = one_hot(h_type)

    age = patients["age"].to_numpy(float) / 100.0
    iss = patients["iss_proxy"].to_numpy(float) / 50.0
    shock = patients["shock_index"].to_numpy(float) / 2.5
    vital = patients["vital_score"].to_numpy(float)
    pred_base = patients["pred_base_survival"].to_numpy(float)
    pred_hazard = patients["pred_hazard"].to_numpy(float)
    deadline = patients["deadline_min"].to_numpy(float) / 360.0
    severe = ((patients["iss_proxy"].to_numpy(float) >= 25) | (patients["pred_base_survival"].to_numpy(float) < 0.65)).astype(float)

    capacity = hospitals["capacity"].to_numpy(float)
    cap_share = capacity / max(float(n), 1.0)
    quality = hospitals["quality"].to_numpy(float)
    hx = hospitals["x"].to_numpy(float) / 60.0
    hy = hospitals["y"].to_numpy(float) / 60.0
    px = patients["x"].to_numpy(float) / 60.0
    py = patients["y"].to_numpy(float) / 60.0

    match = np.where(p_type[:, None] == h_type[None, :], 1.0, 0.82)
    severe_mismatch = severe[:, None] * (1.0 - (p_type[:, None] == h_type[None, :]).astype(float))

    patient_block = np.column_stack([age, iss, shock, vital, pred_base, pred_hazard, deadline, severe, px, py, p_type_oh])
    hospital_block = np.column_stack([cap_share, quality, hx, hy, h_type_oh])
    pair_rows = []
    for i in range(n):
        p_rep = np.repeat(patient_block[i : i + 1], h, axis=0)
        pair_specific = np.column_stack(
            [
                travel[i] / 120.0,
                match[i],
                severe_mismatch[i],
                pred_utility[i],
                np.log1p(capacity) / math.log1p(max(capacity.max(), 1.0)),
            ]
        )
        pair_rows.append(np.column_stack([p_rep, hospital_block, pair_specific]))
    return np.asarray(pair_rows, dtype=np.float32)


def generate_instance(
    feature_table: pd.DataFrame,
    scenario: str,
    seed: int,
    patients_base: int = 140,
    hospitals_count: int = 8,
) -> Instance:
    rng = np.random.default_rng(seed)
    n = int(patients_base * 1.35) if scenario == "mass_casualty" else patients_base
    patients = make_patients_from_features(feature_table, n, rng, scenario)
    hospitals = make_hospitals(n, hospitals_count, rng, scenario)
    travel = travel_minutes(patients, hospitals, rng, scenario)
    capacities = hospitals["capacity"].to_numpy()
    pred_utility = survival_matrix(patients, hospitals, travel, predicted=True, dynamic=True)
    true_utility = survival_matrix(patients, hospitals, travel, predicted=False, dynamic=True)
    teacher_assignment = optimize_assignment(pred_utility, capacities)
    oracle_assignment = optimize_assignment(true_utility, capacities)
    pair_features = build_pair_features(patients, hospitals, travel, pred_utility)
    return Instance(
        scenario=scenario,
        seed=seed,
        patients=patients,
        hospitals=hospitals,
        travel=travel,
        pair_features=pair_features,
        teacher_assignment=teacher_assignment,
        oracle_assignment=oracle_assignment,
        true_utility=true_utility,
        pred_utility=pred_utility,
    )


def make_instances(
    feature_table: pd.DataFrame,
    train_reps: int,
    test_reps: int,
    patients_base: int,
    hospitals_count: int,
) -> tuple[list[Instance], list[Instance]]:
    train, test = [], []
    for s_idx, scenario in enumerate(SCENARIOS):
        for rep in range(train_reps):
            seed = BASE_SEED + 10_000 * s_idx + rep
            train.append(generate_instance(feature_table, scenario, seed, patients_base, hospitals_count))
        for rep in range(test_reps):
            seed = BASE_SEED + 10_000 * s_idx + 1000 + rep
            test.append(generate_instance(feature_table, scenario, seed, patients_base, hospitals_count))
    return train, test


def soft_teacher_distribution(inst: Instance) -> np.ndarray:
    logits = inst.pred_utility / TARGET_TEMPERATURE
    logits[np.arange(len(inst.teacher_assignment)), inst.teacher_assignment] += TARGET_TEACHER_BONUS
    z = logits - logits.max(axis=1, keepdims=True)
    q = np.exp(z)
    q = q / q.sum(axis=1, keepdims=True)
    q = (1.0 - TARGET_UNIFORM_MIX) * q + TARGET_UNIFORM_MIX / q.shape[1]
    return q.astype(np.float32)


def stack_training(instances: list[Instance]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.concatenate([inst.pair_features for inst in instances], axis=0)
    y = np.concatenate([inst.teacher_assignment for inst in instances], axis=0)
    q = np.concatenate([soft_teacher_distribution(inst) for inst in instances], axis=0)
    mean = x.reshape(-1, x.shape[-1]).mean(axis=0)
    std = x.reshape(-1, x.shape[-1]).std(axis=0) + 1e-7
    return ((x - mean) / std).astype(np.float32), y.astype(np.int64), q, mean.astype(np.float32), std.astype(np.float32)


def soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -(target * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()


def train_model(x: np.ndarray, y: np.ndarray, q: np.ndarray, seed: int = BASE_SEED) -> PairwiseScorer:
    set_seed(seed)
    model = PairwiseScorer(x.shape[-1])
    ds = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long), torch.tensor(q, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=256, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(120):
        for xb, yb, qb in loader:
            logits = model(xb)
            loss = 0.65 * soft_cross_entropy(logits, qb) + 0.35 * loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model


def predict_scores(model: PairwiseScorer, pair_features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    x = ((pair_features - mean) / std).astype(np.float32)
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(x, dtype=torch.float32)).numpy()


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def probability_vector(scores: np.ndarray) -> np.ndarray:
    return softmax(scores, temperature=PROBABILITY_TEMPERATURE)


def assign_from_scores(
    scores: np.ndarray,
    capacities: np.ndarray,
    order: np.ndarray,
    capacity_aware: bool,
    gamma: float = 1.9,
) -> np.ndarray:
    remaining = capacities.astype(int).copy()
    original = capacities.astype(float).copy()
    assignment = np.full(scores.shape[0], -1, dtype=int)
    for i in order:
        adjusted = scores[i].copy() / PROBABILITY_TEMPERATURE
        if capacity_aware:
            availability = (remaining + 0.5) / (original + 0.5)
            adjusted = adjusted + gamma * np.log(np.clip(availability, 1e-6, None))
        adjusted[remaining <= 0] = -1e9
        j = int(np.argmax(adjusted))
        assignment[i] = j
        remaining[j] -= 1
    return assignment


def no_capacity_top1(scores: np.ndarray) -> np.ndarray:
    return np.argmax(scores, axis=1).astype(int)


def evaluate_policy(
    inst: Instance,
    assignment: np.ndarray,
    policy: str,
    scores: np.ndarray | None = None,
) -> dict[str, float | int | str]:
    n = len(inst.patients)
    chosen = inst.true_utility[np.arange(n), assignment]
    loads = np.bincount(assignment, minlength=len(inst.hospitals))
    capacities = inst.hospitals["capacity"].to_numpy()
    p_type = inst.patients["injury_type"].to_numpy()
    h_type = inst.hospitals["hospital_type"].to_numpy()
    profile_match = p_type == h_type[assignment]
    row = {
        "scenario": inst.scenario,
        "seed": inst.seed,
        "policy": policy,
        "mean_survival": float(chosen.mean()),
        "regret_to_oracle": float(inst.true_utility[np.arange(n), inst.oracle_assignment].mean() - chosen.mean()),
        "avg_travel_min": float(inst.travel[np.arange(n), assignment].mean()),
        "profile_match_rate": float(profile_match.mean()),
        "capacity_violations": float(np.maximum(loads - capacities, 0).sum()),
        "max_utilization": float((loads / capacities).max()),
        "top1_teacher_match": float((assignment == inst.teacher_assignment).mean()),
        "top1_true_oracle_match": float((assignment == inst.oracle_assignment).mean()),
    }
    if scores is not None:
        probs = probability_vector(scores)
        top3 = np.argsort(-probs, axis=1)[:, :3]
        row["teacher_top3_rate"] = float(np.mean([inst.teacher_assignment[i] in top3[i] for i in range(n)]))
        row["mean_teacher_prob"] = float(probs[np.arange(n), inst.teacher_assignment].mean())
    else:
        row["teacher_top3_rate"] = float("nan")
        row["mean_teacher_prob"] = float("nan")
    return row


def evaluate_instances(
    model: PairwiseScorer,
    instances: list[Instance],
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    examples = []
    for inst_idx, inst in enumerate(instances):
        scores = predict_scores(model, inst.pair_features, mean, std)
        capacities = inst.hospitals["capacity"].to_numpy()
        severity_order = np.argsort(inst.patients["pred_base_survival"].to_numpy() + 0.005 * inst.patients["deadline_min"].to_numpy())
        assignments = {
            "nearest": nearest_assignment(inst.travel, capacities, np.arange(len(inst.patients))),
            "greedy_utility": greedy_assignment(inst.pred_utility, capacities, severity_order),
            "teacher_optimizer": inst.teacher_assignment,
            "neural_no_capacity": no_capacity_top1(scores),
            "neural_capacity_softmax": assign_from_scores(scores, capacities, severity_order, capacity_aware=True),
            "neural_vector_optimizer": optimize_assignment(scores, capacities),
            "oracle_true": inst.oracle_assignment,
        }
        for policy in POLICIES:
            rows.append(evaluate_policy(inst, assignments[policy], policy, scores if policy.startswith("neural") else None))
        if inst_idx < 3:
            probs = probability_vector(scores)
            second_best = np.partition(probs, -2, axis=1)[:, -2]
            interesting = np.argsort(-second_best)[:8]
            for pid in interesting:
                prob_order = np.argsort(-probs[pid])
                row = {
                    "scenario": inst.scenario,
                    "seed": inst.seed,
                    "patient_id": int(inst.patients.loc[pid, "pid"]),
                    "age": int(round(inst.patients.loc[pid, "age"])),
                    "injury_type": inst.patients.loc[pid, "injury_type"],
                    "base_survival": float(inst.patients.loc[pid, "pred_base_survival"]),
                    "teacher_hospital": f"H{int(inst.teacher_assignment[pid])}",
                    "neural_top1": f"H{int(prob_order[0])}",
                    "top3": "; ".join(f"H{int(j)}:{probs[pid, j]:.3f}/{inst.hospitals.loc[j, 'hospital_type']}/cap{int(inst.hospitals.loc[j, 'capacity'])}" for j in prob_order[:3]),
                }
                for j in range(len(inst.hospitals)):
                    row[f"p_H{j}"] = float(probs[pid, j])
                examples.append(row)
    return pd.DataFrame(rows), pd.DataFrame(examples)


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "mean_survival",
        "regret_to_oracle",
        "avg_travel_min",
        "profile_match_rate",
        "capacity_violations",
        "top1_teacher_match",
        "teacher_top3_rate",
        "mean_teacher_prob",
    ]
    summary = raw.groupby(["scenario", "policy"])[metrics].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(c).strip("_") for c in summary.columns.to_flat_index()]
    return summary


def plot_comparison(summary: pd.DataFrame, out: Path) -> None:
    plot = summary[
        summary["policy"].isin(
            ["nearest", "greedy_utility", "teacher_optimizer", "neural_capacity_softmax", "neural_vector_optimizer", "oracle_true"]
        )
    ].copy()
    labels = {
        "nearest": "Nearest",
        "greedy_utility": "Greedy utility",
        "teacher_optimizer": "Optimizer",
        "neural_capacity_softmax": "Neural online",
        "neural_vector_optimizer": "Neural + optimizer",
        "oracle_true": "True oracle",
    }
    colors = {
        "nearest": "#9aa0a6",
        "greedy_utility": "#5f8db8",
        "teacher_optimizer": "#2f6f73",
        "neural_capacity_softmax": "#b66d3a",
        "neural_vector_optimizer": "#7a5aa6",
        "oracle_true": "#2a2a2a",
    }
    scenarios = SCENARIOS
    policies = list(labels)
    fig, axes = plt.subplots(1, len(scenarios), figsize=(15, 4.8), sharey=True)
    for ax, scenario in zip(axes, scenarios):
        sub = plot[plot["scenario"] == scenario].set_index("policy")
        vals = [sub.loc[p, "mean_survival_mean"] for p in policies]
        errs = [sub.loc[p, "mean_survival_std"] for p in policies]
        ax.bar(np.arange(len(policies)), vals, yerr=errs, color=[colors[p] for p in policies], capsize=3)
        ax.set_title(scenario.replace("_", " "))
        ax.set_xticks(np.arange(len(policies)))
        ax.set_xticklabels([labels[p] for p in policies], rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(0, max(1.02, max(vals) * 1.08))
    axes[0].set_ylabel("Mean true utility")
    fig.suptitle("Probabilistic patient-hospital assignment with capacity-aware neural vector")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_probability_heatmap(examples: pd.DataFrame, out: Path) -> None:
    if examples.empty:
        return
    ex = examples.head(10).copy()
    prob_cols = [c for c in ex.columns if c.startswith("p_H")]
    mat = ex[prob_cols].to_numpy(float)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    im = ax.imshow(mat, cmap="YlGnBu", vmin=0, vmax=max(0.35, mat.max()))
    ax.set_xticks(np.arange(len(prob_cols)))
    ax.set_xticklabels([c.replace("p_", "") for c in prob_cols])
    ax.set_yticks(np.arange(len(ex)))
    ax.set_yticklabels([f"P{r.patient_id} {r.injury_type}" for r in ex.itertuples()])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8, color="#111")
    ax.set_title("Example patient probability vectors over hospitals")
    ax.set_xlabel("Hospital")
    ax.set_ylabel("Patient")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-reps", type=int, default=22)
    parser.add_argument("--test-reps", type=int, default=8)
    parser.add_argument("--patients", type=int, default=130)
    parser.add_argument("--hospitals", type=int, default=8)
    parser.add_argument("--features", type=Path, default=FEATURES)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_table = pd.read_csv(args.features)
    train_instances, test_instances = make_instances(feature_table, args.train_reps, args.test_reps, args.patients, args.hospitals)
    x_train, y_train, q_train, mean, std = stack_training(train_instances)
    model = train_model(x_train, y_train, q_train)
    raw, examples = evaluate_instances(model, test_instances, mean, std)
    summary = summarize(raw)

    raw.to_csv(out_dir / "probabilistic_assignment_raw.csv", index=False)
    summary.to_csv(out_dir / "probabilistic_assignment_summary.csv", index=False)
    examples.to_csv(out_dir / "probability_vector_examples.csv", index=False)
    pd.DataFrame(
        [
            {"parameter": "features", "value": str(args.features)},
            {"parameter": "train_reps_per_scenario", "value": args.train_reps},
            {"parameter": "test_reps_per_scenario", "value": args.test_reps},
            {"parameter": "patients_base", "value": args.patients},
            {"parameter": "hospitals", "value": args.hospitals},
            {"parameter": "seed", "value": BASE_SEED},
            {"parameter": "teacher", "value": "min-cost optimization over predicted dynamic utility with hospital capacity"},
            {"parameter": "soft_teacher_target", "value": f"softmax(predicted utility / {TARGET_TEMPERATURE}) plus teacher bonus {TARGET_TEACHER_BONUS} and uniform mix {TARGET_UNIFORM_MIX}"},
            {"parameter": "neural_output", "value": f"calibrated softmax probability vector over hospitals, temperature {PROBABILITY_TEMPERATURE}"},
        ]
    ).to_csv(out_dir / "experiment_protocol.csv", index=False)

    plot_comparison(summary, out_dir / "probabilistic_assignment_comparison.png")
    plot_probability_heatmap(examples, out_dir / "probability_vector_heatmap.png")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved outputs to {out_dir}")


if __name__ == "__main__":
    main()
