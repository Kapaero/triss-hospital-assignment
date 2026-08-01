from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from patient_allocation_experiment import (
    INJURY_TYPES,
    compatibility,
    greedy_assignment,
    make_patients_from_features,
    nearest_assignment,
    survival_matrix,
)
from probabilistic_hospital_assignment_experiment import (
    BASE_SEED,
    assign_from_scores,
    build_pair_features,
    no_capacity_top1,
    predict_scores,
    probability_vector,
    stack_training,
    train_model,
)

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - fallback for environments without scipy
    linear_sum_assignment = None


ROOT = Path(__file__).resolve().parent
FEATURES = ROOT / "results_external" / "eicu_demo_features.csv"
HOSPITAL_META = ROOT / "results_external" / "eicu_demo_hospitals.csv"
OUT = ROOT / "results_large_scale"

SCENARIOS = [
    "normal",
    "mass_casualty",
    "specialty_shortage",
    "readiness_degradation",
    "transport_disruption",
    "combined_crisis",
]

POLICIES = [
    "nearest",
    "static_optimizer",
    "greedy_dynamic",
    "neural_online",
    "neural_optimizer",
    "oracle_true",
    "neural_no_capacity",
]

SCENARIO_LABELS_RU = {
    "normal": "baseline",
    "mass_casualty": "mass\ncasualty",
    "specialty_shortage": "specialty\nshortage",
    "readiness_degradation": "readiness\ndegradation",
    "transport_disruption": "transport\ndisruption",
    "combined_crisis": "combined\ncrisis",
}

POLICY_LABELS_RU = {
    "nearest": "nearest",
    "static_optimizer": "static opt.",
    "greedy_dynamic": "dyn. rule",
    "neural_online": "neural net",
    "neural_optimizer": "NN + opt.",
    "oracle_true": "oracle",
    "neural_no_capacity": "no capacity",
}


@dataclass
class LargeInstance:
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
    pred_static: np.ndarray


def bed_weight(category: object) -> float:
    text = str(category).strip()
    if text == "<100":
        return 70.0
    if text == "100 - 249":
        return 175.0
    if text == "250 - 499":
        return 375.0
    if text == ">= 500":
        return 600.0
    return 150.0


def scenario_patient_count(base: int, scenario: str) -> int:
    factors = {
        "normal": 1.00,
        "mass_casualty": 1.55,
        "specialty_shortage": 1.10,
        "readiness_degradation": 1.10,
        "transport_disruption": 1.10,
        "combined_crisis": 1.65,
    }
    return int(round(base * factors[scenario]))


def scenario_patient_mode(scenario: str) -> str:
    if scenario in {"mass_casualty", "transport_disruption", "combined_crisis"}:
        return "mass_casualty"
    if scenario == "specialty_shortage":
        return "specialty_shortage"
    return "normal"


def scenario_capacity_factor(scenario: str) -> float:
    return {
        "normal": 1.24,
        "mass_casualty": 1.04,
        "specialty_shortage": 1.08,
        "readiness_degradation": 1.06,
        "transport_disruption": 1.14,
        "combined_crisis": 1.02,
    }[scenario]


def assign_profiles(h: int, rng: np.random.Generator, scenario: str, large_flags: np.ndarray) -> np.ndarray:
    if scenario in {"specialty_shortage", "combined_crisis"}:
        probs = np.array([0.20, 0.06, 0.74])
    else:
        probs = np.array([0.36, 0.14, 0.50])
    profiles = rng.choice(INJURY_TYPES, size=h, p=probs)
    if h >= 3:
        profiles[:3] = INJURY_TYPES
    profiles[large_flags & (rng.random(h) < 0.25)] = "trauma"
    rng.shuffle(profiles)
    return profiles


def scaled_capacities(weights: np.ndarray, total: int) -> np.ndarray:
    raw = weights / weights.sum() * total
    cap = np.maximum(2, np.floor(raw).astype(int))
    while cap.sum() < total:
        cap[int(np.argmax(raw - cap))] += 1
    while cap.sum() > total and cap.max() > 2:
        cap[int(np.argmax(cap - raw))] -= 1
    return cap.astype(int)


def make_hospital_network(
    hospital_meta: pd.DataFrame,
    n_patients: int,
    hospitals_count: int,
    rng: np.random.Generator,
    scenario: str,
) -> pd.DataFrame:
    meta = hospital_meta.copy()
    if len(meta) >= hospitals_count:
        sample = meta.sample(n=hospitals_count, replace=False, random_state=int(rng.integers(0, 2**32 - 1))).reset_index(drop=True)
    else:
        sample = meta.sample(n=hospitals_count, replace=True, random_state=int(rng.integers(0, 2**32 - 1))).reset_index(drop=True)

    weights = sample["numbedscategory"].map(bed_weight).fillna(150).to_numpy(float)
    target_capacity = math.ceil(n_patients * scenario_capacity_factor(scenario))
    capacity = scaled_capacities(weights, target_capacity)

    teaching = sample["teachingstatus"].astype(str).str.lower().eq("t").to_numpy()
    large = weights >= 250
    quality = 0.86 + 0.00035 * weights + 0.035 * teaching.astype(float) + rng.normal(0, 0.035, size=hospitals_count)
    quality = np.clip(quality, 0.82, 1.16)

    if scenario in {"readiness_degradation", "combined_crisis"}:
        degraded = rng.choice(np.arange(hospitals_count), size=max(1, hospitals_count // 4), replace=False)
        quality[degraded] *= rng.uniform(0.76, 0.88, size=len(degraded))
        capacity[degraded] = np.maximum(2, np.floor(capacity[degraded] * rng.uniform(0.72, 0.88, size=len(degraded))).astype(int))
        if capacity.sum() < n_patients:
            capacity += scaled_capacities(weights, n_patients - capacity.sum())

    region_centers = {
        "Midwest": (20.0, 27.0),
        "Northeast": (31.0, 34.0),
        "South": (32.0, 18.0),
        "West": (17.0, 18.0),
    }
    coords = []
    for region in sample["region"].fillna("Midwest"):
        center = region_centers.get(str(region), (25.0, 25.0))
        coords.append(rng.normal(loc=center, scale=(7.5, 7.5), size=2))
    coords = np.asarray(coords)

    hospital_type = assign_profiles(hospitals_count, rng, scenario, large)
    initial_load_rate = rng.beta(2.5, 8.0, size=hospitals_count)
    if scenario in {"mass_casualty", "combined_crisis"}:
        initial_load_rate = np.clip(initial_load_rate + rng.uniform(0.10, 0.28, size=hospitals_count), 0, 0.78)

    return pd.DataFrame(
        {
            "hid": np.arange(hospitals_count),
            "source_hospitalid": sample["hospitalid"].to_numpy(),
            "capacity": capacity,
            "physical_capacity_proxy": np.ceil(capacity / np.clip(1 - initial_load_rate, 0.15, 1.0)).astype(int),
            "initial_load_rate": initial_load_rate,
            "hospital_type": hospital_type,
            "quality": quality,
            "teachingstatus": sample["teachingstatus"].fillna("unknown").to_numpy(),
            "region": sample["region"].fillna("unknown").to_numpy(),
            "numbedscategory": sample["numbedscategory"].fillna("unknown").to_numpy(),
            "x": coords[:, 0],
            "y": coords[:, 1],
        }
    )


def travel_minutes_large(
    patients: pd.DataFrame,
    hospitals: pd.DataFrame,
    rng: np.random.Generator,
    scenario: str,
) -> np.ndarray:
    p_xy = patients[["x", "y"]].to_numpy()
    h_xy = hospitals[["x", "y"]].to_numpy()
    dist = np.sqrt(((p_xy[:, None, :] - h_xy[None, :, :]) ** 2).sum(axis=2))
    base = 5.0 + 1.42 * dist + rng.normal(0, 2.4, size=dist.shape)
    multiplier = {
        "normal": 1.00,
        "mass_casualty": 1.28,
        "specialty_shortage": 1.12,
        "readiness_degradation": 1.05,
        "transport_disruption": 1.55,
        "combined_crisis": 1.48,
    }[scenario]
    disruption_noise = 0.0
    if scenario in {"transport_disruption", "combined_crisis"}:
        disruption_noise = rng.gamma(shape=1.6, scale=5.0, size=dist.shape)
    return np.clip(base * multiplier + disruption_noise, 3, 150)


def optimize_capacitated(utility: np.ndarray, capacities: np.ndarray) -> np.ndarray:
    n = utility.shape[0]
    slots = np.repeat(np.arange(len(capacities)), capacities.astype(int))
    if len(slots) < n:
        raise RuntimeError("No feasible assignment: available capacity is smaller than patient count.")
    if linear_sum_assignment is None:
        from patient_allocation_experiment import optimize_assignment

        return optimize_assignment(utility, capacities)
    cost = -utility[:, slots]
    row_ind, col_ind = linear_sum_assignment(cost)
    assignment = np.full(n, -1, dtype=int)
    assignment[row_ind] = slots[col_ind]
    if (assignment < 0).any():
        raise RuntimeError("Internal optimization error: not all patients assigned.")
    return assignment


def make_instance(
    feature_table: pd.DataFrame,
    hospital_meta: pd.DataFrame,
    scenario: str,
    seed: int,
    base_patients: int,
    hospitals_count: int,
) -> LargeInstance:
    rng = np.random.default_rng(seed)
    n = scenario_patient_count(base_patients, scenario)
    patients = make_patients_from_features(feature_table, n, rng, scenario_patient_mode(scenario))
    hospitals = make_hospital_network(hospital_meta, n, hospitals_count, rng, scenario)
    travel = travel_minutes_large(patients, hospitals, rng, scenario)
    capacities = hospitals["capacity"].to_numpy(int)

    pred_dynamic = survival_matrix(patients, hospitals, travel, predicted=True, dynamic=True)
    pred_static = survival_matrix(patients, hospitals, travel, predicted=True, dynamic=False)
    true_dynamic = survival_matrix(patients, hospitals, travel, predicted=False, dynamic=True)
    teacher_assignment = optimize_capacitated(pred_dynamic, capacities)
    oracle_assignment = optimize_capacitated(true_dynamic, capacities)
    pair_features = build_pair_features(patients, hospitals, travel, pred_dynamic)
    return LargeInstance(
        scenario=scenario,
        seed=seed,
        patients=patients,
        hospitals=hospitals,
        travel=travel,
        pair_features=pair_features,
        teacher_assignment=teacher_assignment,
        oracle_assignment=oracle_assignment,
        true_utility=true_dynamic,
        pred_utility=pred_dynamic,
        pred_static=pred_static,
    )


def train_large_model(
    feature_table: pd.DataFrame,
    hospital_meta: pd.DataFrame,
    train_reps: int,
    train_patients: int,
    train_hospitals: int,
):
    instances = []
    for s_idx, scenario in enumerate(SCENARIOS):
        for rep in range(train_reps):
            instances.append(
                make_instance(
                    feature_table,
                    hospital_meta,
                    scenario,
                    BASE_SEED + 50_000 + 1000 * s_idx + rep,
                    train_patients,
                    train_hospitals,
                )
            )
    x_train, y_train, q_train, mean, std = stack_training(instances)
    model = train_model(x_train, y_train, q_train, seed=BASE_SEED + 17)
    return model, mean, std, len(instances), int(sum(len(inst.patients) for inst in instances))


def gini(values: np.ndarray) -> float:
    arr = np.sort(values.astype(float))
    if arr.sum() <= 0:
        return 0.0
    n = len(arr)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(arr) / (n * arr.sum()))


def evaluate_assignment(inst: LargeInstance, assignment: np.ndarray, policy: str, scores: np.ndarray | None) -> dict[str, float | str | int]:
    n = len(inst.patients)
    chosen = inst.true_utility[np.arange(n), assignment]
    chosen_travel = inst.travel[np.arange(n), assignment]
    loads = np.bincount(assignment, minlength=len(inst.hospitals))
    capacity = inst.hospitals["capacity"].to_numpy(float)
    profile_match = inst.patients["injury_type"].to_numpy() == inst.hospitals["hospital_type"].to_numpy()[assignment]
    severe = (inst.patients["iss_proxy"].to_numpy(float) >= 25) | (inst.patients["true_base_survival"].to_numpy(float) < 0.65)
    deadline_ok = chosen_travel <= inst.patients["deadline_min"].to_numpy(float)
    oracle_mean = inst.true_utility[np.arange(n), inst.oracle_assignment].mean()

    row: dict[str, float | str | int] = {
        "scenario": inst.scenario,
        "seed": inst.seed,
        "policy": policy,
        "patients": n,
        "hospitals": len(inst.hospitals),
        "mean_survival": float(chosen.mean()),
        "severe_mean_survival": float(chosen[severe].mean()) if severe.any() else float("nan"),
        "regret_to_oracle": float(oracle_mean - chosen.mean()),
        "avg_travel_min": float(chosen_travel.mean()),
        "p95_travel_min": float(np.percentile(chosen_travel, 95)),
        "deadline_ok_rate": float(deadline_ok.mean()),
        "profile_match_rate": float(profile_match.mean()),
        "capacity_violations": float(np.maximum(loads - capacity, 0).sum()),
        "max_utilization": float((loads / capacity).max()),
        "load_gini": gini(loads / capacity),
        "teacher_match": float((assignment == inst.teacher_assignment).mean()),
        "oracle_match": float((assignment == inst.oracle_assignment).mean()),
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


def profile_nearest_assignment(inst: LargeInstance, capacities: np.ndarray, order: np.ndarray) -> np.ndarray:
    match = compatibility(inst.patients, inst.hospitals)
    score = 3.0 * match - inst.travel / 120.0 + inst.hospitals["quality"].to_numpy()[None, :] * 0.15
    return greedy_assignment(score, capacities, order)


def evaluate_large(
    feature_table: pd.DataFrame,
    hospital_meta: pd.DataFrame,
    model,
    mean: np.ndarray,
    std: np.ndarray,
    eval_reps: int,
    eval_patients: int,
    eval_hospitals: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    examples = []
    for s_idx, scenario in enumerate(SCENARIOS):
        for rep in range(eval_reps):
            seed = BASE_SEED + 100_000 + 1000 * s_idx + rep
            inst = make_instance(feature_table, hospital_meta, scenario, seed, eval_patients, eval_hospitals)
            capacities = inst.hospitals["capacity"].to_numpy(int)
            severity_order = np.argsort(inst.patients["pred_base_survival"].to_numpy() + 0.004 * inst.patients["deadline_min"].to_numpy())
            scores = predict_scores(model, inst.pair_features, mean, std)
            assignments = {
                "nearest": nearest_assignment(inst.travel, capacities, severity_order),
                "static_optimizer": optimize_capacitated(inst.pred_static, capacities),
                "greedy_dynamic": greedy_assignment(inst.pred_utility, capacities, severity_order),
                "neural_online": assign_from_scores(scores, capacities, severity_order, capacity_aware=True),
                "neural_optimizer": optimize_capacitated(scores, capacities),
                "oracle_true": inst.oracle_assignment,
                "neural_no_capacity": no_capacity_top1(scores),
            }
            for policy in POLICIES:
                rows.append(evaluate_assignment(inst, assignments[policy], policy, scores if policy.startswith("neural") else None))

            if rep == 0:
                probs = probability_vector(scores)
                second_best = np.partition(probs, -2, axis=1)[:, -2]
                for pid in np.argsort(-second_best)[:8]:
                    order = np.argsort(-probs[pid])
                    row = {
                        "scenario": scenario,
                        "patient_id": int(inst.patients.loc[pid, "pid"]),
                        "age": float(inst.patients.loc[pid, "age"]),
                        "injury_type": inst.patients.loc[pid, "injury_type"],
                        "base_survival": float(inst.patients.loc[pid, "pred_base_survival"]),
                        "top3": "; ".join(
                            f"H{int(j)}:{probs[pid, j]:.3f}/{inst.hospitals.loc[j, 'hospital_type']}/cap{int(inst.hospitals.loc[j, 'capacity'])}"
                            for j in order[:3]
                        ),
                    }
                    for j in range(len(inst.hospitals)):
                        row[f"p_H{j}"] = float(probs[pid, j])
                    examples.append(row)
    return pd.DataFrame(rows), pd.DataFrame(examples)


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "mean_survival",
        "severe_mean_survival",
        "regret_to_oracle",
        "avg_travel_min",
        "p95_travel_min",
        "deadline_ok_rate",
        "profile_match_rate",
        "capacity_violations",
        "max_utilization",
        "load_gini",
        "teacher_top3_rate",
        "mean_teacher_prob",
    ]
    grouped = raw.groupby(["scenario", "policy"])[metrics].agg(["mean", "std", "count"]).reset_index()
    grouped.columns = ["_".join(c).strip("_") for c in grouped.columns.to_flat_index()]
    for metric in metrics:
        grouped[f"{metric}_ci95"] = 1.96 * grouped[f"{metric}_std"] / np.sqrt(grouped[f"{metric}_count"].clip(lower=1))
    return grouped


def plot_survival(summary: pd.DataFrame, out: Path) -> None:
    policies = ["nearest", "static_optimizer", "greedy_dynamic", "neural_online", "neural_optimizer", "oracle_true"]
    colors = {
        "nearest": "#9aa0a6",
        "static_optimizer": "#d4a84f",
        "greedy_dynamic": "#5f8db8",
        "neural_online": "#b66d3a",
        "neural_optimizer": "#7a5aa6",
        "oracle_true": "#2f2f2f",
    }
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=True)
    for ax, scenario in zip(axes.ravel(), SCENARIOS):
        sub = summary[summary["scenario"] == scenario].set_index("policy")
        vals = [sub.loc[p, "mean_survival_mean"] for p in policies]
        errs = [sub.loc[p, "mean_survival_ci95"] for p in policies]
        ax.bar(np.arange(len(policies)), vals, yerr=errs, capsize=3, color=[colors[p] for p in policies])
        ax.set_title(SCENARIO_LABELS_RU[scenario].replace("\n", " "))
        ax.set_xticks(np.arange(len(policies)))
        ax.set_xticklabels([POLICY_LABELS_RU[p] for p in policies], rotation=24, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(0, 1.02)
    axes[0, 0].set_ylabel("Mean utility")
    axes[1, 0].set_ylabel("Mean utility")
    fig.suptitle("Patient distribution under capacity constraints")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_regret(summary: pd.DataFrame, out: Path) -> None:
    policies = ["nearest", "static_optimizer", "greedy_dynamic", "neural_online", "neural_optimizer"]
    labels = [POLICY_LABELS_RU[p] for p in policies]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(SCENARIOS))
    width = 0.15
    colors = ["#9aa0a6", "#d4a84f", "#5f8db8", "#b66d3a", "#7a5aa6"]
    for idx, (policy, label, color) in enumerate(zip(policies, labels, colors)):
        sub = summary[summary["policy"] == policy].set_index("scenario").loc[SCENARIOS]
        ax.bar(x + (idx - 2) * width, sub["regret_to_oracle_mean"], width, yerr=sub["regret_to_oracle_ci95"], label=label, color=color, capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS_RU[s] for s in SCENARIOS], rotation=0, ha="center")
    ax.set_ylabel("Gap relative to oracle")
    ax.set_title("Deviation of policies from the oracle assignment")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_capacity_ablation(raw: pd.DataFrame, out: Path) -> None:
    sub = raw[raw["policy"].isin(["neural_optimizer", "neural_no_capacity"])].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    grouped = sub.groupby(["scenario", "policy"])["capacity_violations"].mean().reset_index()
    for ax, policy, color in zip(axes, ["neural_optimizer", "neural_no_capacity"], ["#7a5aa6", "#c94f4f"]):
        vals = grouped[grouped["policy"] == policy].set_index("scenario").loc[SCENARIOS]["capacity_violations"]
        ax.bar(np.arange(len(SCENARIOS)), vals, color=color)
        ax.set_title(POLICY_LABELS_RU[policy])
        ax.set_xticks(np.arange(len(SCENARIOS)))
        ax.set_xticklabels([SCENARIO_LABELS_RU[s] for s in SCENARIOS], rotation=0, ha="center")
        ax.set_ylabel("Mean number of violations")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Capacity ablation: unconstrained neural-network routing is infeasible")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=FEATURES)
    parser.add_argument("--hospital-meta", type=Path, default=HOSPITAL_META)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--train-reps", type=int, default=6)
    parser.add_argument("--eval-reps", type=int, default=8)
    parser.add_argument("--train-patients", type=int, default=260)
    parser.add_argument("--eval-patients", type=int, default=520)
    parser.add_argument("--train-hospitals", type=int, default=14)
    parser.add_argument("--eval-hospitals", type=int, default=28)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_table = pd.read_csv(args.features)
    hospital_meta = pd.read_csv(args.hospital_meta)

    model, mean, std, train_instances, train_patient_count = train_large_model(
        feature_table,
        hospital_meta,
        args.train_reps,
        args.train_patients,
        args.train_hospitals,
    )
    raw, examples = evaluate_large(
        feature_table,
        hospital_meta,
        model,
        mean,
        std,
        args.eval_reps,
        args.eval_patients,
        args.eval_hospitals,
    )
    summary = summarize(raw)

    raw.to_csv(out_dir / "large_scale_raw.csv", index=False)
    summary.to_csv(out_dir / "large_scale_summary.csv", index=False)
    examples.to_csv(out_dir / "large_scale_probability_examples.csv", index=False)
    pd.DataFrame(
        [
            {"parameter": "features", "value": str(args.features)},
            {"parameter": "hospital_meta", "value": str(args.hospital_meta)},
            {"parameter": "scenarios", "value": ", ".join(SCENARIOS)},
            {"parameter": "train_reps_per_scenario", "value": args.train_reps},
            {"parameter": "eval_reps_per_scenario", "value": args.eval_reps},
            {"parameter": "train_instances", "value": train_instances},
            {"parameter": "train_patients_total", "value": train_patient_count},
            {"parameter": "eval_patient_base", "value": args.eval_patients},
            {"parameter": "eval_hospitals", "value": args.eval_hospitals},
            {"parameter": "evaluated_assignments", "value": int(raw["patients"].sum())},
            {"parameter": "seed_base", "value": BASE_SEED},
        ]
    ).to_csv(out_dir / "large_scale_protocol.csv", index=False)

    plot_survival(summary, out_dir / "large_scale_survival_comparison.png")
    plot_regret(summary, out_dir / "large_scale_regret_to_oracle.png")
    plot_capacity_ablation(raw, out_dir / "large_scale_capacity_ablation.png")

    display = summary[
        summary["policy"].isin(["nearest", "greedy_dynamic", "neural_online", "neural_optimizer", "oracle_true"])
    ][["scenario", "policy", "mean_survival_mean", "regret_to_oracle_mean", "capacity_violations_mean", "profile_match_rate_mean"]]
    print(display.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved outputs to {out_dir}")


if __name__ == "__main__":
    main()
