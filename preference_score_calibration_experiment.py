"""
Preference-score calibration experiment for patient-to-hospital assignment.

This script implements the method without a neural target ambiguity:

1. Build pairwise preference score s_ij from interpretable components.
2. Calibrate component weights against system-level metrics.
3. Solve a capacitated assignment problem using s_ij.
4. Compare against baseline routing strategies.

Run example:
  python preference_score_calibration_experiment.py --calibration all --train-reps 2 --test-reps 4
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from patient_allocation_experiment import (
    capacity_aware_random,
    compatibility,
    make_hospitals,
    make_patients_from_features,
    nearest_assignment,
    optimize_assignment,
    survival_matrix,
    travel_minutes,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_FEATURES = ROOT / "results_external" / "eicu_demo_features.csv"
FALLBACK_FEATURES = ROOT / "results" / "mimic_dynamic_risk_features.csv"
OUT = ROOT / "results_preference_score"
BASE_SEED = 20260628
SCENARIOS = ("normal", "mass_casualty", "specialty_shortage")
POLICIES = (
    "random",
    "nearest",
    "max_survival",
    "min_load",
    "preference_optimizer",
    "oracle_true",
)
WEIGHT_NAMES = ("survival", "profile", "transport", "capacity", "load")


@dataclass
class AssignmentInstance:
    scenario: str
    seed: int
    patients: pd.DataFrame
    hospitals: pd.DataFrame
    travel: np.ndarray
    distance: np.ndarray
    free_capacity: np.ndarray
    physical_capacity: np.ndarray
    initial_load: np.ndarray
    pred_survival: np.ndarray
    true_survival: np.ndarray


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def safe_minmax(x: np.ndarray, axis: int | None = None) -> np.ndarray:
    lo = np.nanmin(x, axis=axis, keepdims=True)
    hi = np.nanmax(x, axis=axis, keepdims=True)
    return (x - lo) / np.maximum(hi - lo, 1e-9)


def gini(values: np.ndarray) -> float:
    arr = np.sort(values.astype(float))
    if arr.size == 0 or arr.sum() <= 0:
        return 0.0
    n = arr.size
    return float((2 * np.arange(1, n + 1) - n - 1).dot(arr) / (n * arr.sum()))


def scenario_patient_count(base: int, scenario: str) -> int:
    return {
        "normal": base,
        "mass_casualty": int(round(base * 1.35)),
        "specialty_shortage": base,
    }[scenario]


def pair_distance(patients: pd.DataFrame, hospitals: pd.DataFrame) -> np.ndarray:
    p_xy = patients[["x", "y"]].to_numpy(float)
    h_xy = hospitals[["x", "y"]].to_numpy(float)
    return np.sqrt(((p_xy[:, None, :] - h_xy[None, :, :]) ** 2).sum(axis=2))


def add_load_state(hospitals: pd.DataFrame, rng: np.random.Generator, scenario: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    free_capacity = hospitals["capacity"].to_numpy(int)
    load_rate = rng.beta(2.2, 7.0, size=len(hospitals))
    if scenario == "mass_casualty":
        load_rate = np.clip(load_rate + rng.uniform(0.10, 0.24, size=len(hospitals)), 0, 0.82)
    if scenario == "specialty_shortage":
        load_rate = np.clip(load_rate + rng.uniform(0.03, 0.16, size=len(hospitals)), 0, 0.72)
    physical_capacity = np.maximum(free_capacity + 1, np.ceil(free_capacity / np.clip(1 - load_rate, 0.12, 1.0)).astype(int))
    initial_load = np.maximum(0, physical_capacity - free_capacity)
    return free_capacity, physical_capacity, initial_load


def make_instance(
    feature_table: pd.DataFrame,
    scenario: str,
    seed: int,
    patients_base: int,
    hospitals_count: int,
) -> AssignmentInstance:
    rng = np.random.default_rng(seed)
    n = scenario_patient_count(patients_base, scenario)
    patients = make_patients_from_features(feature_table, n, rng, scenario)
    hospitals = make_hospitals(n, hospitals_count, rng, scenario)
    travel = travel_minutes(patients, hospitals, rng, scenario)
    distance = pair_distance(patients, hospitals)
    free_capacity, physical_capacity, initial_load = add_load_state(hospitals, rng, scenario)
    hospitals = hospitals.copy()
    hospitals["free_capacity"] = free_capacity
    hospitals["physical_capacity"] = physical_capacity
    hospitals["initial_load"] = initial_load
    hospitals["initial_load_rate"] = initial_load / np.maximum(physical_capacity, 1)
    pred_survival = survival_matrix(patients, hospitals, travel, predicted=True, dynamic=True)
    true_survival = survival_matrix(patients, hospitals, travel, predicted=False, dynamic=True)
    return AssignmentInstance(
        scenario=scenario,
        seed=seed,
        patients=patients,
        hospitals=hospitals,
        travel=travel,
        distance=distance,
        free_capacity=free_capacity,
        physical_capacity=physical_capacity,
        initial_load=initial_load,
        pred_survival=pred_survival,
        true_survival=true_survival,
    )


def survival_score(inst: AssignmentInstance) -> np.ndarray:
    return inst.pred_survival


def profile_score(inst: AssignmentInstance) -> np.ndarray:
    return compatibility(inst.patients, inst.hospitals)


def transport_score(inst: AssignmentInstance) -> np.ndarray:
    return np.exp(-inst.travel / 45.0)


def capacity_score(inst: AssignmentInstance) -> np.ndarray:
    free = inst.free_capacity.astype(float)
    score = np.log1p(free) / math.log1p(max(float(free.max()), 1.0))
    return np.repeat(score[None, :], len(inst.patients), axis=0)


def load_penalty(inst: AssignmentInstance) -> np.ndarray:
    load = inst.initial_load / np.maximum(inst.physical_capacity, 1)
    return np.repeat(load[None, :], len(inst.patients), axis=0)


def score_components(inst: AssignmentInstance) -> dict[str, np.ndarray]:
    return {
        "survival": survival_score(inst),
        "profile": profile_score(inst),
        "transport": transport_score(inst),
        "capacity": capacity_score(inst),
        "load": load_penalty(inst),
    }


def normalize_weights(weights: np.ndarray | dict[str, float]) -> dict[str, float]:
    if isinstance(weights, dict):
        arr = np.array([weights[name] for name in WEIGHT_NAMES], dtype=float)
    else:
        arr = np.array(weights, dtype=float)
    arr = np.clip(arr, 0, None)
    if arr.sum() <= 0:
        arr = np.ones(len(WEIGHT_NAMES), dtype=float)
    arr = arr / arr.sum()
    return {name: float(value) for name, value in zip(WEIGHT_NAMES, arr)}


def build_preference_score(inst: AssignmentInstance, weights: np.ndarray | dict[str, float]) -> np.ndarray:
    w = normalize_weights(weights)
    c = score_components(inst)
    score = (
        w["survival"] * c["survival"]
        + w["profile"] * c["profile"]
        + w["transport"] * c["transport"]
        + w["capacity"] * c["capacity"]
        - w["load"] * c["load"]
    )
    return score.astype(float)


def assignment_by_min_load(inst: AssignmentInstance, order: np.ndarray) -> np.ndarray:
    remaining = inst.free_capacity.astype(int).copy()
    current_load = inst.initial_load.astype(float).copy()
    physical = inst.physical_capacity.astype(float)
    assignment = np.full(len(inst.patients), -1, dtype=int)
    for i in order:
        load_ratio = current_load / np.maximum(physical, 1)
        candidate_score = -load_ratio - 0.002 * inst.travel[i]
        candidate_score[remaining <= 0] = -1e9
        j = int(np.argmax(candidate_score))
        assignment[i] = j
        remaining[j] -= 1
        current_load[j] += 1
    return assignment


def severity_order(inst: AssignmentInstance) -> np.ndarray:
    severity = (
        -inst.patients["pred_base_survival"].to_numpy(float)
        + 0.012 * inst.patients["iss_proxy"].to_numpy(float)
        + 0.2 * inst.patients["shock_index"].to_numpy(float)
    )
    return np.argsort(-severity)


def solve_preference_assignment(inst: AssignmentInstance, weights: np.ndarray | dict[str, float]) -> np.ndarray:
    score = build_preference_score(inst, weights)
    return optimize_assignment(score, inst.free_capacity)


def evaluate_assignment(
    inst: AssignmentInstance,
    assignment: np.ndarray,
    policy: str,
    weights: dict[str, float] | None = None,
) -> dict[str, float | int | str]:
    n = len(inst.patients)
    idx = np.arange(n)
    loads = np.bincount(assignment, minlength=len(inst.hospitals))
    total_after = inst.initial_load + loads
    utilization = total_after / np.maximum(inst.physical_capacity, 1)
    p_type = inst.patients["injury_type"].to_numpy()
    h_type = inst.hospitals["hospital_type"].to_numpy()[assignment]
    pred = inst.pred_survival[idx, assignment]
    true = inst.true_survival[idx, assignment]
    row: dict[str, float | int | str] = {
        "scenario": inst.scenario,
        "seed": inst.seed,
        "policy": policy,
        "mean_pred_survival": float(pred.mean()),
        "mean_true_survival": float(true.mean()),
        "avg_travel_min": float(inst.travel[idx, assignment].mean()),
        "avg_distance": float(inst.distance[idx, assignment].mean()),
        "profile_match_rate": float((p_type == h_type).mean()),
        "profile_center_count": int((p_type == h_type).sum()),
        "capacity_violations": float(np.maximum(loads - inst.free_capacity, 0).sum()),
        "overloaded_hospitals": int((loads > inst.free_capacity).sum()),
        "mean_utilization": float(utilization.mean()),
        "max_utilization": float(utilization.max()),
        "load_gini": gini(utilization),
    }
    if weights:
        for name, value in weights.items():
            row[f"w_{name}"] = float(value)
    return row


def system_objective(metrics: dict[str, float | int | str]) -> float:
    return float(
        metrics["mean_true_survival"]
        + 0.045 * metrics["profile_match_rate"]
        - 0.0012 * metrics["avg_travel_min"]
        - 0.035 * metrics["load_gini"]
        - 0.20 * metrics["capacity_violations"]
    )


def evaluate_weights(instances: list[AssignmentInstance], weights: np.ndarray | dict[str, float]) -> dict[str, float]:
    rows = []
    norm = normalize_weights(weights)
    for inst in instances:
        assignment = solve_preference_assignment(inst, norm)
        row = evaluate_assignment(inst, assignment, "preference_optimizer", norm)
        row["objective"] = system_objective(row)
        rows.append(row)
    df = pd.DataFrame(rows)
    result = {name: norm[name] for name in WEIGHT_NAMES}
    for col in [
        "objective",
        "mean_true_survival",
        "mean_pred_survival",
        "avg_travel_min",
        "profile_match_rate",
        "max_utilization",
        "load_gini",
    ]:
        result[col] = float(df[col].mean())
    return result


def grid_search(instances: list[AssignmentInstance], levels: int) -> pd.DataFrame:
    values = np.linspace(0.0, 1.0, levels)
    rows = []
    for raw in itertools.product(values, repeat=len(WEIGHT_NAMES)):
        if sum(raw) <= 0:
            continue
        row = evaluate_weights(instances, np.array(raw, dtype=float))
        row["calibration_method"] = "grid"
        rows.append(row)
    return pd.DataFrame(rows)


def random_search(instances: list[AssignmentInstance], iterations: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for _ in range(iterations):
        raw = rng.gamma(shape=1.2, scale=1.0, size=len(WEIGHT_NAMES))
        row = evaluate_weights(instances, raw)
        row["calibration_method"] = "random"
        rows.append(row)
    return pd.DataFrame(rows)


def bayesian_optimization(instances: list[AssignmentInstance], iterations: int, rng: np.random.Generator) -> pd.DataFrame:
    # Lightweight Gaussian-kernel surrogate with expected-improvement sampling.
    seed_rows = random_search(instances, max(8, len(WEIGHT_NAMES) * 2), rng)
    rows = seed_rows.to_dict("records")
    for _ in range(iterations):
        x = np.array([[r[name] for name in WEIGHT_NAMES] for r in rows], dtype=float)
        y = np.array([r["objective"] for r in rows], dtype=float)
        candidates = rng.dirichlet(np.ones(len(WEIGHT_NAMES)), size=256)
        dist2 = ((candidates[:, None, :] - x[None, :, :]) ** 2).sum(axis=2)
        kernel = np.exp(-dist2 / (2 * 0.18**2))
        weights = kernel / np.maximum(kernel.sum(axis=1, keepdims=True), 1e-12)
        mu = weights @ y
        local_var = (weights * ((y[None, :] - mu[:, None]) ** 2)).sum(axis=1)
        exploration = 0.08 * np.sqrt(np.maximum(local_var, 1e-12))
        pick = int(np.argmax(mu + exploration))
        row = evaluate_weights(instances, candidates[pick])
        row["calibration_method"] = "bayesian_kernel"
        rows.append(row)
    return pd.DataFrame(rows)


def cma_es_search(instances: list[AssignmentInstance], iterations: int, population: int, rng: np.random.Generator) -> pd.DataFrame:
    mean = np.zeros(len(WEIGHT_NAMES), dtype=float)
    sigma = 1.0
    rows = []
    for _ in range(iterations):
        raw_pop = rng.normal(mean, sigma, size=(population, len(WEIGHT_NAMES)))
        candidates = np.exp(raw_pop)
        scored = []
        for cand in candidates:
            row = evaluate_weights(instances, cand)
            row["calibration_method"] = "cma_es"
            scored.append(row)
        scored.sort(key=lambda r: r["objective"], reverse=True)
        rows.extend(scored)
        elite = np.array([[r[name] for name in WEIGHT_NAMES] for r in scored[: max(2, population // 4)]])
        mean = np.log(np.maximum(elite, 1e-9)).mean(axis=0)
        sigma = max(0.15, sigma * 0.82)
    return pd.DataFrame(rows)


def dominates(a: dict[str, float], b: dict[str, float]) -> bool:
    better_or_equal = (
        a["mean_true_survival"] >= b["mean_true_survival"]
        and a["profile_match_rate"] >= b["profile_match_rate"]
        and a["avg_travel_min"] <= b["avg_travel_min"]
        and a["load_gini"] <= b["load_gini"]
    )
    strictly_better = (
        a["mean_true_survival"] > b["mean_true_survival"]
        or a["profile_match_rate"] > b["profile_match_rate"]
        or a["avg_travel_min"] < b["avg_travel_min"]
        or a["load_gini"] < b["load_gini"]
    )
    return bool(better_or_equal and strictly_better)


def nsga2_search(instances: list[AssignmentInstance], generations: int, population: int, rng: np.random.Generator) -> pd.DataFrame:
    pop = rng.dirichlet(np.ones(len(WEIGHT_NAMES)), size=population)
    rows = []
    for _ in range(generations):
        candidates = []
        for weights in pop:
            mutation = rng.normal(0, 0.08, size=len(WEIGHT_NAMES))
            candidates.append(np.clip(weights + mutation, 1e-6, None))
        candidates = np.vstack([pop, np.asarray(candidates)])
        scored = []
        for cand in candidates:
            row = evaluate_weights(instances, cand)
            row["calibration_method"] = "nsga2"
            scored.append(row)
        rows.extend(scored)
        ranks = []
        for i, a in enumerate(scored):
            rank = sum(dominates(b, a) for j, b in enumerate(scored) if i != j)
            ranks.append(rank)
        order = np.lexsort((-np.array([r["objective"] for r in scored]), np.array(ranks)))
        pop = np.array([[scored[i][name] for name in WEIGHT_NAMES] for i in order[:population]])
    return pd.DataFrame(rows)


def calibrate_weights(
    instances: list[AssignmentInstance],
    method: str,
    rng: np.random.Generator,
    grid_levels: int,
    random_iter: int,
    bo_iter: int,
    cma_iter: int,
    nsga_gen: int,
    population: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    frames = []
    if method in {"grid", "all"}:
        frames.append(grid_search(instances, grid_levels))
    if method in {"random", "all"}:
        frames.append(random_search(instances, random_iter, rng))
    if method in {"bayesian", "all"}:
        frames.append(bayesian_optimization(instances, bo_iter, rng))
    if method in {"cma_es", "all"}:
        frames.append(cma_es_search(instances, cma_iter, population, rng))
    if method in {"nsga2", "all"}:
        frames.append(nsga2_search(instances, nsga_gen, population, rng))
    results = pd.concat(frames, ignore_index=True)
    best_row = results.sort_values("objective", ascending=False).iloc[0]
    best = {name: float(best_row[name]) for name in WEIGHT_NAMES}
    return best, results


def evaluate_policies(instances: list[AssignmentInstance], weights: dict[str, float], rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    examples = []
    for inst in instances:
        order = severity_order(inst)
        score = build_preference_score(inst, weights)
        assignments = {
            "random": capacity_aware_random(len(inst.patients), inst.free_capacity, rng),
            "nearest": nearest_assignment(inst.travel, inst.free_capacity, np.arange(len(inst.patients))),
            "max_survival": optimize_assignment(inst.pred_survival, inst.free_capacity),
            "min_load": assignment_by_min_load(inst, order),
            "preference_optimizer": optimize_assignment(score, inst.free_capacity),
            "oracle_true": optimize_assignment(inst.true_survival, inst.free_capacity),
        }
        for policy in POLICIES:
            rows.append(evaluate_assignment(inst, assignments[policy], policy, weights if policy == "preference_optimizer" else None))

        probs = np.exp(score - score.max(axis=1, keepdims=True))
        probs = probs / probs.sum(axis=1, keepdims=True)
        for pid in np.argsort(-inst.patients["iss_proxy"].to_numpy(float))[:5]:
            top = np.argsort(-probs[pid])[:5]
            examples.append(
                {
                    "scenario": inst.scenario,
                    "seed": inst.seed,
                    "patient_id": int(inst.patients.loc[pid, "pid"]),
                    "injury_type": str(inst.patients.loc[pid, "injury_type"]),
                    "iss_proxy": float(inst.patients.loc[pid, "iss_proxy"]),
                    "top_hospitals": "; ".join(
                        f"H{int(j)} p={probs[pid, j]:.3f} s={score[pid, j]:.3f} type={inst.hospitals.loc[j, 'hospital_type']} free={int(inst.free_capacity[j])}"
                        for j in top
                    ),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(examples)


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "mean_pred_survival",
        "mean_true_survival",
        "avg_travel_min",
        "avg_distance",
        "profile_match_rate",
        "profile_center_count",
        "capacity_violations",
        "overloaded_hospitals",
        "mean_utilization",
        "max_utilization",
        "load_gini",
    ]
    summary = raw.groupby(["scenario", "policy"])[metrics].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.to_flat_index()]
    return summary


def build_instances(feature_table: pd.DataFrame, reps: int, patients: int, hospitals: int, seed_offset: int) -> list[AssignmentInstance]:
    instances = []
    for s_idx, scenario in enumerate(SCENARIOS):
        for rep in range(reps):
            seed = BASE_SEED + seed_offset + 1000 * s_idx + rep
            instances.append(make_instance(feature_table, scenario, seed, patients, hospitals))
    return instances


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES if DEFAULT_FEATURES.exists() else FALLBACK_FEATURES)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--calibration", choices=["grid", "random", "bayesian", "cma_es", "nsga2", "all"], default="all")
    parser.add_argument("--train-reps", type=int, default=2)
    parser.add_argument("--test-reps", type=int, default=4)
    parser.add_argument("--patients", type=int, default=120)
    parser.add_argument("--hospitals", type=int, default=8)
    parser.add_argument("--grid-levels", type=int, default=3)
    parser.add_argument("--random-iter", type=int, default=40)
    parser.add_argument("--bo-iter", type=int, default=18)
    parser.add_argument("--cma-iter", type=int, default=8)
    parser.add_argument("--nsga-gen", type=int, default=6)
    parser.add_argument("--population", type=int, default=18)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(BASE_SEED)
    feature_table = pd.read_csv(args.features)

    train_instances = build_instances(feature_table, args.train_reps, args.patients, args.hospitals, seed_offset=10_000)
    test_instances = build_instances(feature_table, args.test_reps, args.patients, args.hospitals, seed_offset=50_000)

    best_weights, calibration = calibrate_weights(
        train_instances,
        args.calibration,
        rng,
        args.grid_levels,
        args.random_iter,
        args.bo_iter,
        args.cma_iter,
        args.nsga_gen,
        args.population,
    )

    raw, examples = evaluate_policies(test_instances, best_weights, rng)
    summary = summarize(raw)

    calibration.to_csv(args.out / "preference_weight_calibration.csv", index=False)
    raw.to_csv(args.out / "preference_assignment_raw.csv", index=False)
    summary.to_csv(args.out / "preference_assignment_summary.csv", index=False)
    examples.to_csv(args.out / "preference_patient_rankings.csv", index=False)
    with open(args.out / "best_preference_weights.json", "w", encoding="utf-8") as f:
        json.dump(best_weights, f, ensure_ascii=False, indent=2)
    pd.DataFrame(
        [
            {
                "features": str(args.features),
                "calibration": args.calibration,
                "train_reps": args.train_reps,
                "test_reps": args.test_reps,
                "patients": args.patients,
                "hospitals": args.hospitals,
                "base_seed": BASE_SEED,
            }
        ]
    ).to_csv(args.out / "preference_experiment_protocol.csv", index=False)

    print("Best weights:", json.dumps(best_weights, ensure_ascii=False))
    print("Wrote:", args.out)
    print(summary[summary["policy"] == "preference_optimizer"].to_string(index=False))


if __name__ == "__main__":
    main()
