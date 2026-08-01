"""
Reproducible patient-to-hospital allocation experiment.

The script is intentionally lightweight: it needs only numpy and pandas, both
available in the bundled Codex runtime. It models the second half of the paper:
given patient risk estimates and a hospital network, compare allocation policies.

Run:
  python patient_allocation_experiment.py --replicates 30 --out results
"""

from __future__ import annotations

import argparse
import heapq
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


INJURY_TYPES = np.array(["trauma", "burn", "general"])
POLICIES = [
    "random",
    "nearest",
    "severity_first",
    "greedy_dynamic",
    "static_optimized",
    "dynamic_optimized",
    "oracle_optimized",
]


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class Edge:
    to: int
    rev: int
    cap: int
    cost: int


class MinCostFlow:
    def __init__(self, n: int) -> None:
        self.graph: list[list[Edge]] = [[] for _ in range(n)]

    def add_edge(self, src: int, dst: int, cap: int, cost: int) -> None:
        fwd = Edge(dst, len(self.graph[dst]), cap, cost)
        rev = Edge(src, len(self.graph[src]), 0, -cost)
        self.graph[src].append(fwd)
        self.graph[dst].append(rev)

    def solve(self, source: int, sink: int, flow_required: int) -> int:
        total_cost = 0
        flow = 0
        n = len(self.graph)

        while flow < flow_required:
            dist = [math.inf] * n
            prev_node = [-1] * n
            prev_edge = [-1] * n
            in_queue = [False] * n

            dist[source] = 0
            queue = [source]
            in_queue[source] = True

            while queue:
                node = queue.pop(0)
                in_queue[node] = False
                for edge_idx, edge in enumerate(self.graph[node]):
                    if edge.cap <= 0:
                        continue
                    nd = dist[node] + edge.cost
                    if nd < dist[edge.to]:
                        dist[edge.to] = nd
                        prev_node[edge.to] = node
                        prev_edge[edge.to] = edge_idx
                        if not in_queue[edge.to]:
                            queue.append(edge.to)
                            in_queue[edge.to] = True

            if prev_node[sink] == -1:
                raise RuntimeError("No feasible assignment: total hospital capacity is too small.")

            add = flow_required - flow
            node = sink
            while node != source:
                edge = self.graph[prev_node[node]][prev_edge[node]]
                add = min(add, edge.cap)
                node = prev_node[node]

            node = sink
            while node != source:
                edge = self.graph[prev_node[node]][prev_edge[node]]
                edge.cap -= add
                self.graph[node][edge.rev].cap += add
                node = prev_node[node]

            flow += add
            total_cost += add * dist[sink]

        return total_cost


def make_patients(n: int, rng: np.random.Generator, scenario: str) -> pd.DataFrame:
    if scenario == "specialty_shortage":
        type_probs = [0.52, 0.28, 0.20]
    else:
        type_probs = [0.45, 0.15, 0.40]

    injury_type = rng.choice(INJURY_TYPES, size=n, p=type_probs)
    age = rng.integers(18, 91, size=n)
    iss = np.clip(rng.gamma(shape=2.3, scale=7.0, size=n), 1, 50)

    if scenario == "mass_casualty":
        iss = np.clip(iss * rng.uniform(1.05, 1.35, size=n), 1, 55)

    shock_index = np.clip(rng.lognormal(mean=-0.20, sigma=0.35, size=n), 0.35, 2.3)
    vital_score = np.clip(
        1.0
        / (1.0 + np.exp(-(
            1.10
            - 0.060 * iss
            - 0.85 * np.maximum(shock_index - 0.9, 0)
            - 0.012 * np.maximum(age - 55, 0)
            + rng.normal(0, 0.25, size=n)
        ))),
        0.02,
        0.99,
    )

    # Synthetic personalized risk signal. In a real deployment, these terms
    # come from the learned temporal risk module.
    base_logit = -1.2 + 3.10 * vital_score - 0.040 * iss - 0.020 * np.maximum(age - 55, 0) - 0.35 * shock_index
    true_base = np.clip(sigmoid(base_logit), 0.02, 0.995)
    true_hazard = np.clip(0.030 + 0.010 * iss + 0.16 * np.maximum(shock_index - 0.75, 0) + 0.10 * (1 - vital_score), 0.02, 0.95)

    pred_base = np.clip(true_base + rng.normal(0, 0.035, size=n), 0.01, 0.995)
    pred_hazard = np.clip(true_hazard * np.exp(rng.normal(0, 0.12, size=n)), 0.015, 1.20)

    threshold = 0.70
    deadline = np.where(
        true_base > threshold,
        -np.log(threshold / true_base) / true_hazard * 60.0,
        0.0,
    )

    patient_xy = rng.normal(loc=[25, 25], scale=[6, 6], size=(n, 2))

    return pd.DataFrame(
        {
            "pid": np.arange(n),
            "age": age,
            "iss_proxy": iss,
            "shock_index": shock_index,
            "vital_score": vital_score,
            "injury_type": injury_type,
            "true_base_survival": true_base,
            "true_hazard": true_hazard,
            "pred_base_survival": pred_base,
            "pred_hazard": pred_hazard,
            "deadline_min": np.clip(deadline, 0, 360),
            "x": patient_xy[:, 0],
            "y": patient_xy[:, 1],
        }
    )


def make_patients_from_features(
    feature_table: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
    scenario: str,
) -> pd.DataFrame:
    base = (
        feature_table.sort_values("time_from_icu_h")
        .groupby("stay_id")
        .last()
        .reset_index()
    )
    if "vital_score" not in base:
        if "RTS" in base:
            base["vital_score"] = (base["RTS"] / 7.84).clip(0, 1)
        else:
            base["vital_score"] = 0.5
    base = base.dropna(subset=["anchor_age", "vital_score", "shock_index", "survival_t0", "hazard_proxy"])
    replace = len(base) < n
    sample = base.sample(n=n, replace=replace, random_state=int(rng.integers(0, 2**32 - 1))).reset_index(drop=True)

    age = sample["anchor_age"].to_numpy()
    iss = np.clip(sample.get("iss_proxy", pd.Series(np.ones(n))).to_numpy(), 1, 50)
    shock_index = np.clip(sample["shock_index"].to_numpy() * np.exp(rng.normal(0, 0.05, size=n)), 0.35, 2.3)
    vital_score = np.clip(sample["vital_score"].to_numpy() + rng.normal(0, 0.015, size=n), 0, 1)
    true_base = np.clip(sample["survival_t0"].to_numpy() + rng.normal(0, 0.015, size=n), 0.02, 0.995)
    true_hazard = np.clip(sample["hazard_proxy"].to_numpy() * np.exp(rng.normal(0, 0.08, size=n)), 0.015, 1.0)

    injury_count = sample.get("injury_dx_count", pd.Series(np.zeros(n))).to_numpy()
    injury_type = np.where(injury_count > 0, "trauma", "general").astype(object)
    non_trauma = injury_count <= 0
    burn_probability = 0.12 if scenario != "specialty_shortage" else 0.22
    injury_type[non_trauma] = rng.choice(["general", "burn"], size=non_trauma.sum(), p=[1 - burn_probability, burn_probability])

    pred_base = np.clip(true_base + rng.normal(0, 0.035, size=n), 0.01, 0.995)
    pred_hazard = np.clip(true_hazard * np.exp(rng.normal(0, 0.12, size=n)), 0.015, 1.20)

    threshold = 0.70
    deadline = np.where(
        true_base > threshold,
        -np.log(threshold / true_base) / true_hazard * 60.0,
        0.0,
    )
    patient_xy = rng.normal(loc=[25, 25], scale=[6, 6], size=(n, 2))

    return pd.DataFrame(
        {
            "pid": np.arange(n),
            "age": age,
            "iss_proxy": iss,
            "shock_index": shock_index,
            "vital_score": vital_score,
            "injury_type": injury_type,
            "true_base_survival": true_base,
            "true_hazard": true_hazard,
            "pred_base_survival": pred_base,
            "pred_hazard": pred_hazard,
            "deadline_min": np.clip(deadline, 0, 360),
            "x": patient_xy[:, 0],
            "y": patient_xy[:, 1],
        }
    )


def make_hospitals(n: int, h: int, rng: np.random.Generator, scenario: str) -> pd.DataFrame:
    capacity_factor = {
        "normal": 1.22,
        "mass_casualty": 1.04,
        "specialty_shortage": 1.08,
    }[scenario]

    raw = rng.dirichlet(np.ones(h) * 1.4)
    total_capacity = math.ceil(n * capacity_factor)
    capacity = np.maximum(5, np.floor(raw * total_capacity).astype(int))
    while capacity.sum() < n:
        capacity[rng.integers(0, h)] += 1

    hospital_type = rng.choice(INJURY_TYPES, size=h, p=[0.35, 0.18, 0.47])
    if scenario == "specialty_shortage":
        hospital_type = rng.choice(INJURY_TYPES, size=h, p=[0.22, 0.08, 0.70])
    hospital_type[:3] = INJURY_TYPES
    rng.shuffle(hospital_type)

    angle = np.linspace(0, 2 * np.pi, h, endpoint=False)
    radius = rng.uniform(17, 32, size=h)
    hospital_xy = np.column_stack([25 + radius * np.cos(angle), 25 + radius * np.sin(angle)])

    return pd.DataFrame(
        {
            "hid": np.arange(h),
            "capacity": capacity,
            "hospital_type": hospital_type,
            "quality": rng.uniform(0.88, 1.14, size=h),
            "x": hospital_xy[:, 0],
            "y": hospital_xy[:, 1],
        }
    )


def travel_minutes(patients: pd.DataFrame, hospitals: pd.DataFrame, rng: np.random.Generator, scenario: str) -> np.ndarray:
    p_xy = patients[["x", "y"]].to_numpy()
    h_xy = hospitals[["x", "y"]].to_numpy()
    dist = np.sqrt(((p_xy[:, None, :] - h_xy[None, :, :]) ** 2).sum(axis=2))
    base = 4.0 + dist * 1.55 + rng.normal(0, 2.0, size=dist.shape)
    multiplier = 1.0
    if scenario == "mass_casualty":
        multiplier = 1.35
    if scenario == "specialty_shortage":
        multiplier = 1.12
    return np.clip(base * multiplier, 3, 120)


def compatibility(patients: pd.DataFrame, hospitals: pd.DataFrame) -> np.ndarray:
    p_type = patients["injury_type"].to_numpy()
    h_type = hospitals["hospital_type"].to_numpy()
    match = np.where(p_type[:, None] == h_type[None, :], 1.00, 0.82)
    severe = patients["iss_proxy"].to_numpy() >= 25
    match[severe, :] *= np.where(p_type[severe, None] == h_type[None, :], 1.00, 0.82)
    return match


def survival_matrix(
    patients: pd.DataFrame,
    hospitals: pd.DataFrame,
    travel: np.ndarray,
    predicted: bool,
    dynamic: bool,
) -> np.ndarray:
    base_col = "pred_base_survival" if predicted else "true_base_survival"
    hazard_col = "pred_hazard" if predicted else "true_hazard"
    base = patients[base_col].to_numpy()[:, None]
    hazard = patients[hazard_col].to_numpy()[:, None]
    quality = hospitals["quality"].to_numpy()[None, :]
    match = compatibility(patients, hospitals)
    if dynamic:
        survival = base * np.exp(-hazard * (travel / 60.0))
    else:
        survival = base
    return np.clip(survival * quality * match, 0, 1)


def optimize_assignment(utility: np.ndarray, capacities: np.ndarray) -> np.ndarray:
    n, h = utility.shape
    source = 0
    patient_offset = 1
    hospital_offset = patient_offset + n
    sink = hospital_offset + h
    graph = MinCostFlow(sink + 1)

    for i in range(n):
        graph.add_edge(source, patient_offset + i, 1, 0)
    for i in range(n):
        for j in range(h):
            graph.add_edge(patient_offset + i, hospital_offset + j, 1, int(round(-utility[i, j] * 1_000_000)))
    for j, cap in enumerate(capacities):
        graph.add_edge(hospital_offset + j, sink, int(cap), 0)

    graph.solve(source, sink, n)

    assignment = np.full(n, -1, dtype=int)
    for i in range(n):
        node = patient_offset + i
        for edge in graph.graph[node]:
            if hospital_offset <= edge.to < hospital_offset + h and edge.cap == 0:
                assignment[i] = edge.to - hospital_offset
                break

    if (assignment < 0).any():
        raise RuntimeError("Internal optimization error: not all patients assigned.")
    return assignment


def capacity_aware_random(n: int, capacities: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    slots = np.repeat(np.arange(len(capacities)), capacities)
    rng.shuffle(slots)
    return slots[:n].astype(int)


def nearest_assignment(travel: np.ndarray, capacities: np.ndarray, order: np.ndarray) -> np.ndarray:
    remaining = capacities.astype(int).copy()
    assignment = np.full(travel.shape[0], -1, dtype=int)
    for i in order:
        for j in np.argsort(travel[i]):
            if remaining[j] > 0:
                assignment[i] = j
                remaining[j] -= 1
                break
    return assignment


def greedy_assignment(utility: np.ndarray, capacities: np.ndarray, order: np.ndarray) -> np.ndarray:
    remaining = capacities.astype(int).copy()
    assignment = np.full(utility.shape[0], -1, dtype=int)
    for i in order:
        for j in np.argsort(-utility[i]):
            if remaining[j] > 0:
                assignment[i] = j
                remaining[j] -= 1
                break
    return assignment


def evaluate(
    patients: pd.DataFrame,
    hospitals: pd.DataFrame,
    travel: np.ndarray,
    assignment: np.ndarray,
    true_utility: np.ndarray,
) -> dict[str, float]:
    n = len(patients)
    chosen_utility = true_utility[np.arange(n), assignment]
    chosen_travel = travel[np.arange(n), assignment]
    severe = (patients["iss_proxy"].to_numpy() >= 25) | (patients["true_base_survival"].to_numpy() < 0.65)
    deadline_ok = chosen_travel <= patients["deadline_min"].to_numpy()
    loads = np.bincount(assignment, minlength=len(hospitals))
    cap = hospitals["capacity"].to_numpy()

    return {
        "mean_survival": float(chosen_utility.mean()),
        "severe_mean_survival": float(chosen_utility[severe].mean()) if severe.any() else float("nan"),
        "deadline_ok_rate": float(deadline_ok.mean()),
        "avg_travel_min": float(chosen_travel.mean()),
        "p95_travel_min": float(np.percentile(chosen_travel, 95)),
        "max_utilization": float((loads / cap).max()),
        "capacity_violations": float(np.maximum(loads - cap, 0).sum()),
    }


def run_one(
    seed: int,
    scenario: str,
    n: int,
    h: int,
    feature_table: pd.DataFrame | None = None,
) -> list[dict[str, float | int | str]]:
    rng = np.random.default_rng(seed)
    if feature_table is None:
        patients = make_patients(n, rng, scenario)
    else:
        patients = make_patients_from_features(feature_table, n, rng, scenario)
    hospitals = make_hospitals(n, h, rng, scenario)
    travel = travel_minutes(patients, hospitals, rng, scenario)
    capacities = hospitals["capacity"].to_numpy()

    pred_dynamic = survival_matrix(patients, hospitals, travel, predicted=True, dynamic=True)
    pred_static = survival_matrix(patients, hospitals, travel, predicted=True, dynamic=False)
    true_dynamic = survival_matrix(patients, hospitals, travel, predicted=False, dynamic=True)

    severity_order = np.argsort(
        patients["true_base_survival"].to_numpy() + 0.005 * patients["deadline_min"].to_numpy()
    )

    assignments = {
        "random": capacity_aware_random(n, capacities, rng),
        "nearest": nearest_assignment(travel, capacities, np.arange(n)),
        "severity_first": nearest_assignment(travel, capacities, severity_order),
        "greedy_dynamic": greedy_assignment(pred_dynamic, capacities, severity_order),
        "static_optimized": optimize_assignment(pred_static, capacities),
        "dynamic_optimized": optimize_assignment(pred_dynamic, capacities),
        "oracle_optimized": optimize_assignment(true_dynamic, capacities),
    }

    rows = []
    oracle_score = None
    for policy in POLICIES:
        metrics = evaluate(patients, hospitals, travel, assignments[policy], true_dynamic)
        if policy == "oracle_optimized":
            oracle_score = metrics["mean_survival"]
        rows.append({"seed": seed, "scenario": scenario, "policy": policy, **metrics})

    for row in rows:
        row["regret_to_oracle"] = float(oracle_score - row["mean_survival"]) if oracle_score is not None else float("nan")
    return rows


def write_svg(summary: pd.DataFrame, out_path: Path) -> None:
    plot = summary.pivot(index="policy", columns="scenario", values="mean_survival_mean").loc[POLICIES]
    width = 980
    height = 430
    margin_left = 135
    margin_bottom = 70
    margin_top = 30
    plot_w = width - margin_left - 30
    plot_h = height - margin_top - margin_bottom
    max_y = max(0.75, float(plot.max().max()) * 1.08)
    colors = {"normal": "#2f6f73", "mass_casualty": "#b45f35", "specialty_shortage": "#6b5aa6"}
    bar_group = plot_w / len(POLICIES)
    bar_w = bar_group / (len(plot.columns) + 1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="24" font-family="Arial" font-size="16" font-weight="700">Mean expected survival by allocation policy</text>',
    ]

    for tick in np.linspace(0, max_y, 5):
        y = margin_top + plot_h - (tick / max_y) * plot_h
        parts.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width-30}" y2="{y:.1f}" stroke="#e6e6e6"/>')
        parts.append(f'<text x="{margin_left-10}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{tick:.2f}</text>')

    for p_idx, policy in enumerate(POLICIES):
        x0 = margin_left + p_idx * bar_group + bar_w * 0.5
        for s_idx, scenario in enumerate(plot.columns):
            value = float(plot.loc[policy, scenario])
            x = x0 + s_idx * bar_w
            h = (value / max_y) * plot_h
            y = margin_top + plot_h - h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w*0.85:.1f}" height="{h:.1f}" fill="{colors.get(scenario, "#777")}"/>')
        label = policy.replace("_", " ")
        lx = x0 + bar_w
        parts.append(f'<text x="{lx:.1f}" y="{height-44}" text-anchor="end" transform="rotate(-35 {lx:.1f},{height-44})" font-family="Arial" font-size="11">{label}</text>')

    legend_x = width - 330
    for idx, scenario in enumerate(plot.columns):
        x = legend_x + idx * 115
        parts.append(f'<rect x="{x}" y="20" width="12" height="12" fill="{colors.get(scenario, "#777")}"/>')
        parts.append(f'<text x="{x+18}" y="31" font-family="Arial" font-size="12">{scenario}</text>')

    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--patients", type=int, default=160)
    parser.add_argument("--hospitals", type=int, default=8)
    parser.add_argument("--patient-features", type=Path, default=None)
    args = parser.parse_args()

    scenarios = ["normal", "mass_casualty", "specialty_shortage"]
    args.out.mkdir(parents=True, exist_ok=True)
    feature_table = None
    if args.patient_features is not None:
        feature_table = pd.read_csv(args.patient_features)

    rows = []
    for scenario in scenarios:
        for rep in range(args.replicates):
            seed = 10_000 * scenarios.index(scenario) + rep
            n = args.patients
            if scenario == "mass_casualty":
                n = int(args.patients * 1.35)
            rows.extend(run_one(seed, scenario, n, args.hospitals, feature_table))

    results = pd.DataFrame(rows)
    results.to_csv(args.out / "allocation_results_raw.csv", index=False)

    metric_cols = [
        "mean_survival",
        "severe_mean_survival",
        "deadline_ok_rate",
        "avg_travel_min",
        "p95_travel_min",
        "max_utilization",
        "capacity_violations",
        "regret_to_oracle",
    ]
    summary = (
        results.groupby(["scenario", "policy"])[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.to_flat_index()]
    summary.to_csv(args.out / "allocation_results_summary.csv", index=False)
    write_svg(summary, args.out / "allocation_survival_bars.svg")

    display_cols = [
        "scenario",
        "policy",
        "mean_survival_mean",
        "deadline_ok_rate_mean",
        "avg_travel_min_mean",
        "regret_to_oracle_mean",
    ]
    print(summary[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved: {args.out / 'allocation_results_summary.csv'}")
    print(f"Saved: {args.out / 'allocation_survival_bars.svg'}")


if __name__ == "__main__":
    main()
