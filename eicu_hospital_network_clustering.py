from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from eicu_discrete_choice_experiment import (
    BASE_SEED,
    DEFAULT_FEATURES,
    DEFAULT_HOSPITALS,
    NEED_CATEGORIES,
    ROOT,
    UNIT_TYPES,
    infer_need_matrix,
    infer_unit_matrix,
    stratified_split,
)


PROFILE_PATH = ROOT / "results_eicu_choice_comparison" / "hospital_profiles_used.csv"
OUT = ROOT / "results_eicu_hospital_network"
PROFILE_ALPHA = 0.72
SUBSTITUTION_THETA = 0.90

NEED_RU = {
    "trauma": "Trauma",
    "burn": "Burns",
    "cardiac": "Cardiac",
    "neuro": "Neuro",
    "respiratory": "Resp.",
    "sepsis": "Sepsis",
    "gi": "GI",
    "surgical": "Surgical",
    "toxicology": "Toxicology",
    "general": "General",
}

UNIT_RU = {
    "Med-Surg ICU": "Med-Surg",
    "MICU": "MICU",
    "SICU": "SICU",
    "Cardiac ICU": "Cardiac",
    "CCU-CTICU": "CCU/CTICU",
    "Neuro ICU": "Neuro",
    "CSICU": "CSICU",
    "CTICU": "CTICU",
}

CLUSTER_COLORS = [
    "#2d6cdf",
    "#c74b50",
    "#208b68",
    "#d28a1e",
    "#7b59c0",
    "#2f7f7f",
    "#a45a2a",
    "#5e6c84",
    "#b5427c",
    "#5f8d3a",
    "#b78a22",
    "#4f63b0",
]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    paths = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
    ]
    for path in paths:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
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


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def profile_matrix(profiles: pd.DataFrame, alpha: float = PROFILE_ALPHA) -> tuple[np.ndarray, list[str], np.ndarray]:
    need_cols = [f"need_{c}" for c in NEED_CATEGORIES]
    unit_cols = [f"unit_{u}" for u in UNIT_TYPES]
    need = profiles[need_cols].to_numpy(float)
    unit = profiles[unit_cols].to_numpy(float)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    x = np.column_stack([math.sqrt(alpha) * need, math.sqrt(1.0 - alpha) * unit])
    labels = [*NEED_CATEGORIES, *UNIT_TYPES]
    return normalize_rows(x), labels, need


def experiment_reproducibility_tables(profiles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    design = pd.DataFrame(
        [
            {
                "parameter_ru": "Patient feature source",
                "value_ru": str(DEFAULT_FEATURES),
                "reproduction_note_ru": "CSV file with prepared eICU Demo patient features.",
            },
            {
                "parameter_ru": "Hospital metadata source",
                "value_ru": str(DEFAULT_HOSPITALS),
                "reproduction_note_ru": "CSV file with bed count, region, and teaching status of hospitals.",
            },
            {
                "parameter_ru": "Clustering unit",
                "value_ru": "hospital",
                "reproduction_note_ru": "One row in hospital_profiles_used.csv corresponds to one hospital.",
            },
            {
                "parameter_ru": "Number of hospitals",
                "value_ru": str(len(profiles)),
                "reproduction_note_ru": "Hospitals included in the profile matrix after filtering by case count are used.",
            },
            {
                "parameter_ru": "Hospital filter",
                "value_ru": "at least 10 cases in the original experiment",
                "reproduction_note_ru": "Profiles are taken from hospital_profiles_used.csv, generated with min_cases_per_hospital=10.",
            },
            {
                "parameter_ru": "Data split",
                "value_ru": "stratified, test_fraction=0.30",
                "reproduction_note_ru": "The r_j^need and r_j^unit profiles are built only from the training portion.",
            },
            {
                "parameter_ru": "Original profiling seed",
                "value_ru": str(BASE_SEED),
                "reproduction_note_ru": "Used for the train/test split and coordinate synthesis in the previous experiment.",
            },
            {
                "parameter_ru": "Clustering seed",
                "value_ru": "20260629 + 17*k",
                "reproduction_note_ru": "A fixed k-means seed is used for each number of clusters k.",
            },
            {
                "parameter_ru": "Profile smoothing",
                "value_ru": "eta=8.0",
                "reproduction_note_ru": "Blends the hospital profile with the global distribution so small hospitals do not produce noisy vectors.",
            },
            {
                "parameter_ru": "r_j^need formula",
                "value_ru": "r_jg=(n_jg + eta*bar_n_g)/(N_j + eta)",
                "reproduction_note_ru": "n_jg -- number of patients at hospital j with need g; bar_n_g -- global share of need g. Components may be non-mutually-exclusive.",
            },
            {
                "parameter_ru": "r_j^unit formula",
                "value_ru": "r_jh=(u_jh + eta*bar_u_h)/(N_j + eta)",
                "reproduction_note_ru": "u_jh -- number of patients at hospital j from unit type h; bar_u_h -- global share of type h.",
            },
            {
                "parameter_ru": "Cluster vector",
                "value_ru": f"v_j=normalize([sqrt({PROFILE_ALPHA:.2f})*r_j^need, sqrt({1.0 - PROFILE_ALPHA:.2f})*r_j^unit])",
                "reproduction_note_ru": "The clinical profile receives greater weight; sensitivity to this parameter is checked separately.",
            },
            {
                "parameter_ru": "Similarity metric",
                "value_ru": "cosine similarity rho_jk",
                "reproduction_note_ru": "Used for the substitution graph and reallocation constraints.",
            },
            {
                "parameter_ru": "Range of k",
                "value_ru": "2..12",
                "reproduction_note_ru": "The macro level is chosen by maximum silhouette; the operational level -- among k>=4 with minimum cluster size at least 8.",
            },
            {
                "parameter_ru": "Effective capacity",
                "value_ru": "C_j=ceil(L_j*1.22 + 2)",
                "reproduction_note_ru": "L_j -- observed hospital load in the profile matrix.",
            },
            {
                "parameter_ru": "Closure scenario",
                "value_ru": f"each hospital closed one at a time, theta={SUBSTITUTION_THETA:.2f}",
                "reproduction_note_ru": "The load of the closed hospital is reallocated only to profile-close hospitals.",
            },
            {
                "parameter_ru": "Overload scenario",
                "value_ru": "top 30% by load within each cluster is multiplied by 1.85",
                "reproduction_note_ru": "A flow problem for reallocating the excess is then solved.",
            },
            {
                "parameter_ru": "Optimizer",
                "value_ru": "maximum flow",
                "reproduction_note_ru": "The volume of covered load is maximized subject to free-capacity and profile-similarity constraints.",
            },
        ]
    )

    need_rules = {
        "trauma": "injury_dx_count>0 or diagnosis contains trauma/fracture/fall/injur/laceration/contusion/crush",
        "burn": "burn_dx_count>0 or diagnosis contains burn",
        "cardiac": "diagnosis contains cardiac/heart/myocard/infarction/rhythm/coronary/angina/cardiogenic/arrhythm",
        "neuro": "diagnosis contains coma/consciousness/stroke/seizure/neuro/intracranial/head/cerebral",
        "respiratory": "diagnosis contains respiratory/pulmonary/pneumonia/asthma/copd/ventilat/hypox/airway",
        "sepsis": "diagnosis contains sepsis/septic",
        "gi": "diagnosis contains gi/gastro/abdomen/abdominal/bleeding/perforation/rupture/pancrea/liver",
        "surgical": "diagnosis contains surgery/surgical/post-op/operative/replacement/transplant/bypass/resection",
        "toxicology": "diagnosis contains overdose/toxin/poison/drug",
        "general": "no match with any specialized need",
    }
    rows = []
    for idx, name in enumerate(NEED_CATEGORIES, start=1):
        rows.append(
            {
                "vector_ru": "r_j^need",
                "component_order": idx,
                "component_name": name,
                "component_name_ru": NEED_RU[name],
                "source_columns": "apacheadmissiondx, injury_dx_count, burn_dx_count",
                "patient_level_rule_ru": need_rules[name],
                "hospital_level_aggregation_ru": "smoothed share of hospital j patients with this need; r_j^need components may overlap",
            }
        )
    for idx, name in enumerate(UNIT_TYPES, start=1):
        rows.append(
            {
                "vector_ru": "r_j^unit",
                "component_order": idx,
                "component_name": name,
                "component_name_ru": UNIT_RU[name],
                "source_columns": "unittype",
                "patient_level_rule_ru": f"unittype == {name}; missing or unknown type is replaced with Med-Surg ICU",
                "hospital_level_aggregation_ru": "smoothed share of hospital j patients from this unit type",
            }
        )
    components = pd.DataFrame(rows)
    return design, components


def pairwise_cosine(x: np.ndarray) -> np.ndarray:
    x = normalize_rows(x)
    return np.clip(x @ x.T, -1.0, 1.0)


def kmeans(x: np.ndarray, k: int, seed: int, n_init: int = 35, max_iter: int = 250) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    best_labels: np.ndarray | None = None
    best_centers: np.ndarray | None = None
    best_inertia = float("inf")
    n = len(x)
    for _ in range(n_init):
        centers = np.empty((k, x.shape[1]), dtype=float)
        first = int(rng.integers(0, n))
        centers[0] = x[first]
        dist2 = np.sum((x[:, None, :] - centers[None, :1, :]) ** 2, axis=2).min(axis=1)
        for c in range(1, k):
            prob = dist2 / max(dist2.sum(), 1e-12)
            idx = int(rng.choice(n, p=prob))
            centers[c] = x[idx]
            dist2 = np.minimum(dist2, np.sum((x - centers[c]) ** 2, axis=1))
        labels = np.zeros(n, dtype=int)
        for _it in range(max_iter):
            dist = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            new_labels = dist.argmin(axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for c in range(k):
                mask = labels == c
                centers[c] = x[mask].mean(axis=0) if mask.any() else x[int(rng.integers(0, n))]
        inertia = float(np.sum((x - centers[labels]) ** 2))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()
    assert best_labels is not None and best_centers is not None
    return best_labels, best_centers, best_inertia


def silhouette_score_from_distance(labels: np.ndarray, distance: np.ndarray) -> float:
    unique = np.unique(labels)
    if len(unique) < 2:
        return 0.0
    scores = np.zeros(len(labels), dtype=float)
    for i in range(len(labels)):
        same = labels == labels[i]
        same[i] = False
        a = float(distance[i, same].mean()) if same.any() else 0.0
        b = float("inf")
        for c in unique:
            if c == labels[i]:
                continue
            other = labels == c
            if other.any():
                b = min(b, float(distance[i, other].mean()))
        scores[i] = (b - a) / max(a, b, 1e-12)
    return float(scores.mean())


def choose_clusters(x: np.ndarray, sim: np.ndarray) -> tuple[pd.DataFrame, int, int, np.ndarray, np.ndarray]:
    dist = 1.0 - sim
    rows = []
    labels_by_k: dict[int, np.ndarray] = {}
    centers_by_k: dict[int, np.ndarray] = {}
    for k in range(2, 13):
        labels, centers, inertia = kmeans(x, k, seed=20260629 + k * 17)
        labels_by_k[k] = labels
        centers_by_k[k] = centers
        sizes = np.bincount(labels, minlength=k)
        rows.append(
            {
                "k": k,
                "inertia": inertia,
                "silhouette": silhouette_score_from_distance(labels, dist),
                "min_cluster_size": int(sizes.min()),
                "max_cluster_size": int(sizes.max()),
            }
        )
    selection = pd.DataFrame(rows)
    feasible = selection[selection["min_cluster_size"] >= 5].copy()
    macro_k = int((feasible if not feasible.empty else selection).sort_values("silhouette", ascending=False).iloc[0]["k"])
    operational = selection[(selection["k"] >= 4) & (selection["min_cluster_size"] >= 8)].copy()
    if operational.empty:
        operational = selection[selection["k"] >= 4].copy()
    operational_k = int(operational.sort_values("silhouette", ascending=False).iloc[0]["k"])
    return selection, macro_k, operational_k, labels_by_k[operational_k], centers_by_k[operational_k]


def relabel_by_size(labels: np.ndarray) -> np.ndarray:
    counts = pd.Series(labels).value_counts().sort_values(ascending=False)
    mapping = {old: new for new, old in enumerate(counts.index.to_list(), start=1)}
    return np.array([mapping[int(label)] for label in labels], dtype=int)


def cluster_summary(profiles: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    rows = []
    need_cols = [f"need_{c}" for c in NEED_CATEGORIES]
    unit_cols = [f"unit_{u}" for u in UNIT_TYPES]
    for cluster in sorted(np.unique(labels)):
        part = profiles[labels == cluster]
        need_mean = part[need_cols].mean()
        unit_mean = part[unit_cols].mean()
        top_need = need_mean.sort_values(ascending=False).head(3)
        top_unit = unit_mean.sort_values(ascending=False).head(2)
        row = {
            "cluster": int(cluster),
            "hospitals": int(len(part)),
            "observed_load": int(part["train_count"].sum()),
            "mean_capacity_norm": float(part["capacity_norm"].mean()),
            "mean_free_capacity_proxy": float(part["free_capacity_proxy"].mean()),
            "mean_quality_proxy": float(part["quality_proxy"].mean()),
            "dominant_needs_ru": "; ".join(f"{NEED_RU[c.replace('need_', '')]}={v:.2f}" for c, v in top_need.items()),
            "dominant_units_ru": "; ".join(f"{UNIT_RU[c.replace('unit_', '')]}={v:.2f}" for c, v in top_unit.items()),
        }
        for col in need_cols:
            row[col] = float(need_mean[col])
        for col in unit_cols:
            row[col] = float(unit_mean[col])
        rows.append(row)
    return pd.DataFrame(rows)


def cluster_interpretation(cluster_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in cluster_df.iterrows():
        cluster = int(row["cluster"])
        med_surg = float(row["unit_Med-Surg ICU"])
        micu = float(row["unit_MICU"])
        cardiac_unit = float(row["unit_Cardiac ICU"])
        capacity = float(row["mean_capacity_norm"])
        general = float(row["need_general"])
        cardiac = float(row["need_cardiac"])

        if cardiac_unit >= 0.30:
            type_ru = "cardiac-oriented specialized type"
            role_ru = "reserve for cardiac and mixed cardiorespiratory flows"
        elif micu >= 0.30:
            type_ru = "medical type with a pronounced MICU component"
            role_ru = "reserve for general, respiratory, and septic patients"
        elif capacity >= 0.55:
            type_ru = "high-capacity multi-profile type"
            role_ru = "reallocation hub during closures or local overload"
        elif med_surg >= 0.75 and general >= 0.35:
            type_ru = "general Med-Surg type"
            role_ru = "baseline high-volume intake for general-profile patients"
        elif med_surg >= 0.75:
            type_ru = "mixed Med-Surg type"
            role_ru = "primary universal circuit of the network"
        else:
            type_ru = "mixed transitional type"
            role_ru = "local reserve between universal and specialized hospitals"

        needs = {NEED_RU[name]: float(row[f"need_{name}"]) for name in NEED_CATEGORIES}
        dominant = ", ".join([name for name, _ in sorted(needs.items(), key=lambda item: item[1], reverse=True)[:3]])
        rows.append(
            {
                "cluster": cluster,
                "type_ru": type_ru,
                "hospitals": int(row["hospitals"]),
                "observed_load": int(row["observed_load"]),
                "dominant_profiles_ru": dominant,
                "dominant_needs_ru": row["dominant_needs_ru"],
                "dominant_units_ru": row["dominant_units_ru"],
                "management_role_ru": role_ru,
            }
        )
    return pd.DataFrame(rows)


def profile_overlap_tables(profiles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    need_cols = [f"need_{c}" for c in NEED_CATEGORIES]
    need_values = profiles[need_cols].to_numpy(float).T
    rows = []
    for a in range(len(NEED_CATEGORIES)):
        for b in range(a + 1, len(NEED_CATEGORIES)):
            va = need_values[a]
            vb = need_values[b]
            sim = float((va @ vb) / max(np.linalg.norm(va) * np.linalg.norm(vb), 1e-12))
            corr = float(np.corrcoef(va, vb)[0, 1]) if np.std(va) > 1e-12 and np.std(vb) > 1e-12 else 0.0
            rows.append(
                {
                    "profile_a": NEED_CATEGORIES[a],
                    "profile_b": NEED_CATEGORIES[b],
                    "profile_a_ru": NEED_RU[NEED_CATEGORIES[a]],
                    "profile_b_ru": NEED_RU[NEED_CATEGORIES[b]],
                    "hospital_level_cosine": sim,
                    "hospital_level_correlation": corr,
                }
            )
    hospital_overlap = pd.DataFrame(rows).sort_values("hospital_level_cosine", ascending=False)

    features = pd.read_csv(DEFAULT_FEATURES)
    need = infer_need_matrix(features)
    patient_rows = []
    for a in range(len(NEED_CATEGORIES)):
        for b in range(a + 1, len(NEED_CATEGORIES)):
            va = need[:, a]
            vb = need[:, b]
            cos = float((va @ vb) / max(np.linalg.norm(va) * np.linalg.norm(vb), 1e-12))
            both = float(((va > 0) & (vb > 0)).mean())
            patient_rows.append(
                {
                    "profile_a": NEED_CATEGORIES[a],
                    "profile_b": NEED_CATEGORIES[b],
                    "profile_a_ru": NEED_RU[NEED_CATEGORIES[a]],
                    "profile_b_ru": NEED_RU[NEED_CATEGORIES[b]],
                    "patient_cooccurrence_cosine": cos,
                    "patient_share_both": both,
                }
            )
    patient_overlap = pd.DataFrame(patient_rows).sort_values("patient_cooccurrence_cosine", ascending=False)
    return hospital_overlap, patient_overlap


def substitution_edges(profiles: pd.DataFrame, labels: np.ndarray, sim: np.ndarray, top_k: int = 4) -> pd.DataFrame:
    ids = profiles["hospital_id"].astype(int).to_numpy()
    rows = []
    for i, hospital_id in enumerate(ids):
        order = np.argsort(-sim[i])
        rank = 0
        for j in order:
            if i == j:
                continue
            rank += 1
            rows.append(
                {
                    "source_hospital": int(hospital_id),
                    "target_hospital": int(ids[j]),
                    "source_cluster": int(labels[i]),
                    "target_cluster": int(labels[j]),
                    "rank": int(rank),
                    "profile_similarity": float(sim[i, j]),
                    "same_cluster": int(labels[i] == labels[j]),
                    "target_free_capacity_proxy": float(profiles.iloc[j]["free_capacity_proxy"]),
                    "target_quality_proxy": float(profiles.iloc[j]["quality_proxy"]),
                }
            )
            if rank >= top_k:
                break
    return pd.DataFrame(rows)


def effective_capacity(load: np.ndarray, reserve: float = 0.22, min_reserve: int = 2) -> np.ndarray:
    return np.ceil(load * (1.0 + reserve) + min_reserve).astype(int)


def flow_reallocate(
    initial_load: np.ndarray,
    capacity: np.ndarray,
    source_excess: np.ndarray,
    sim: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    closed_index: int | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    load = initial_load.astype(float).copy()
    if closed_index is not None:
        load[closed_index] = 0.0
        capacity = capacity.copy()
        capacity[closed_index] = 0
    spare = np.maximum(capacity.astype(float) - load, 0.0)
    n = len(load)
    source_node = n
    sink_node = n + 1
    graph: list[list[list[float | int]]] = [[] for _ in range(n + 2)]

    def add_edge(u: int, v: int, cap: float) -> tuple[int, int]:
        fwd = [v, len(graph[v]), float(cap)]
        rev = [u, len(graph[u]), 0.0]
        graph[u].append(fwd)
        graph[v].append(rev)
        return u, len(graph[u]) - 1

    eps = 1e-9
    total_source = float(source_excess.sum())
    for src, amount in enumerate(source_excess):
        if amount > eps:
            add_edge(source_node, src, float(amount))
    for dst, amount in enumerate(spare):
        if amount > eps and dst != closed_index:
            add_edge(dst, sink_node, float(amount))

    records: list[tuple[int, int, int]] = []
    inf = max(total_source, float(spare.sum()), 1.0)
    for src, amount in enumerate(source_excess):
        if amount <= eps:
            continue
        for dst in range(n):
            if dst == src or dst == closed_index or spare[dst] <= eps:
                continue
            if sim[src, dst] < threshold:
                continue
            _, edge_idx = add_edge(src, dst, inf)
            records.append((src, dst, edge_idx))

    def bfs() -> list[int]:
        level = [-1] * (n + 2)
        level[source_node] = 0
        queue = [source_node]
        for u in queue:
            for v, _, cap in graph[u]:
                if float(cap) > eps and level[int(v)] < 0:
                    level[int(v)] = level[u] + 1
                    queue.append(int(v))
        return level

    def dfs(u: int, pushed: float, level: list[int], it: list[int]) -> float:
        if u == sink_node:
            return pushed
        while it[u] < len(graph[u]):
            edge = graph[u][it[u]]
            v, rev, cap = int(edge[0]), int(edge[1]), float(edge[2])
            if cap > eps and level[v] == level[u] + 1:
                tr = dfs(v, min(pushed, cap), level, it)
                if tr > eps:
                    edge[2] = float(edge[2]) - tr
                    graph[v][rev][2] = float(graph[v][rev][2]) + tr
                    return tr
            it[u] += 1
        return 0.0

    max_flow = 0.0
    while True:
        level = bfs()
        if level[sink_node] < 0:
            break
        it = [0] * (n + 2)
        while True:
            pushed = dfs(source_node, inf, level, it)
            if pushed <= eps:
                break
            max_flow += pushed

    moves = []
    for src, dst, edge_idx in records:
        edge = graph[src][edge_idx]
        rev_idx = int(edge[1])
        moved = float(graph[dst][rev_idx][2])
        if moved <= eps:
            continue
        moves.append(
            {
                "source_index": int(src),
                "target_index": int(dst),
                "moved": moved,
                "similarity": float(sim[src, dst]),
                "same_cluster": int(labels[src] == labels[dst]),
                "source_cluster": int(labels[src]),
                "target_cluster": int(labels[dst]),
            }
        )
    residual = total_source - max_flow
    return np.array([residual]), pd.DataFrame(moves)


def closure_robustness(profiles: pd.DataFrame, labels: np.ndarray, sim: np.ndarray, threshold: float = 0.90) -> tuple[pd.DataFrame, pd.DataFrame]:
    load = profiles["train_count"].to_numpy(float)
    capacity = effective_capacity(load)
    rows = []
    all_moves = []
    ids = profiles["hospital_id"].astype(int).to_numpy()
    for closed in range(len(load)):
        source_excess = np.zeros_like(load)
        source_excess[closed] = load[closed]
        residual_arr, moves = flow_reallocate(load, capacity, source_excess, sim, labels, threshold=threshold, closed_index=closed)
        moved = float(moves["moved"].sum()) if not moves.empty else 0.0
        residual = float(residual_arr[0])
        if not moves.empty:
            moves = moves.copy()
            moves["closed_hospital"] = int(ids[closed])
            moves["target_hospital"] = moves["target_index"].map(lambda idx: int(ids[int(idx)]))
            all_moves.append(moves)
            mean_sim = float(np.average(moves["similarity"], weights=moves["moved"]))
            same_cluster_share = float(np.average(moves["same_cluster"], weights=moves["moved"]))
            main_cluster = int(moves.groupby("target_cluster")["moved"].sum().idxmax())
        else:
            mean_sim = float("nan")
            same_cluster_share = float("nan")
            main_cluster = -1
        rows.append(
            {
                "closed_hospital": int(ids[closed]),
                "cluster": int(labels[closed]),
                "closed_load": float(load[closed]),
                "coverage": moved / max(load[closed], 1e-12),
                "uncovered_load": residual,
                "mean_substitute_similarity": mean_sim,
                "same_cluster_share": same_cluster_share,
                "main_replacement_cluster": main_cluster,
                "substitute_hospitals_used": int(moves["target_index"].nunique()) if not moves.empty else 0,
            }
        )
    move_df = pd.concat(all_moves, ignore_index=True) if all_moves else pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["coverage", "closed_load"], ascending=[True, False]), move_df


def load_balancing(
    profiles: pd.DataFrame,
    labels: np.ndarray,
    sim: np.ndarray,
    thresholds: tuple[float, ...] = (0.00, 0.80, 0.85, 0.90, 0.95),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    load = profiles["train_count"].to_numpy(float)
    capacity = effective_capacity(load)
    surge_load = load.copy()
    hot = np.zeros(len(load), dtype=bool)
    for cluster in sorted(np.unique(labels)):
        idx = np.flatnonzero(labels == cluster)
        cutoff = np.quantile(load[idx], 0.70)
        hot[idx] = load[idx] >= cutoff
    surge_load[hot] = np.ceil(load[hot] * 1.85)
    overload = np.maximum(surge_load - capacity, 0.0)
    capped_initial = np.minimum(surge_load, capacity)
    rows = []
    move_frames = []
    before_total = float(overload.sum())
    before_count = int((overload > 0).sum())
    for threshold in thresholds:
        residual_arr, moves = flow_reallocate(capped_initial, capacity, overload, sim, labels, threshold=threshold)
        moved = float(moves["moved"].sum()) if not moves.empty else 0.0
        residual = float(residual_arr[0])
        if not moves.empty:
            moves = moves.copy()
            moves["threshold"] = threshold
            move_frames.append(moves)
            mean_sim = float(np.average(moves["similarity"], weights=moves["moved"]))
            same_cluster_share = float(np.average(moves["same_cluster"], weights=moves["moved"]))
        else:
            mean_sim = float("nan")
            same_cluster_share = float("nan")
        rows.append(
            {
                "similarity_threshold": threshold,
                "overload_before": before_total,
                "overloaded_hospitals_before": before_count,
                "hotspot_hospitals": int(hot.sum()),
                "moved_load": moved,
                "residual_overload": residual,
                "overload_reduction": before_total - residual,
                "overload_reduction_share": (before_total - residual) / max(before_total, 1e-12),
                "mean_move_similarity": mean_sim,
                "same_cluster_move_share": same_cluster_share,
            }
        )
    moves_df = pd.concat(move_frames, ignore_index=True) if move_frames else pd.DataFrame()
    return pd.DataFrame(rows), moves_df


def alpha_sensitivity(profiles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for alpha in np.round(np.arange(0.0, 1.0001, 0.04), 2):
        x_alpha, _, _ = profile_matrix(profiles, alpha=float(alpha))
        sim_alpha = pairwise_cosine(x_alpha)
        selection, macro_k, operational_k, labels_raw, _ = choose_clusters(x_alpha, sim_alpha)
        labels_alpha = relabel_by_size(labels_raw)
        closure, _ = closure_robustness(profiles, labels_alpha, sim_alpha, threshold=SUBSTITUTION_THETA)
        balance, _ = load_balancing(profiles, labels_alpha, sim_alpha, thresholds=(SUBSTITUTION_THETA,))
        operational_row = selection[selection["k"] == operational_k].iloc[0]
        macro_row = selection[selection["k"] == macro_k].iloc[0]
        balance_row = balance.iloc[0]
        rows.append(
            {
                "alpha": float(alpha),
                "macro_k": int(macro_k),
                "macro_silhouette": float(macro_row["silhouette"]),
                "operational_k": int(operational_k),
                "operational_silhouette": float(operational_row["silhouette"]),
                "mean_closure_coverage": float(closure["coverage"].mean()),
                "critical_closure_share": float((closure["coverage"] < 0.80).mean()),
                "overload_reduction_share_at_theta": float(balance_row["overload_reduction_share"]),
                "mean_move_similarity_at_theta": float(balance_row["mean_move_similarity"]),
                "same_cluster_move_share_at_theta": float(balance_row["same_cluster_move_share"]),
                "selected_alpha": int(abs(float(alpha) - PROFILE_ALPHA) < 1e-9),
            }
        )
    return pd.DataFrame(rows)


def theta_sensitivity(profiles: pd.DataFrame, labels: np.ndarray, sim: np.ndarray) -> pd.DataFrame:
    thresholds = (0.00, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98)
    rows = []
    for threshold in thresholds:
        closure, _ = closure_robustness(profiles, labels, sim, threshold=threshold)
        balance, _ = load_balancing(profiles, labels, sim, thresholds=(threshold,))
        balance_row = balance.iloc[0]
        rows.append(
            {
                "theta": float(threshold),
                "mean_closure_coverage": float(closure["coverage"].mean()),
                "critical_closure_share": float((closure["coverage"] < 0.80).mean()),
                "overload_reduction_share": float(balance_row["overload_reduction_share"]),
                "mean_move_similarity": float(balance_row["mean_move_similarity"]),
                "same_cluster_move_share": float(balance_row["same_cluster_move_share"]),
                "selected_theta": int(abs(float(threshold) - SUBSTITUTION_THETA) < 1e-9),
            }
        )
    return pd.DataFrame(rows)


def patient_hospital_fit_validation(
    profiles: pd.DataFrame,
    labels: np.ndarray,
    sim: np.ndarray,
    alpha: float = PROFILE_ALPHA,
    theta: float = SUBSTITUTION_THETA,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = pd.read_csv(DEFAULT_FEATURES).reset_index(drop=True)
    features = features.dropna(subset=["subject_id"]).copy()
    features["subject_id"] = features["subject_id"].astype(int)
    features["_source_index"] = np.arange(len(features))
    counts = features["subject_id"].value_counts()
    eligible_ids = counts[counts >= 10].index.astype(int).to_numpy()
    data = features[features["subject_id"].isin(eligible_ids)].reset_index(drop=True).copy()
    train_idx, test_idx = stratified_split(data, "subject_id", 0.30, BASE_SEED)
    test = data.iloc[test_idx].reset_index(drop=True).copy()

    full_need = infer_need_matrix(features)
    full_unit = infer_unit_matrix(features)
    patient_need = full_need[test["_source_index"].to_numpy()]
    patient_unit = full_unit[test["_source_index"].to_numpy()]

    hospital_ids = profiles["hospital_id"].astype(int).to_numpy()
    id_to_idx = {int(hid): idx for idx, hid in enumerate(hospital_ids)}
    actual_idx = np.array([id_to_idx[int(hid)] for hid in test["subject_id"].astype(int).to_numpy()], dtype=int)
    rows = np.arange(len(test))

    hospital_need = profiles[[f"need_{name}" for name in NEED_CATEGORIES]].to_numpy(float)
    hospital_unit = profiles[[f"unit_{name}" for name in UNIT_TYPES]].to_numpy(float)
    active_need_count = np.maximum(patient_need.sum(axis=1, keepdims=True), 1.0)
    clinical_match = (patient_need @ hospital_need.T) / active_need_count
    unit_match = patient_unit @ hospital_unit.T
    suitability = alpha * clinical_match + (1.0 - alpha) * unit_match

    actual_fit = suitability[rows, actual_idx]
    actual_clinical = clinical_match[rows, actual_idx]
    actual_unit = unit_match[rows, actual_idx]
    random_fit = suitability.mean(axis=1)
    best_fit = suitability.max(axis=1)
    ranks = 1 + (suitability > actual_fit[:, None]).sum(axis=1)
    actual_relative = actual_fit / np.maximum(best_fit, 1e-12)

    same_cluster = labels[None, :] == labels[actual_idx][:, None]
    best_same_cluster = np.where(same_cluster, suitability, -np.inf).max(axis=1)

    allowed_substitute = sim[actual_idx] >= theta
    allowed_substitute[rows, actual_idx] = False
    has_substitute = allowed_substitute.any(axis=1)
    best_substitute_fit = np.full(len(test), np.nan)
    best_substitute_similarity = np.full(len(test), np.nan)
    best_substitute_same_cluster = np.full(len(test), np.nan)
    for i in range(len(test)):
        if not has_substitute[i]:
            continue
        candidates = np.flatnonzero(allowed_substitute[i])
        best_local = candidates[np.argmax(suitability[i, candidates])]
        best_substitute_fit[i] = suitability[i, best_local]
        best_substitute_similarity[i] = sim[actual_idx[i], best_local]
        best_substitute_same_cluster[i] = float(labels[actual_idx[i]] == labels[best_local])

    metrics = pd.DataFrame(
        [
            {
                "test_patients": int(len(test)),
                "candidate_hospitals": int(len(hospital_ids)),
                "alpha": float(alpha),
                "theta": float(theta),
                "actual_mean_suitability": float(actual_fit.mean()),
                "random_mean_suitability": float(random_fit.mean()),
                "best_possible_mean_suitability": float(best_fit.mean()),
                "best_same_cluster_mean_suitability": float(best_same_cluster.mean()),
                "actual_relative_to_best_mean": float(actual_relative.mean()),
                "actual_above_random_share": float((actual_fit > random_fit).mean()),
                "actual_top5_share": float((ranks <= 5).mean()),
                "actual_top10_share": float((ranks <= 10).mean()),
                "actual_top25_share": float((ranks <= 25).mean()),
                "actual_median_rank": float(np.median(ranks)),
                "actual_mean_clinical_match": float(actual_clinical.mean()),
                "actual_mean_unit_match": float(actual_unit.mean()),
                "substitute_available_share": float(has_substitute.mean()),
                "best_substitute_mean_suitability": float(np.nanmean(best_substitute_fit)),
                "best_substitute_relative_to_best_mean": float(np.nanmean(best_substitute_fit / np.maximum(best_fit, 1e-12))),
                "best_substitute_mean_similarity": float(np.nanmean(best_substitute_similarity)),
                "best_substitute_same_cluster_share": float(np.nanmean(best_substitute_same_cluster)),
                "best_substitute_near_best_share": float(np.nanmean(best_substitute_fit >= 0.90 * best_fit)),
            }
        ]
    )

    example = test[
        [
            "stay_id",
            "subject_id",
            "unittype",
            "apacheadmissiondx",
            "survival_t0",
            "iss_proxy",
        ]
    ].copy()
    example["actual_hospital_id"] = test["subject_id"].astype(int).to_numpy()
    example["actual_cluster"] = labels[actual_idx]
    example["actual_suitability"] = actual_fit
    example["random_mean_suitability"] = random_fit
    example["best_possible_suitability"] = best_fit
    example["actual_rank"] = ranks
    example["actual_relative_to_best"] = actual_relative
    example["best_same_cluster_suitability"] = best_same_cluster
    example["substitute_available"] = has_substitute.astype(int)
    example["best_substitute_suitability"] = best_substitute_fit
    example["best_substitute_similarity"] = best_substitute_similarity
    example = example.sort_values(["actual_relative_to_best", "actual_suitability"], ascending=[False, False]).head(30)
    return metrics, example


def pca2(x: np.ndarray) -> np.ndarray:
    xc = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    coords = xc @ vt[:2].T
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    return (coords - lo) / np.maximum(hi - lo, 1e-12)


def save_cluster_selection(selection: pd.DataFrame, macro_k: int, operational_k: int, path: Path) -> None:
    scale = 2
    width, height = 1100, 620
    img = Image.new("RGB", (width * scale, height * scale), "white")
    draw = ImageDraw.Draw(img)
    draw_text(draw, (34 * scale, 26 * scale), "Choosing the number of hospital types", 28 * scale, bold=True)
    draw_text(draw, (34 * scale, 66 * scale), "Clustering of profile vectors: clinical needs + ICU unit types", 16 * scale, fill="#555")
    left, top, plot_w, plot_h = 90 * scale, 125 * scale, 840 * scale, 360 * scale
    draw.rectangle([left, top, left + plot_w, top + plot_h], fill="#fbfbfb", outline="#cccccc", width=2 * scale)
    k_values = selection["k"].to_numpy(float)
    sil = selection["silhouette"].to_numpy(float)
    inertia = selection["inertia"].to_numpy(float)
    inertia_norm = (inertia.max() - inertia) / max(inertia.max() - inertia.min(), 1e-12)
    y_series = [("Silhouette", sil, "#2d6cdf"), ("Inertia reduction", inertia_norm, "#c74b50")]
    for tick in np.linspace(0, 1, 6):
        y = top + plot_h - tick * plot_h
        draw.line([(left, y), (left + plot_w, y)], fill="#e4e4e4")
        draw_text(draw, (38 * scale, int(y - 8 * scale)), f"{tick:.1f}", 13 * scale, fill="#555")
    for tick in k_values:
        x = left + (tick - k_values.min()) / (k_values.max() - k_values.min()) * plot_w
        draw.line([(x, top), (x, top + plot_h)], fill="#eeeeee")
        draw_text(draw, (int(x - 7 * scale), top + plot_h + 18 * scale), f"{int(tick)}", 13 * scale, fill="#555")
    for label, values, color in y_series:
        pts = []
        for k, value in zip(k_values, values):
            x = left + (k - k_values.min()) / (k_values.max() - k_values.min()) * plot_w
            y = top + plot_h - value * plot_h
            pts.append((x, y))
        draw.line(pts, fill=color, width=4 * scale)
        for x, y in pts:
            draw.ellipse([x - 5 * scale, y - 5 * scale, x + 5 * scale, y + 5 * scale], fill=color)
    macro_x = left + (macro_k - k_values.min()) / (k_values.max() - k_values.min()) * plot_w
    operational_x = left + (operational_k - k_values.min()) / (k_values.max() - k_values.min()) * plot_w
    draw.line([(macro_x, top), (macro_x, top + plot_h)], fill="#222", width=2 * scale)
    draw.line([(operational_x, top), (operational_x, top + plot_h)], fill="#7b3f95", width=2 * scale)
    draw_text(draw, (int(macro_x + 10 * scale), top + 15 * scale), f"macro k={macro_k}", 15 * scale, bold=True)
    draw_text(draw, (int(operational_x + 10 * scale), top + 45 * scale), f"operational k={operational_k}", 15 * scale, fill="#7b3f95", bold=True)
    draw_text(draw, (left + 360 * scale, top + plot_h + 55 * scale), "Number of clusters k", 16 * scale)
    draw.rectangle([945 * scale, 160 * scale, 965 * scale, 180 * scale], fill="#2d6cdf")
    draw_text(draw, (974 * scale, 157 * scale), "Silhouette", 15 * scale)
    draw.rectangle([945 * scale, 195 * scale, 965 * scale, 215 * scale], fill="#c74b50")
    draw_text(draw, (974 * scale, 192 * scale), "Inertia", 15 * scale)
    img = img.resize((width, height), Image.Resampling.LANCZOS)
    img.save(path)


def save_hospital_cluster_scatter(profiles: pd.DataFrame, labels: np.ndarray, x: np.ndarray, cluster_df: pd.DataFrame, path: Path) -> None:
    coords = pca2(x)
    scale = 2
    width, height = 1320, 760
    img = Image.new("RGB", (width * scale, height * scale), "white")
    draw = ImageDraw.Draw(img)
    draw_text(draw, (34 * scale, 26 * scale), "Clustering of hospitals in profile space", 28 * scale, bold=True)
    draw_text(draw, (34 * scale, 66 * scale), "Each point is a hospital; color is the operational type; size is the observed load", 16 * scale, fill="#555")

    left, top, plot_w, plot_h = 80 * scale, 120 * scale, 820 * scale, 520 * scale
    draw.rectangle([left, top, left + plot_w, top + plot_h], fill="#fbfbfb", outline="#cccccc", width=2 * scale)
    for tick in np.linspace(0, 1, 6):
        x0 = left + tick * plot_w
        y0 = top + plot_h - tick * plot_h
        draw.line([(x0, top), (x0, top + plot_h)], fill="#eeeeee")
        draw.line([(left, y0), (left + plot_w, y0)], fill="#eeeeee")
    draw_text(draw, (left + 275 * scale, top + plot_h + 42 * scale), "First axis of profile variation", 15 * scale)
    draw_text(draw, (20 * scale, top + 230 * scale), "Second axis\nof profile\nvariation", 13 * scale, fill="#555")

    loads = profiles["train_count"].to_numpy(float)
    lo, hi = float(loads.min()), float(loads.max())
    order = np.argsort(loads)
    for idx in order:
        cluster = int(labels[idx])
        color = CLUSTER_COLORS[(cluster - 1) % len(CLUSTER_COLORS)]
        px = left + coords[idx, 0] * plot_w
        py = top + (1 - coords[idx, 1]) * plot_h
        radius = (4.5 + 7.5 * math.sqrt((loads[idx] - lo) / max(hi - lo, 1e-12))) * scale
        draw.ellipse([px - radius, py - radius, px + radius, py + radius], fill=color, outline="white", width=1 * scale)

    for cluster in sorted(cluster_df["cluster"].astype(int)):
        mask = labels == cluster
        cx = left + coords[mask, 0].mean() * plot_w
        cy = top + (1 - coords[mask, 1].mean()) * plot_h
        color = CLUSTER_COLORS[(cluster - 1) % len(CLUSTER_COLORS)]
        radius = 19 * scale
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=color, outline="#111", width=2 * scale)
        label = str(cluster)
        bbox = draw.textbbox((0, 0), label, font=font(17 * scale, bold=True))
        draw.text((cx - (bbox[2] - bbox[0]) / 2, cy - (bbox[3] - bbox[1]) / 2 - 1 * scale), label, font=font(17 * scale, bold=True), fill="white")

    draw_text(draw, (940 * scale, 130 * scale), "Cluster interpretation", 19 * scale, bold=True)
    for pos, row in enumerate(cluster_df.itertuples(index=False)):
        y = 176 * scale + pos * 72 * scale
        cluster = int(row.cluster)
        color = CLUSTER_COLORS[(cluster - 1) % len(CLUSTER_COLORS)]
        draw.ellipse([942 * scale, y - 12 * scale, 966 * scale, y + 12 * scale], fill=color, outline="#222")
        draw_text(draw, (978 * scale, y - 18 * scale), f"Cluster {cluster}: {int(row.hospitals)} hospitals", 14 * scale, bold=True)
        draw_text(draw, (978 * scale, y + 4 * scale), f"load {int(row.observed_load)}; {str(row.dominant_needs_ru)[:44]}", 12 * scale, fill="#555")
        draw_text(draw, (978 * scale, y + 23 * scale), str(row.dominant_units_ru)[:52], 12 * scale, fill="#555")

    img = img.resize((width, height), Image.Resampling.LANCZOS)
    img.save(path)


def save_cluster_interpretation_table(interpretation: pd.DataFrame, path: Path) -> None:
    scale = 2
    width, height = 1420, 640
    img = Image.new("RGB", (width * scale, height * scale), "white")
    draw = ImageDraw.Draw(img)
    draw_text(draw, (34 * scale, 26 * scale), "Semantic interpretation of identified types", 28 * scale, bold=True)
    draw_text(draw, (34 * scale, 66 * scale), "Clusters are translated from numeric labels into managerial hospital types", 16 * scale, fill="#555")

    left, top = 34 * scale, 118 * scale
    col_w = [110, 365, 170, 285, 390]
    col_x = [left]
    for w in col_w[:-1]:
        col_x.append(col_x[-1] + w * scale)
    headers = ["Type", "Interpretation", "Size", "Leading profiles", "Managerial role"]
    row_h = 86 * scale
    table_w = sum(col_w) * scale
    draw.rectangle([left, top, left + table_w, top + 38 * scale], fill="#f0f3f8", outline="#d5d9e2")
    for idx, header in enumerate(headers):
        draw_text(draw, (col_x[idx] + 10 * scale, top + 10 * scale), header, 13 * scale, bold=True)
    for row_idx, row in enumerate(interpretation.itertuples(index=False)):
        y = top + 38 * scale + row_idx * row_h
        fill = "#ffffff" if row_idx % 2 == 0 else "#fafafa"
        draw.rectangle([left, y, left + table_w, y + row_h], fill=fill, outline="#e2e2e2")
        cluster = int(row.cluster)
        color = CLUSTER_COLORS[(cluster - 1) % len(CLUSTER_COLORS)]
        draw.ellipse([col_x[0] + 14 * scale, y + 24 * scale, col_x[0] + 42 * scale, y + 52 * scale], fill=color, outline="#222")
        draw_text(draw, (col_x[0] + 53 * scale, y + 27 * scale), str(cluster), 16 * scale, bold=True)

        values = [
            str(row.type_ru),
            f"{int(row.hospitals)} hospitals\nload {int(row.observed_load)}",
            str(row.dominant_profiles_ru),
            str(row.management_role_ru),
        ]
        columns = [1, 2, 3, 4]
        wraps = [38, 18, 28, 42]
        for value, col, wrap_width in zip(values, columns, wraps):
            lines: list[str] = []
            for part in value.split("\n"):
                lines.extend(textwrap.wrap(part, width=wrap_width) or [""])
            for line_idx, line in enumerate(lines[:3]):
                draw_text(draw, (col_x[col] + 10 * scale, y + (12 + line_idx * 18) * scale), line, 12 * scale, fill="#222")

    img = img.resize((width, height), Image.Resampling.LANCZOS)
    img.save(path)


def save_cluster_heatmap(cluster_df: pd.DataFrame, path: Path) -> None:
    need_cols = [f"need_{c}" for c in NEED_CATEGORIES]
    data = cluster_df[need_cols].to_numpy(float)
    scale = 2
    width = 1420
    height = 170 + 58 * len(cluster_df)
    img = Image.new("RGB", (width * scale, height * scale), "white")
    draw = ImageDraw.Draw(img)
    draw_text(draw, (34 * scale, 26 * scale), "Profiles of the identified hospital types", 28 * scale, bold=True)
    draw_text(draw, (34 * scale, 66 * scale), "Mean strength of clinical profiles within each cluster", 16 * scale, fill="#555")
    left, top = 270 * scale, 135 * scale
    cell_w, row_h = 98 * scale, 46 * scale
    for c, name in enumerate(NEED_CATEGORIES):
        draw_text(draw, (left + c * cell_w + 8 * scale, 105 * scale), NEED_RU[name], 13 * scale, bold=True)
    vmin, vmax = float(data.min()), float(data.max())
    for r, row in enumerate(cluster_df.itertuples(index=False)):
        y = top + r * row_h
        color = CLUSTER_COLORS[(int(row.cluster) - 1) % len(CLUSTER_COLORS)]
        draw.rectangle([34 * scale, y, 48 * scale, y + row_h], fill=color)
        draw_text(draw, (58 * scale, y + 10 * scale), f"Cluster {int(row.cluster)}", 15 * scale, bold=True)
        draw_text(draw, (155 * scale, y + 10 * scale), f"n={int(row.hospitals)}", 14 * scale, fill="#555")
        for c, value in enumerate(data[r]):
            t = (value - vmin) / max(vmax - vmin, 1e-12)
            fill = tuple(int(round(245 * (1 - t) + base * t)) for base in (45, 108, 223))
            x = left + c * cell_w
            draw.rectangle([x, y, x + cell_w, y + row_h], fill=fill, outline="#eeeeee")
            text = f"{value:.2f}"
            bbox = draw.textbbox((0, 0), text, font=font(13 * scale, bold=True))
            draw.text((x + cell_w / 2 - (bbox[2] - bbox[0]) / 2, y + row_h / 2 - 8 * scale), text, font=font(13 * scale, bold=True), fill="#111")
    img = img.resize((width, height), Image.Resampling.LANCZOS)
    img.save(path)


def save_overlap_bar(overlap: pd.DataFrame, path: Path) -> None:
    top = overlap.head(12).copy()
    scale = 2
    width, height = 1260, 720
    img = Image.new("RGB", (width * scale, height * scale), "white")
    draw = ImageDraw.Draw(img)
    draw_text(draw, (34 * scale, 26 * scale), "Which clinical profiles overlap", 28 * scale, bold=True)
    draw_text(draw, (34 * scale, 66 * scale), "Cosine similarity of profile distributions across hospitals; higher means stronger overlap", 16 * scale, fill="#555")
    left, top_y, bar_w, row_h = 420 * scale, 125 * scale, 650 * scale, 43 * scale
    xmax = min(1.0, max(0.8, float(top["hospital_level_cosine"].max()) * 1.05))
    for tick in np.linspace(0, xmax, 6):
        x = left + tick / xmax * bar_w
        draw.line([(x, top_y - 25 * scale), (x, top_y + row_h * len(top))], fill="#e4e4e4")
        draw_text(draw, (int(x - 15 * scale), top_y + row_h * len(top) + 18 * scale), f"{tick:.1f}", 13 * scale, fill="#555")
    for idx, row in enumerate(top.itertuples(index=False)):
        y = top_y + idx * row_h
        label = f"{row.profile_a_ru} - {row.profile_b_ru}"
        draw_text(draw, (34 * scale, y + 8 * scale), label, 15 * scale)
        value = float(row.hospital_level_cosine)
        draw.rounded_rectangle([left, y + 7 * scale, left + value / xmax * bar_w, y + 29 * scale], radius=5 * scale, fill="#2d6cdf")
        draw_text(draw, (int(left + value / xmax * bar_w + 10 * scale), y + 4 * scale), f"{value:.3f}", 14 * scale, bold=True)
    draw_text(draw, (left + 230 * scale, top_y + row_h * len(top) + 55 * scale), "Profile similarity", 16 * scale)
    img = img.resize((width, height), Image.Resampling.LANCZOS)
    img.save(path)


def save_cluster_graph(cluster_df: pd.DataFrame, profiles: pd.DataFrame, labels: np.ndarray, x: np.ndarray, sim: np.ndarray, path: Path) -> None:
    cluster_ids = sorted(cluster_df["cluster"].astype(int).to_list())
    cluster_vectors = []
    for cluster in cluster_ids:
        cluster_vectors.append(x[labels == cluster].mean(axis=0))
    cluster_vectors = normalize_rows(np.vstack(cluster_vectors))
    c_sim = pairwise_cosine(cluster_vectors)
    coords = pca2(cluster_vectors)
    scale = 2
    width, height = 1180, 760
    img = Image.new("RGB", (width * scale, height * scale), "white")
    draw = ImageDraw.Draw(img)
    draw_text(draw, (34 * scale, 26 * scale), "Substitution graph of hospital types", 28 * scale, bold=True)
    draw_text(draw, (34 * scale, 66 * scale), "Solid edges are strong profile similarity; dashed is the nearest reserve type", 16 * scale, fill="#555")
    left, top, plot_w, plot_h = 70 * scale, 120 * scale, 760 * scale, 520 * scale
    pts = {}
    for idx, cluster in enumerate(cluster_ids):
        pts[cluster] = (
            left + (0.08 + 0.84 * coords[idx, 0]) * plot_w,
            top + (0.08 + 0.84 * (1 - coords[idx, 1])) * plot_h,
        )

    def dashed_line(p1: tuple[float, float], p2: tuple[float, float], fill: str, width_line: int) -> None:
        x1, y1 = p1
        x2, y2 = p2
        length = math.hypot(x2 - x1, y2 - y1)
        if length <= 0:
            return
        dash = 10 * scale
        gap = 7 * scale
        steps = int(length // (dash + gap)) + 1
        for step_idx in range(steps):
            start = step_idx * (dash + gap)
            end = min(start + dash, length)
            if start >= length:
                break
            xa = x1 + (x2 - x1) * start / length
            ya = y1 + (y2 - y1) * start / length
            xb = x1 + (x2 - x1) * end / length
            yb = y1 + (y2 - y1) * end / length
            draw.line([(xa, ya), (xb, yb)], fill=fill, width=width_line)

    strong_edges: set[tuple[int, int]] = set()
    reserve_edges: set[tuple[int, int]] = set()
    for a in range(len(cluster_ids)):
        for b in range(a + 1, len(cluster_ids)):
            value = float(c_sim[a, b])
            if value >= 0.88:
                strong_edges.add((a, b))

    for a in range(len(cluster_ids)):
        candidates = [(float(c_sim[a, b]), b) for b in range(len(cluster_ids)) if b != a]
        _, best_b = max(candidates, key=lambda item: item[0])
        edge = tuple(sorted((a, best_b)))
        if edge not in strong_edges:
            reserve_edges.add(edge)

    for a, b in sorted(reserve_edges):
        value = float(c_sim[a, b])
        p1 = pts[cluster_ids[a]]
        p2 = pts[cluster_ids[b]]
        dashed_line(p1, p2, fill="#c9c9c9", width_line=2 * scale)
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        draw_text(draw, (int(mx), int(my)), f"{value:.2f}", 10 * scale, fill="#777")

    for a, b in sorted(strong_edges):
        value = float(c_sim[a, b])
        p1 = pts[cluster_ids[a]]
        p2 = pts[cluster_ids[b]]
        width_edge = int(max(1, round((value - 0.86) * 22))) * scale
        draw.line([p1, p2], fill="#6f88c9", width=width_edge)
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        draw_text(draw, (int(mx), int(my)), f"{value:.2f}", 10 * scale, fill="#444")
    max_size = max(cluster_df["hospitals"])
    for idx, row in enumerate(cluster_df.itertuples(index=False)):
        cluster = int(row.cluster)
        px, py = pts[cluster]
        radius = (17 + 22 * math.sqrt(float(row.hospitals) / max_size)) * scale
        color = CLUSTER_COLORS[(cluster - 1) % len(CLUSTER_COLORS)]
        draw.ellipse([px - radius, py - radius, px + radius, py + radius], fill=color, outline="#222", width=2 * scale)
        label = str(cluster)
        bbox = draw.textbbox((0, 0), label, font=font(18 * scale, bold=True))
        draw.text((px - (bbox[2] - bbox[0]) / 2, py - (bbox[3] - bbox[1]) / 2 - 2 * scale), label, font=font(18 * scale, bold=True), fill="white")
    draw_text(draw, (870 * scale, 130 * scale), "Types", 19 * scale, bold=True)
    for idx, row in enumerate(cluster_df.itertuples(index=False)):
        y = 170 * scale + idx * 55 * scale
        color = CLUSTER_COLORS[(int(row.cluster) - 1) % len(CLUSTER_COLORS)]
        draw.ellipse([872 * scale, y - 12 * scale, 896 * scale, y + 12 * scale], fill=color, outline="#222")
        draw_text(draw, (906 * scale, y - 14 * scale), f"Cluster {int(row.cluster)}: {int(row.hospitals)} hospitals", 14 * scale, bold=True)
        draw_text(draw, (906 * scale, y + 6 * scale), str(row.dominant_needs_ru)[:42], 11 * scale, fill="#555")
    legend_y = 470 * scale
    draw.line([(872 * scale, legend_y), (932 * scale, legend_y)], fill="#6f88c9", width=4 * scale)
    draw_text(draw, (942 * scale, legend_y - 10 * scale), "similarity >= 0.88", 12 * scale, fill="#444")
    dashed_line((872 * scale, legend_y + 36 * scale), (932 * scale, legend_y + 36 * scale), fill="#c9c9c9", width_line=2 * scale)
    draw_text(draw, (942 * scale, legend_y + 26 * scale), "nearest reserve", 12 * scale, fill="#444")
    img = img.resize((width, height), Image.Resampling.LANCZOS)
    img.save(path)


def save_closure_robustness(closure: pd.DataFrame, path: Path) -> None:
    plot = closure.sort_values("coverage").head(15).copy()
    scale = 2
    width, height = 1280, 720
    img = Image.new("RGB", (width * scale, height * scale), "white")
    draw = ImageDraw.Draw(img)
    draw_text(draw, (34 * scale, 26 * scale), "Network robustness under hospital closure", 28 * scale, bold=True)
    draw_text(draw, (34 * scale, 66 * scale), "Shows the most vulnerable closures: share of load covered by profile-based substitutes", 16 * scale, fill="#555")
    left, top, bar_w, row_h = 410 * scale, 125 * scale, 650 * scale, 36 * scale
    for tick in np.linspace(0, 1, 6):
        x0 = left + tick * bar_w
        draw.line([(x0, top - 24 * scale), (x0, top + row_h * len(plot))], fill="#e4e4e4")
        draw_text(draw, (int(x0 - 12 * scale), top + row_h * len(plot) + 16 * scale), f"{tick:.1f}", 13 * scale, fill="#555")
    for idx, row in enumerate(plot.itertuples(index=False)):
        y = top + idx * row_h
        draw_text(draw, (34 * scale, y + 5 * scale), f"H{int(row.closed_hospital)} / cluster {int(row.cluster)} / load {int(row.closed_load)}", 13 * scale)
        value = float(row.coverage)
        color = "#2d6cdf" if value >= 0.8 else "#c74b50"
        draw.rounded_rectangle([left, y + 6 * scale, left + value * bar_w, y + 25 * scale], radius=4 * scale, fill=color)
        draw_text(draw, (int(left + value * bar_w + 10 * scale), y + 2 * scale), f"{value:.2f}", 13 * scale, bold=True)
    draw_text(draw, (left + 220 * scale, top + row_h * len(plot) + 52 * scale), "Share of covered load", 16 * scale)
    img = img.resize((width, height), Image.Resampling.LANCZOS)
    img.save(path)


def save_load_balancing(balance: pd.DataFrame, path: Path) -> None:
    scale = 2
    width, height = 1180, 680
    img = Image.new("RGB", (width * scale, height * scale), "white")
    draw = ImageDraw.Draw(img)
    draw_text(draw, (34 * scale, 26 * scale), "Overload reduction through nearby substitutes", 28 * scale, bold=True)
    draw_text(draw, (34 * scale, 66 * scale), "Simulated load surge: reallocation of excess to profile-similar hospitals", 16 * scale, fill="#555")
    left, top, plot_w, plot_h = 105 * scale, 130 * scale, 820 * scale, 390 * scale
    thresholds = balance["similarity_threshold"].to_numpy(float)
    reduction = balance["overload_reduction_share"].to_numpy(float)
    residual = balance["residual_overload"].to_numpy(float)
    x_labels = ["no\nthreshold", "0.80", "0.85", "0.90", "0.95"]
    draw.rectangle([left, top, left + plot_w, top + plot_h], fill="#fbfbfb", outline="#cccccc", width=2 * scale)
    for tick in np.linspace(0, 1, 6):
        y = top + plot_h - tick * plot_h
        draw.line([(left, y), (left + plot_w, y)], fill="#e4e4e4")
        draw_text(draw, (45 * scale, int(y - 8 * scale)), f"{tick:.1f}", 13 * scale, fill="#555")
    draw_text(draw, (34 * scale, 105 * scale), "Share of overload\neliminated", 13 * scale, fill="#555")
    step = plot_w / len(thresholds)
    bar_w = step * 0.50
    for idx, value in enumerate(reduction):
        x = left + idx * step + step / 2 - bar_w / 2
        h = value * plot_h
        color = "#208b68" if value >= 0.8 else "#d28a1e"
        draw.rounded_rectangle([x, top + plot_h - h, x + bar_w, top + plot_h], radius=5 * scale, fill=color)
        draw_text(draw, (int(x + 6 * scale), int(top + plot_h - h - 24 * scale)), f"{value:.2f}", 13 * scale, bold=True)
        draw_text(draw, (int(x + 8 * scale), top + plot_h + 18 * scale), x_labels[idx], 13 * scale)
    draw_text(draw, (left + 245 * scale, top + plot_h + 68 * scale), "Profile similarity threshold for reallocation", 16 * scale)
    draw_text(draw, (940 * scale, 145 * scale), f"Baseline overload:\n{float(balance.iloc[0]['overload_before']):.0f} equiv. patients", 15 * scale, fill="#333", bold=True)
    draw_text(draw, (940 * scale, 225 * scale), "The higher the threshold,\nthe stricter the substitution\nrequirement.", 14 * scale, fill="#555")
    img = img.resize((width, height), Image.Resampling.LANCZOS)
    img.save(path)


def save_alpha_sensitivity(alpha_df: pd.DataFrame, path: Path) -> None:
    scale = 2
    width, height = 1280, 760
    img = Image.new("RGB", (width * scale, height * scale), "white")
    draw = ImageDraw.Draw(img)
    draw_text(draw, (34 * scale, 26 * scale), "Sensitivity of results to clinical profile weight α", 27 * scale, bold=True)
    draw_text(draw, (34 * scale, 66 * scale), "α=0 -- unit types only; α=1 -- clinical needs only", 16 * scale, fill="#555")
    left, top, plot_w, plot_h = 105 * scale, 130 * scale, 880 * scale, 450 * scale
    draw.rectangle([left, top, left + plot_w, top + plot_h], fill="#fbfbfb", outline="#cccccc", width=2 * scale)
    for tick in np.linspace(0, 1, 6):
        y = top + plot_h - tick * plot_h
        x = left + tick * plot_w
        draw.line([(left, y), (left + plot_w, y)], fill="#e8e8e8")
        draw.line([(x, top), (x, top + plot_h)], fill="#eeeeee")
        draw_text(draw, (45 * scale, int(y - 8 * scale)), f"{tick:.1f}", 13 * scale, fill="#555")
        draw_text(draw, (int(x - 12 * scale), top + plot_h + 16 * scale), f"{tick:.1f}", 13 * scale, fill="#555")

    series = [
        ("Silhouette", "operational_silhouette", "#2d6cdf"),
        ("Closure coverage", "mean_closure_coverage", "#208b68"),
        ("Overload reduction", "overload_reduction_share_at_theta", "#d28a1e"),
    ]
    xs = alpha_df["alpha"].to_numpy(float)
    for label, col, color in series:
        values = alpha_df[col].to_numpy(float)
        pts = []
        for x_val, y_val in zip(xs, values):
            px = left + x_val * plot_w
            py = top + plot_h - np.clip(y_val, 0, 1) * plot_h
            pts.append((px, py))
        draw.line(pts, fill=color, width=4 * scale)
        for px, py in pts[::2]:
            draw.ellipse([px - 3 * scale, py - 3 * scale, px + 3 * scale, py + 3 * scale], fill=color)
    selected_x = left + PROFILE_ALPHA * plot_w
    draw.line([(selected_x, top), (selected_x, top + plot_h)], fill="#111", width=2 * scale)
    draw_text(draw, (int(selected_x + 8 * scale), top + 14 * scale), f"α={PROFILE_ALPHA:.2f}", 14 * scale, bold=True)
    draw_text(draw, (left + 365 * scale, top + plot_h + 55 * scale), "Clinical profile weight α", 16 * scale)
    draw_text(draw, (28 * scale, top + 165 * scale), "Metric\nvalue", 13 * scale, fill="#555")

    legend_x, legend_y = 1010 * scale, 150 * scale
    draw_text(draw, (legend_x, legend_y - 34 * scale), "Metrics", 18 * scale, bold=True)
    for idx, (label, _, color) in enumerate(series):
        y = legend_y + idx * 36 * scale
        draw.line([(legend_x, y), (legend_x + 42 * scale, y)], fill=color, width=4 * scale)
        draw_text(draw, (legend_x + 54 * scale, y - 10 * scale), label, 13 * scale)
    selected = alpha_df[alpha_df["selected_alpha"] == 1].iloc[0]
    draw_text(
        draw,
        (1010 * scale, 310 * scale),
        f"At α={PROFILE_ALPHA:.2f}:\nsilhouette {selected['operational_silhouette']:.3f}\nclosure {selected['mean_closure_coverage']:.3f}\noverload {selected['overload_reduction_share_at_theta']:.3f}",
        13 * scale,
        fill="#444",
    )
    draw_text(draw, (1010 * scale, 460 * scale), "The value is not a\nsingle-metric optimum;\nit is a clinically\noriented working\nbalance.", 13 * scale, fill="#555")
    img = img.resize((width, height), Image.Resampling.LANCZOS)
    img.save(path)


def save_theta_sensitivity(theta_df: pd.DataFrame, path: Path) -> None:
    scale = 2
    width, height = 1280, 720
    img = Image.new("RGB", (width * scale, height * scale), "white")
    draw = ImageDraw.Draw(img)
    draw_text(draw, (34 * scale, 26 * scale), "Sensitivity to the profile similarity threshold θ", 28 * scale, bold=True)
    draw_text(draw, (34 * scale, 66 * scale), "The higher θ is, the stricter the hospital substitutability condition", 16 * scale, fill="#555")
    left, top, plot_w, plot_h = 105 * scale, 130 * scale, 850 * scale, 420 * scale
    draw.rectangle([left, top, left + plot_w, top + plot_h], fill="#fbfbfb", outline="#cccccc", width=2 * scale)
    for tick in np.linspace(0, 1, 6):
        y = top + plot_h - tick * plot_h
        draw.line([(left, y), (left + plot_w, y)], fill="#e8e8e8")
        draw_text(draw, (45 * scale, int(y - 8 * scale)), f"{tick:.1f}", 13 * scale, fill="#555")
    theta = theta_df["theta"].to_numpy(float)
    theta_min, theta_max = float(theta.min()), float(theta.max())
    for tick in theta:
        x = left + (tick - theta_min) / max(theta_max - theta_min, 1e-12) * plot_w
        draw.line([(x, top), (x, top + plot_h)], fill="#eeeeee")
        label = "0" if tick == 0 else f"{tick:.2f}"
        draw_text(draw, (int(x - 14 * scale), top + plot_h + 16 * scale), label, 12 * scale, fill="#555")

    series = [
        ("Closure coverage", "mean_closure_coverage", "#2d6cdf"),
        ("Overload reduction", "overload_reduction_share", "#208b68"),
        ("Mean substitute similarity", "mean_move_similarity", "#d28a1e"),
    ]
    for label, col, color in series:
        pts = []
        for x_val, y_val in zip(theta, theta_df[col].to_numpy(float)):
            px = left + (x_val - theta_min) / max(theta_max - theta_min, 1e-12) * plot_w
            py = top + plot_h - np.clip(y_val, 0, 1) * plot_h
            pts.append((px, py))
        draw.line(pts, fill=color, width=4 * scale)
        for px, py in pts:
            draw.ellipse([px - 4 * scale, py - 4 * scale, px + 4 * scale, py + 4 * scale], fill=color)
    selected_x = left + (SUBSTITUTION_THETA - theta_min) / max(theta_max - theta_min, 1e-12) * plot_w
    draw.line([(selected_x, top), (selected_x, top + plot_h)], fill="#111", width=2 * scale)
    draw_text(draw, (int(selected_x + 8 * scale), top + 14 * scale), f"θ={SUBSTITUTION_THETA:.2f}", 14 * scale, bold=True)
    draw_text(draw, (left + 345 * scale, top + plot_h + 55 * scale), "Profile similarity threshold θ", 16 * scale)
    draw_text(draw, (28 * scale, top + 155 * scale), "Metric\nvalue", 13 * scale, fill="#555")

    legend_x, legend_y = 985 * scale, 150 * scale
    draw_text(draw, (legend_x, legend_y - 34 * scale), "Metrics", 18 * scale, bold=True)
    for idx, (label, _, color) in enumerate(series):
        y = legend_y + idx * 38 * scale
        draw.line([(legend_x, y), (legend_x + 42 * scale, y)], fill=color, width=4 * scale)
        draw_text(draw, (legend_x + 54 * scale, y - 10 * scale), label, 13 * scale)
    selected = theta_df[theta_df["selected_theta"] == 1].iloc[0]
    draw_text(
        draw,
        (985 * scale, 315 * scale),
        f"At θ={SUBSTITUTION_THETA:.2f}:\nclosure {selected['mean_closure_coverage']:.3f}\noverload {selected['overload_reduction_share']:.3f}\nsimilarity {selected['mean_move_similarity']:.3f}",
        13 * scale,
        fill="#444",
    )
    draw_text(draw, (985 * scale, 455 * scale), "Beyond 0.90,\nstrictness increases,\nbut the covered\noverload drops\nnoticeably.", 13 * scale, fill="#555")
    img = img.resize((width, height), Image.Resampling.LANCZOS)
    img.save(path)


def save_patient_fit_validation(metrics: pd.DataFrame, path: Path) -> None:
    row = metrics.iloc[0]
    values = [
        ("Random\nhospital", float(row["random_mean_suitability"]), "#9aa4b2"),
        ("Actual\nhospital", float(row["actual_mean_suitability"]), "#2d6cdf"),
        ("Best within\ncluster", float(row["best_same_cluster_mean_suitability"]), "#208b68"),
        ("Best allowed\nsubstitute", float(row["best_substitute_mean_suitability"]), "#d28a1e"),
        ("Best of all\nhospitals", float(row["best_possible_mean_suitability"]), "#7b59c0"),
    ]
    scale = 2
    width, height = 1280, 720
    img = Image.new("RGB", (width * scale, height * scale), "white")
    draw = ImageDraw.Draw(img)
    draw_text(draw, (34 * scale, 26 * scale), "Patient-hospital fit validation", 28 * scale, bold=True)
    draw_text(draw, (34 * scale, 66 * scale), "Suitability score a_ij: clinical match + unit-type match", 16 * scale, fill="#555")
    left, top, plot_w, plot_h = 105 * scale, 130 * scale, 840 * scale, 390 * scale
    draw.rectangle([left, top, left + plot_w, top + plot_h], fill="#fbfbfb", outline="#cccccc", width=2 * scale)
    for tick in np.linspace(0, 1, 6):
        y = top + plot_h - tick * plot_h
        draw.line([(left, y), (left + plot_w, y)], fill="#e8e8e8")
        draw_text(draw, (45 * scale, int(y - 8 * scale)), f"{tick:.1f}", 13 * scale, fill="#555")
    step = plot_w / len(values)
    bar_w = step * 0.56
    for idx, (label, value, color) in enumerate(values):
        x = left + idx * step + step / 2 - bar_w / 2
        h = np.clip(value, 0, 1) * plot_h
        draw.rounded_rectangle([x, top + plot_h - h, x + bar_w, top + plot_h], radius=5 * scale, fill=color)
        draw_text(draw, (int(x + 8 * scale), int(top + plot_h - h - 24 * scale)), f"{value:.3f}", 13 * scale, bold=True)
        for line_idx, line in enumerate(label.split("\n")):
            draw_text(draw, (int(x - 4 * scale), top + plot_h + (18 + 18 * line_idx) * scale), line, 12 * scale)
    draw_text(draw, (left + 290 * scale, top + plot_h + 70 * scale), "Type of compared hospital", 16 * scale)
    draw_text(draw, (28 * scale, top + 155 * scale), "Mean\nsuitability", 13 * scale, fill="#555")
    draw_text(
        draw,
        (980 * scale, 145 * scale),
        f"Test patients:\n{int(row['test_patients'])}\n\nActual above\nrandom:\n{row['actual_above_random_share']:.3f}\n\nActual top-10:\n{row['actual_top10_share']:.3f}\n\nSubstitute available at θ=0.90:\n{row['substitute_available_share']:.3f}",
        14 * scale,
        fill="#444",
    )
    draw_text(draw, (980 * scale, 430 * scale), "This does not prove\nroute optimality,\nbut verifies that\nobserved and\nsubstitute hospitals are\nprofile-compatible.", 13 * scale, fill="#555")
    img = img.resize((width, height), Image.Resampling.LANCZOS)
    img.save(path)


def write_article_summary(
    macro_k: int,
    operational_k: int,
    selection: pd.DataFrame,
    cluster_df: pd.DataFrame,
    interpretation: pd.DataFrame,
    design: pd.DataFrame,
    components: pd.DataFrame,
    alpha_df: pd.DataFrame,
    theta_df: pd.DataFrame,
    fit_metrics: pd.DataFrame,
    overlap: pd.DataFrame,
    closure: pd.DataFrame,
    balance: pd.DataFrame,
) -> None:
    macro_row = selection[selection["k"] == macro_k].iloc[0]
    operational_row = selection[selection["k"] == operational_k].iloc[0]
    best_overlap = overlap.head(5)
    closure_mean = closure["coverage"].mean()
    closure_bad = (closure["coverage"] < 0.8).mean()
    bal90 = balance[balance["similarity_threshold"] == 0.90].iloc[0]
    alpha_selected = alpha_df[alpha_df["selected_alpha"] == 1].iloc[0]
    theta_selected = theta_df[theta_df["selected_theta"] == 1].iloc[0]
    theta_strict = theta_df[theta_df["theta"] == 0.95].iloc[0]
    fit = fit_metrics.iloc[0]
    text = f"""# Support material for article 2: profile structure of the hospital network

## Problem statement

We consider a set of hospitals J = (j_1,...,j_m). For each hospital j the following are known: historical frequencies of clinical needs of admitted patients, the distribution of intensive care unit types, observed load L_j, effective capacity C_j, and free capacity A_j.

The goal is to build:

1. a partition of hospitals into profile types C_1,...,C_K;
2. a profile-similarity matrix rho_jk between hospitals;
3. a substitution graph G_theta=(J,E_theta);
4. an estimate of network robustness under hospital closure;
5. an estimate of the potential to reduce overload through reallocation between similar hospitals.

In this article the object of management is not an individual patient, but the hospital network as an organizational system.

## Method

For hospital j a profile vector is built

```text
v_j = normalize([sqrt(alpha) * r_j^need, sqrt(1-alpha) * r_j^unit]).
```

Here r_j^need is the vector of smoothed frequencies of patient clinical needs, r_j^unit is the distribution of intensive care unit types, alpha=0.72 is the weight share of the clinical profile. Components of r_j^need may overlap, because a single patient can have several clinical features; the general component is assigned only when there is no specialized match. Load, capacity, and free capacity are not mixed into profile similarity; they are used at the second stage, when checking robustness and reallocating load.

Similarity between two hospitals is defined by cosine similarity:

```text
rho_jk = <v_j, v_k> / (||v_j|| * ||v_k||).
```

Hospital clusters are found as a solution to the within-cluster heterogeneity minimization problem:

```text
min sum_r sum_{{j in C_r}} ||v_j - mu_r||^2.
```

The number of clusters is estimated by the silhouette criterion. An operational clustering level is additionally identified: it may have a lower silhouette, but must be suitable for managerial substitution and overload analysis.

The substitution graph is defined as G_theta=(J,E_theta), where edge (j,k) is included when rho_jk >= theta, and for visual analysis the nearest reserve type for each cluster is additionally shown.

The optimization stage is formulated as a flow problem. When hospital j is closed, the maximum share of its load that can be covered by profile-close substitutes is estimated:

```text
B_j(theta) = max sum_k z_jk / L_j
0 <= z_jk <= A_k,
z_jk = 0, if rho_jk < theta,
sum_k z_jk <= L_j.
```

Here L_j is the load of the closed hospital, A_k is the free capacity of the substitute hospital. The smaller B_j(theta) is, the more structurally critical the hospital is.

For the overload scenario, an analogous maximum-flow problem is solved:

```text
max sum_j sum_k z_jk
sum_k z_jk <= e_j,
sum_j z_jk <= A_k,
z_jk = 0, if rho_jk < theta.
```

Here e_j=max(0,L'_j-C_j) is the excess load after simulating a surge in admissions. This checks what share of the overload can be eliminated without resorting to profile-distant substitutes.

## Experimental design

The experiment consists of four parts:

1. building hospital profile vectors from data derived from eICU;
2. choosing the number of clusters and interpreting the identified types;
3. building the substitution graph from cosine profile similarity;
4. stress tests: closing each hospital and a local load surge.

Validation is performed not through classification accuracy, but through system-level metrics: clustering silhouette, load coverage under closure, share of critical closures, share of eliminated overload, mean similarity of substitutes used, and share of moves within the same cluster.

A retrospective patient-hospital fit check is additionally performed. For patient i and hospital j a suitability score is computed

```text
a_ij = α * clinical_match_ij + (1-α) * unit_match_ij.
```

Here clinical_match_ij shows how well the patient's clinical needs are represented in the hospital's profile, and unit_match_ij shows the match of the unit type. This check does not prove that the actual routing was optimal: eICU records the fact of treatment, not all alternative EMS decisions. However, it does allow checking whether the actual and substitute hospitals are profile-compatible with the patients.

## Experiment reproducibility table

| Parameter | Value | How to reproduce |
|---|---|---|
"""
    for row in design.itertuples(index=False):
        parameter = str(row.parameter_ru).replace("|", "/")
        value = str(row.value_ru).replace("|", "/")
        note = str(row.reproduction_note_ru).replace("|", "/")
        text += f"| {parameter} | {value} | {note} |\n"
    text += """
## Breakdown of profile vectors r_j^need and r_j^unit

| Vector | Component | Source | Patient-level rule | Hospital-level aggregation |
|---|---|---|---|---|
"""
    for row in components.itertuples(index=False):
        vector = str(row.vector_ru).replace("|", "/")
        component = f"{row.component_order}. {row.component_name_ru} ({row.component_name})".replace("|", "/")
        source = str(row.source_columns).replace("|", "/")
        rule = str(row.patient_level_rule_ru).replace("|", "/")
        aggregation = str(row.hospital_level_aggregation_ru).replace("|", "/")
        text += f"| {vector} | {component} | {source} | {rule} | {aggregation} |\n"
    text += f"""
## Justification of the α and θ parameters

The parameter α sets the relative role of the clinical profile in the hospital's profile vector. The value α=0.72 is not treated as a universal mathematical optimum on a single metric. It is a working clinico-organizational setting in which patient clinical needs dominate over unit structure, but ICU-type information is not excluded entirely. To check robustness, a sensitivity analysis over α from 0 to 1 with step 0.04 was performed.

At α={PROFILE_ALPHA:.2f} the following values were obtained:

```text
operational silhouette = {alpha_selected['operational_silhouette']:.3f}
mean closure coverage = {alpha_selected['mean_closure_coverage']:.3f}
overload reduction share at θ=0.90 = {alpha_selected['overload_reduction_share_at_theta']:.3f}
```

The threshold θ defines the minimum cosine similarity at which two hospitals are considered valid substitutes. The value θ=0.90 is chosen as a compromise between strictness of profile matching and the ability to actually cover overload. At θ=0.90 a high share of eliminated overload is maintained, while raising it further to 0.95 noticeably reduces the covered volume:

```text
θ=0.90: overload reduction share = {theta_selected['overload_reduction_share']:.3f}, mean substitute similarity = {theta_selected['mean_move_similarity']:.3f}
θ=0.95: overload reduction share = {theta_strict['overload_reduction_share']:.3f}, mean substitute similarity = {theta_strict['mean_move_similarity']:.3f}
```

The corresponding tables and charts are saved in the files `alpha_sensitivity.csv`, `theta_sensitivity.csv`, `figure_ru_alpha_sensitivity.png`, and `figure_ru_theta_sensitivity.png`.

## Patient-hospital fit validation

For the test portion of the sample, the suitability of the actual hospital and of alternative hospitals was computed. Mean values:

```text
random hospital = {fit['random_mean_suitability']:.3f}
actual hospital = {fit['actual_mean_suitability']:.3f}
best hospital within cluster = {fit['best_same_cluster_mean_suitability']:.3f}
best allowed substitute at θ=0.90 = {fit['best_substitute_mean_suitability']:.3f}
best hospital overall = {fit['best_possible_mean_suitability']:.3f}
```

The actual hospital has suitability above the average random candidate for {fit['actual_above_random_share']:.1%} of patients. For {fit['substitute_available_share']:.1%} of patients, at least one valid profile substitute exists at θ=0.90. Therefore the substitution check is not performed only at the level of abstract clusters: for patients it is also assessed how well the substitute hospital preserves clinico-organizational fit.

## Number of hospital types

By the silhouette criterion, the most pronounced macro level has the following number of clusters:

```text
k_macro = {macro_k}, silhouette = {macro_row['silhouette']:.3f}
```

However, two macro types are not enough for the managerial substitution and balancing task. Therefore operational subtypes are additionally identified:

```text
k_operational = {operational_k}, silhouette = {operational_row['silhouette']:.3f}
```

It is the operational subtypes that are used in the substitution graph, the closure stress test, and the load-balancing scenario. The `hospital_clusters.csv` table lists, for each hospital, the operational cluster, the nearest profile substitutes, and organizational metrics.

## Cluster interpretation

| Cluster | Type | Hospitals | Load | Leading profiles | Managerial role |
|---:|---|---:|---:|---|---|
"""
    for row in interpretation.itertuples(index=False):
        text += f"| {int(row.cluster)} | {row.type_ru} | {int(row.hospitals)} | {int(row.observed_load)} | {row.dominant_profiles_ru} | {row.management_role_ru} |\n"
    text += f"""
The bulk of the network is formed by universal and general Med-Surg hospitals. The smaller clusters are not statistical noise: they have a more pronounced specialization or greater free capacity and are therefore important for substitution and balancing tasks.

## Which profiles overlap

The closest pairs of clinical profiles by the distribution of hospital capabilities:

"""
    for row in best_overlap.itertuples(index=False):
        text += f"- {row.profile_a_ru} -- {row.profile_b_ru}: similarity {row.hospital_level_cosine:.3f}\n"
    text += f"""
This can be used as a justification for the substitution graph: if two profiles are frequently expressed at the same hospitals, then routing between them is potentially less risky than transferring to a structurally different profile.

## Hospital substitutability

For each hospital, the nearest profile substitutes were built by cosine similarity of profile vectors. A graph edge means that one hospital can be considered a candidate substitute for another based on the structure of its clinical profile and unit type. Main files:

In the figure, solid edges correspond to strong profile similarity, and dashed edges show the nearest reserve type for a cluster, even if its similarity is below the chosen strict threshold.

- `hospital_substitution_edges.csv` -- top substitutes for each hospital;
- `figure_ru_cluster_graph.png` -- graph of similarity between hospital types;
- `figure_ru_cluster_profile_heatmap.png` -- heatmap of cluster profiles.

## Robustness under hospital closure

In the closure stress test, the load of the closed hospital was reallocated only to profile-close hospitals with a similarity threshold of 0.90 and free capacity. Mean load coverage:

```text
mean coverage = {closure_mean:.3f}
share closures below 0.80 = {closure_bad:.3f}
```

This makes it possible to identify hospitals whose closure is poorly compensated by the network. Such hospitals are structurally critical not because they are the largest, but because they have few close substitutes with available capacity.

## Overload reduction

In the local load-surge scenario, it was checked whether overload can be reduced by reallocating patients between similar hospitals. The overload was not created uniformly across the whole network, but in a group of peak-loaded hospitals within each operational cluster. At a profile similarity threshold of 0.90:

```text
overload reduction share = {bal90['overload_reduction_share']:.3f}
mean move similarity = {bal90['mean_move_similarity']:.3f}
same cluster move share = {bal90['same_cluster_move_share']:.3f}
```

Thus the cluster structure can be used not only to describe hospital types, but also as a management mechanism: the system first looks for a substitute within a close profile, then within a close cluster, and only after that allows a more distant transfer.

## Clustering conclusions

1. The hospital network is heterogeneous: at the macro level 2 large types are identified, but for managerial routing 5 operational types are more informative.
2. Clusters 1 and 2 form the mass universal circuit of the network, taking on most of the load.
3. Clusters 3--5 are smaller in number of hospitals, but important as reserve and specialized elements: it is these that determine how well the system withstands a closure or local overload.
4. Profiles overlap unevenly: the closest pair is trauma and neuro, which confirms the possibility of profile-based substitution between these directions.
5. The substitution graph is not complete: some types have strong connections, while others require the nearest reserve type. This makes it possible to identify structurally critical hospitals.
6. Reallocation within close clusters reduces overload without fully abandoning profile matching: at a threshold of 0.90, {bal90['overload_reduction_share'] * 100:.1f}% of the original overload is eliminated.

## Scientific idea for the article

The proposed method can be formulated as a profile-cluster model of hospital network robustness. Unlike an individual patient-routing model, the object of study here is the organizational system itself: which hospital types exist, which of them are interchangeable, which closures are critical, and how much overload can be reduced through reallocation within close clusters.
"""
    (OUT / "hospital_network_article_support_ru.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    profiles = pd.read_csv(PROFILE_PATH)
    x, feature_labels, _ = profile_matrix(profiles)
    sim = pairwise_cosine(x)
    selection, macro_k, operational_k, labels_raw, centers = choose_clusters(x, sim)
    labels = relabel_by_size(labels_raw)
    profiles_out = profiles.copy()
    profiles_out["cluster"] = labels

    design, components = experiment_reproducibility_tables(profiles)
    cluster_df = cluster_summary(profiles, labels)
    interpretation = cluster_interpretation(cluster_df)
    overlap_hospital, overlap_patient = profile_overlap_tables(profiles)
    edges = substitution_edges(profiles, labels, sim)
    closure, closure_moves = closure_robustness(profiles, labels, sim, threshold=SUBSTITUTION_THETA)
    balance, balance_moves = load_balancing(profiles, labels, sim)
    alpha_df = alpha_sensitivity(profiles)
    theta_df = theta_sensitivity(profiles, labels, sim)
    fit_metrics, fit_examples = patient_hospital_fit_validation(profiles, labels, sim)

    selection.to_csv(OUT / "cluster_selection.csv", index=False)
    alpha_df.to_csv(OUT / "alpha_sensitivity.csv", index=False)
    theta_df.to_csv(OUT / "theta_sensitivity.csv", index=False)
    fit_metrics.to_csv(OUT / "patient_hospital_fit_metrics.csv", index=False, encoding="utf-8-sig")
    fit_examples.to_csv(OUT / "patient_hospital_fit_examples.csv", index=False, encoding="utf-8-sig")
    design.to_csv(OUT / "experiment_reproducibility_ru.csv", index=False, encoding="utf-8-sig")
    components.to_csv(OUT / "profile_vector_components_ru.csv", index=False, encoding="utf-8-sig")
    profiles_out.to_csv(OUT / "hospital_clusters.csv", index=False, encoding="utf-8-sig")
    cluster_df.to_csv(OUT / "cluster_profiles.csv", index=False, encoding="utf-8-sig")
    interpretation.to_csv(OUT / "cluster_interpretation_ru.csv", index=False, encoding="utf-8-sig")
    overlap_hospital.to_csv(OUT / "profile_overlap_hospital_level.csv", index=False, encoding="utf-8-sig")
    overlap_patient.to_csv(OUT / "profile_overlap_patient_level.csv", index=False, encoding="utf-8-sig")
    edges.to_csv(OUT / "hospital_substitution_edges.csv", index=False)
    closure.to_csv(OUT / "closure_robustness.csv", index=False)
    closure_moves.to_csv(OUT / "closure_reallocation_moves.csv", index=False)
    balance.to_csv(OUT / "load_balancing_summary.csv", index=False)
    balance_moves.to_csv(OUT / "load_balancing_moves.csv", index=False)

    save_cluster_selection(selection, macro_k, operational_k, OUT / "figure_ru_cluster_selection.png")
    save_hospital_cluster_scatter(profiles, labels, x, cluster_df, OUT / "figure_ru_hospital_clusters_pca.png")
    save_cluster_interpretation_table(interpretation, OUT / "figure_ru_cluster_interpretation.png")
    save_cluster_heatmap(cluster_df, OUT / "figure_ru_cluster_profile_heatmap.png")
    save_overlap_bar(overlap_hospital, OUT / "figure_ru_profile_overlap.png")
    save_cluster_graph(cluster_df, profiles, labels, x, sim, OUT / "figure_ru_cluster_graph.png")
    save_closure_robustness(closure, OUT / "figure_ru_closure_robustness.png")
    save_load_balancing(balance, OUT / "figure_ru_load_balancing.png")
    save_alpha_sensitivity(alpha_df, OUT / "figure_ru_alpha_sensitivity.png")
    save_theta_sensitivity(theta_df, OUT / "figure_ru_theta_sensitivity.png")
    save_patient_fit_validation(fit_metrics, OUT / "figure_ru_patient_hospital_fit.png")
    write_article_summary(
        macro_k,
        operational_k,
        selection,
        cluster_df,
        interpretation,
        design,
        components,
        alpha_df,
        theta_df,
        fit_metrics,
        overlap_hospital,
        closure,
        balance,
    )

    protocol = {
        "input_profile_path": str(PROFILE_PATH),
        "hospitals": int(len(profiles)),
        "macro_clusters_by_silhouette": int(macro_k),
        "operational_clusters_used": int(operational_k),
        "profile_alpha": PROFILE_ALPHA,
        "substitution_theta": SUBSTITUTION_THETA,
        "profile_features": feature_labels,
        "clustering": "k-means over row-normalized weighted profile vector; macro k selected by silhouette; operational k selected among k>=4 with min cluster size >=8",
        "substitution": "cosine similarity over the same profile vector",
        "reallocation_optimizer": "maximum flow over allowed hospital substitution edges",
        "closure_threshold": 0.90,
        "capacity_model": "effective_capacity = ceil(observed_load * 1.22 + 2)",
        "surge_scenario": "localized surge: hospitals above 70th load percentile inside each cluster receive multiplier 1.85",
        "sensitivity_outputs": ["alpha_sensitivity.csv", "theta_sensitivity.csv"],
        "patient_fit_outputs": ["patient_hospital_fit_metrics.csv", "patient_hospital_fit_examples.csv"],
    }
    (OUT / "hospital_network_protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Macro k={macro_k}; operational k={operational_k}")
    print(cluster_df[["cluster", "hospitals", "observed_load", "dominant_needs_ru"]].to_string(index=False))
    print()
    print("Top profile overlaps:")
    print(overlap_hospital[["profile_a_ru", "profile_b_ru", "hospital_level_cosine"]].head(8).to_string(index=False))
    print()
    print("Load balancing:")
    print(balance.to_string(index=False))
    print(f"Saved outputs to {OUT}")


if __name__ == "__main__":
    main()
