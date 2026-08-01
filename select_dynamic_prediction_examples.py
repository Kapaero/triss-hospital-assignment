from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from dynamic_triss_torch_experiment import (
    BASE_SEED,
    BINS,
    MLPModel,
    TransformerModel,
    build_dataset,
    scale_by_train,
    set_seed,
    train_torch_model,
)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_triss" / "prediction_candidates"
N_CANDIDATES = 10


def zone_id(values: np.ndarray) -> np.ndarray:
    return np.where(values < 0.50, 0, np.where(values < 0.75, 1, 2))


def zone_label(value: float) -> str:
    if value < 0.50:
        return "red"
    if value < 0.75:
        return "yellow"
    return "green"


def candidate_score(y_true: np.ndarray, preds: dict[str, np.ndarray]) -> dict[str, float]:
    z_true = zone_id(y_true)
    changes = int(np.sum(z_true[1:] != z_true[:-1]))
    final_change = int(z_true[-1] != z_true[-2])
    start_end_change = int(z_true[-1] != z_true[0])
    dynamic_range = float(y_true.max() - y_true.min())
    delta = float(y_true[-1] - y_true[0])
    pred_end_match = np.mean([zone_id(pred)[-1] == z_true[-1] for pred in preds.values()])
    pred_trend_match = np.mean([np.sign(pred[-1] - pred[0]) == np.sign(delta) for pred in preds.values()])
    end_mae = float(np.mean([abs(pred[-1] - y_true[-1]) for pred in preds.values()]))
    min_mae = float(np.mean([abs(pred.min() - y_true.min()) for pred in preds.values()]))
    score = (
        3.0 * changes
        + 2.0 * final_change
        + 1.5 * start_end_change
        + 2.5 * dynamic_range
        + 1.0 * pred_end_match
        + 0.7 * pred_trend_match
        - 1.2 * end_mae
        - 0.7 * min_mae
    )
    return {
        "score": float(score),
        "zone_changes": changes,
        "final_zone_change": final_change,
        "start_end_zone_change": start_end_change,
        "dynamic_range": dynamic_range,
        "delta": delta,
        "end_mae": end_mae,
        "min_mae": min_mae,
        "pred_end_zone_match_rate": float(pred_end_match),
        "pred_trend_match_rate": float(pred_trend_match),
    }


def train_candidate_models() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    set_seed(BASE_SEED)
    X, y, red, ids = build_dataset()
    strat = red if len(np.unique(red)) > 1 and min(np.bincount(red.astype(int))) >= 2 else None
    train_idx, test_idx = train_test_split(
        np.arange(len(X)),
        test_size=0.25,
        random_state=BASE_SEED,
        stratify=strat,
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    X_train_s, X_test_s = scale_by_train(X_train, X_test)

    ridge = Ridge(alpha=3.0)
    ridge.fit(X_train_s.reshape(len(X_train_s), -1), y_train)
    preds = {
        "Temporal ridge": np.clip(ridge.predict(X_test_s.reshape(len(X_test_s), -1)), 0, 1),
        "Deep MLP": train_torch_model(MLPModel(X.shape[-1], X.shape[1]), X_train_s, y_train, X_test_s, BASE_SEED + 101),
        "Transformer": train_torch_model(TransformerModel(X.shape[-1]), X_train_s, y_train, X_test_s, BASE_SEED + 202),
    }
    preds = {name: np.clip(pred, 0, 1) for name, pred in preds.items()}
    return ids[test_idx], y_test, test_idx, preds


def plot_candidate(rank: int, stay_id: int, y_true: np.ndarray, preds: dict[str, np.ndarray], stats: dict[str, float]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(BINS))
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.axhspan(0.75, 1.0, color="#dff2e5", zorder=0)
    ax.axhspan(0.50, 0.75, color="#fff1c7", zorder=0)
    ax.axhspan(0.0, 0.50, color="#f8dddd", zorder=0)
    ax.axhline(0.75, color="#94b89c", linewidth=1.2)
    ax.axhline(0.50, color="#d6a0a0", linewidth=1.2)

    ax.plot(x, y_true, marker="o", linewidth=3.2, color="#222222", label="Actual")
    colors = {
        "Temporal ridge": "#2f6f73",
        "Deep MLP": "#b66d3a",
        "Transformer": "#4267b2",
    }
    for name, pred in preds.items():
        ax.plot(x, pred, marker="o", linestyle="--", linewidth=2.3, color=colors[name], label=name)

    title = (
        f"Candidate {rank:02d}: stay {stay_id} | "
        f"actual {zone_label(y_true[0])}->{zone_label(y_true[-1])}, "
        f"Delta={stats['delta']:+.2f}"
    )
    ax.set_title(title, fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(BINS, rotation=18)
    ax.set_xlabel("Time window from ICU admission, hours")
    ax.set_ylabel("P_i(t), estimated probability of favorable outcome")
    ax.set_ylim(0, 1.02)
    ax.grid(True, color="#d5d5d5", linewidth=0.8)
    ax.legend(loc="lower left", frameon=True)
    ax.text(0.02, 0.96, "green > 0.75", transform=ax.transAxes, fontsize=9, color="#4f8f5b")
    ax.text(0.02, 0.90, "yellow 0.50-0.75", transform=ax.transAxes, fontsize=9, color="#a37824")
    ax.text(0.02, 0.84, "red < 0.50", transform=ax.transAxes, fontsize=9, color="#a54e4e")
    fig.tight_layout()
    out = OUT / f"candidate_{rank:02d}_stay_{stay_id}.png"
    fig.savefig(out, dpi=190)
    plt.close(fig)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    ids, y_test, _, preds = train_candidate_models()
    rows = []
    for local_i, stay_id in enumerate(ids):
        pred_i = {name: pred[local_i] for name, pred in preds.items()}
        stats = candidate_score(y_test[local_i], pred_i)
        z = zone_id(y_test[local_i])
        if stats["zone_changes"] == 0 and stats["dynamic_range"] < 0.12:
            continue
        rows.append(
            {
                "stay_id": int(stay_id),
                "local_index": int(local_i),
                "true_start": float(y_test[local_i][0]),
                "true_end": float(y_test[local_i][-1]),
                "true_min": float(y_test[local_i].min()),
                "true_max": float(y_test[local_i].max()),
                "start_zone": zone_label(float(y_test[local_i][0])),
                "end_zone": zone_label(float(y_test[local_i][-1])),
                "zone_path": "-".join(map(str, z.tolist())),
                **stats,
            }
        )
    candidates = pd.DataFrame(rows).sort_values(
        ["final_zone_change", "zone_changes", "score"],
        ascending=[False, False, False],
    )
    selected = candidates.head(N_CANDIDATES).copy()
    out_paths = []
    for rank, row in enumerate(selected.itertuples(index=False), start=1):
        local_i = int(row.local_index)
        pred_i = {name: pred[local_i] for name, pred in preds.items()}
        stats = candidate_score(y_test[local_i], pred_i)
        out_paths.append(plot_candidate(rank, int(row.stay_id), y_test[local_i], pred_i, stats))
    selected["figure"] = [str(path) for path in out_paths]
    selected.to_csv(OUT / "candidate_selection.csv", index=False)
    print(selected[["stay_id", "score", "zone_changes", "final_zone_change", "true_start", "true_end", "true_min", "true_max", "figure"]].to_string(index=False))


if __name__ == "__main__":
    main()
