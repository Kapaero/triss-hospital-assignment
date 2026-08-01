from __future__ import annotations

import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent
INP = ROOT / "results" / "mimic_patient_time_bins.csv"
OUT = ROOT / "results_triss"
BINS = ["0-2", "2-4", "4-6", "6-8", "8-12", "12-18", "18-24"]
FEATURES = ["HR", "SBP", "RR", "GCS", "RTS", "shock_index"]
N_REPEATS = 12
EPOCHS = 160
BASE_SEED = 20260512


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def build_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(INP)
    global_median = df[FEATURES].median()
    sequences, targets, red, ids = [], [], [], []
    for stay_id, group in df.groupby("stay_id"):
        g = group.set_index("time_bin").reindex(BINS)
        x = g[FEATURES].ffill().bfill().fillna(global_median).to_numpy(float)
        y = g["survival_6h"].ffill().bfill()
        if y.isna().all():
            continue
        y = y.fillna(float(y.median())).to_numpy(float)
        sequences.append(x)
        targets.append(y)
        red.append(float((y < 0.50).any()))
        ids.append(stay_id)
    return np.asarray(sequences), np.asarray(targets), np.asarray(red), np.asarray(ids)


def scale_by_train(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    train_shape = X_train.shape
    test_shape = X_test.shape
    X_train_2d = X_train.reshape(-1, X_train.shape[-1])
    X_test_2d = X_test.reshape(-1, X_test.shape[-1])
    X_train_scaled = scaler.fit_transform(X_train_2d).reshape(train_shape)
    X_test_scaled = scaler.transform(X_test_2d).reshape(test_shape)
    return X_train_scaled.astype(np.float32), X_test_scaled.astype(np.float32)


def shape_corr(y_true: np.ndarray, pred: np.ndarray) -> float:
    vals = []
    for a, b in zip(y_true, pred):
        if np.std(a) < 1e-8 or np.std(b) < 1e-8:
            continue
        vals.append(np.corrcoef(a, b)[0, 1])
    return float(np.nanmean(vals)) if vals else float("nan")


def red_auc(y_true_curve: np.ndarray, pred_curve: np.ndarray) -> float:
    y_red = (y_true_curve.min(axis=1) < 0.50).astype(int)
    score = 1 - pred_curve.min(axis=1)
    if len(np.unique(y_red)) < 2:
        return float("nan")
    return float(roc_auc_score(y_red, score))


def eval_pred(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    pred = np.clip(pred, 0, 1)
    return {
        "mse": float(mean_squared_error(y_true.ravel(), pred.ravel())),
        "mae": float(mean_absolute_error(y_true.ravel(), pred.ravel())),
        "shape_corr": shape_corr(y_true, pred),
        "red_auc": red_auc(y_true, pred),
    }


class LSTMModel(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.head(out).squeeze(-1)


class GRUModel(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.head(out).squeeze(-1)


class TransformerModel(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4, layers: int = 2) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, len(BINS), d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.05,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Sequential(nn.Linear(d_model, 32), nn.GELU(), nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x) + self.pos
        z = self.encoder(z)
        return self.head(z).squeeze(-1)


class MLPModel(nn.Module):
    def __init__(self, input_dim: int, seq_len: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim * seq_len, 96),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, seq_len),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_torch_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    seed: int,
) -> np.ndarray:
    set_seed(seed)
    model.train()
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    loader = DataLoader(train_ds, batch_size=24, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    mse = nn.MSELoss()
    for _ in range(EPOCHS):
        for xb, yb in loader:
            pred = model(xb)
            loss = mse(pred, yb)
            # Encourage trajectory-shape agreement.
            loss = loss + 0.25 * mse(pred[:, 1:] - pred[:, :-1], yb[:, 1:] - yb[:, :-1])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X_test, dtype=torch.float32)).numpy()


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    OUT.mkdir(parents=True, exist_ok=True)
    X, y, red, ids = build_dataset()
    raw_rows = []
    pred_examples = []

    for rep in range(N_REPEATS):
        seed = BASE_SEED + rep
        set_seed(seed)
        strat = red if len(np.unique(red)) > 1 and min(np.bincount(red.astype(int))) >= 2 else None
        train_idx, test_idx = train_test_split(np.arange(len(X)), test_size=0.25, random_state=seed, stratify=strat)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        X_train_s, X_test_s = scale_by_train(X_train, X_test)

        mean_pred = np.repeat(y_train.mean(axis=0, keepdims=True), len(y_test), axis=0)
        raw_rows.append({"repeat": rep, "seed": seed, "model": "Mean trajectory", **eval_pred(y_test, mean_pred)})

        ridge = Ridge(alpha=3.0)
        ridge.fit(X_train_s.reshape(len(X_train_s), -1), y_train)
        ridge_pred = np.clip(ridge.predict(X_test_s.reshape(len(X_test_s), -1)), 0, 1)
        raw_rows.append({"repeat": rep, "seed": seed, "model": "Temporal ridge", **eval_pred(y_test, ridge_pred)})

        for name, cls in [
            ("Deep MLP", lambda: MLPModel(X.shape[-1], X.shape[1])),
            ("LSTM", lambda: LSTMModel(X.shape[-1])),
            ("GRU", lambda: GRUModel(X.shape[-1])),
            ("Transformer", lambda: TransformerModel(X.shape[-1])),
        ]:
            pred = train_torch_model(cls(), X_train_s, y_train, X_test_s, seed + hash(name) % 1000)
            raw_rows.append({"repeat": rep, "seed": seed, "model": name, **eval_pred(y_test, pred)})
            if rep == 0 and name in {"Transformer", "LSTM", "GRU"}:
                for local_i in range(min(5, len(y_test))):
                    pred_examples.append(
                        {
                            "model": name,
                            "stay_id": ids[test_idx[local_i]],
                            **{f"true_t{k}": y_test[local_i, k] for k in range(y.shape[1])},
                            **{f"pred_t{k}": pred[local_i, k] for k in range(y.shape[1])},
                        }
                    )

    raw = pd.DataFrame(raw_rows)
    summary = raw.groupby("model")[["mse", "mae", "shape_corr", "red_auc"]].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(c).strip("_") for c in summary.columns.to_flat_index()]
    raw.to_csv(OUT / "dynamic_triss_torch_results_raw.csv", index=False)
    summary.to_csv(OUT / "dynamic_triss_torch_results_summary.csv", index=False)
    pd.DataFrame(pred_examples).to_csv(OUT / "dynamic_triss_torch_prediction_examples.csv", index=False)
    return raw, summary


def plot_summary(summary: pd.DataFrame) -> None:
    order = ["Mean trajectory", "Temporal ridge", "Deep MLP", "LSTM", "GRU", "Transformer"]
    plot = summary.set_index("model").loc[order].reset_index()
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(data=plot, x="model", y="mse_mean", ax=ax, color="#2f6f73")
    ax.errorbar(np.arange(len(plot)), plot["mse_mean"], yerr=plot["mse_std"], fmt="none", c="#222", capsize=3)
    ax.set_title("Dynamic TRISS trajectory prediction: repeated split MSE")
    ax.set_xlabel("")
    ax.set_ylabel("MSE")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(OUT / "dynamic_triss_torch_mse.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(data=plot, x="model", y="shape_corr_mean", ax=ax, color="#5c84a8")
    ax.errorbar(np.arange(len(plot)), plot["shape_corr_mean"], yerr=plot["shape_corr_std"], fmt="none", c="#222", capsize=3)
    ax.set_title("Dynamic TRISS trajectory prediction: shape correlation")
    ax.set_xlabel("")
    ax.set_ylabel("Shape correlation")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(OUT / "dynamic_triss_torch_shape_corr.png", dpi=180)
    plt.close(fig)


def plot_examples() -> None:
    ex = pd.read_csv(OUT / "dynamic_triss_torch_prediction_examples.csv")
    if ex.empty:
        return
    time = np.arange(len(BINS))
    for stay_id, group in ex.groupby("stay_id"):
        fig, ax = plt.subplots(figsize=(8, 4.8))
        first = group.iloc[0]
        true = [first[f"true_t{k}"] for k in range(len(BINS))]
        ax.plot(time, true, marker="o", linewidth=3, label="True", color="#222")
        for _, row in group.iterrows():
            pred = [row[f"pred_t{k}"] for k in range(len(BINS))]
            ax.plot(time, pred, marker="o", linestyle="--", label=row["model"])
        ax.axhspan(0.75, 1.0, color="#dff2e5", zorder=0)
        ax.axhspan(0.50, 0.75, color="#fff1c7", zorder=0)
        ax.axhspan(0.0, 0.50, color="#f8dddd", zorder=0)
        ax.set_title(f"Trajectory prediction example: stay {stay_id}")
        ax.set_xticks(time)
        ax.set_xticklabels(BINS, rotation=20)
        ax.set_ylabel("Survival estimate")
        ax.set_ylim(0, 1.02)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / f"dynamic_triss_prediction_example_{stay_id}.png", dpi=180)
        plt.close(fig)
        break


def main() -> None:
    raw, summary = run()
    plot_summary(summary)
    plot_examples()
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
