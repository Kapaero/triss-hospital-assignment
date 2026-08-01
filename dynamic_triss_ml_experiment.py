from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_triss"
INP = ROOT / "results" / "mimic_patient_time_bins.csv"
BINS = ["0-2", "2-4", "4-6", "6-8", "8-12", "12-18", "18-24"]
FEATURES = ["HR", "SBP", "RR", "GCS", "RTS", "shock_index"]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)


def auc_score(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = y_true.astype(int)
    pos = score[y_true == 1]
    neg = score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def build_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(INP)
    med = df[FEATURES].median()
    X, y_curve, y_red = [], [], []
    for _, group in df.groupby("stay_id"):
        g = group.set_index("time_bin").reindex(BINS)
        feats = g[FEATURES].ffill().bfill().fillna(med)
        target = g["survival_6h"].ffill().bfill().fillna(g["survival_6h"].median())
        X.append(feats.to_numpy(float))
        y_curve.append(target.to_numpy(float))
        y_red.append(float((target < 0.50).any()))
    return np.asarray(X), np.asarray(y_curve), np.asarray(y_red)


def standardize(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    Xb = np.column_stack([np.ones(len(X)), X])
    eye = np.eye(Xb.shape[1])
    eye[0, 0] = 0
    return np.linalg.solve(Xb.T @ Xb + alpha * eye, Xb.T @ y)


def ridge_predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    Xb = np.column_stack([np.ones(len(X)), X])
    return np.clip(Xb @ w, 0, 1)


class MLPRegressor:
    def __init__(self, in_dim: int, out_dim: int, rng: np.random.Generator) -> None:
        dims = [in_dim, 96, 48, out_dim]
        self.w = []
        self.b = []
        for a, b in zip(dims[:-1], dims[1:]):
            self.w.append(rng.normal(0, np.sqrt(2 / a), size=(a, b)))
            self.b.append(np.zeros((1, b)))
        self.mw = [np.zeros_like(w) for w in self.w]
        self.vw = [np.zeros_like(w) for w in self.w]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.step = 0

    def forward(self, X: np.ndarray):
        acts = [X]
        zs = []
        a = X
        for i, (w, b) in enumerate(zip(self.w, self.b)):
            z = a @ w + b
            zs.append(z)
            a = sigmoid(z) if i == len(self.w) - 1 else relu(z)
            acts.append(a)
        return acts, zs

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)[0][-1]

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 900, lr: float = 0.002) -> None:
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        n = len(X)
        for _ in range(epochs):
            acts, zs = self.forward(X)
            pred = acts[-1]
            grad = 2 * (pred - y) / n
            grad = grad * pred * (1 - pred)
            gw, gb = [], []
            for layer in reversed(range(len(self.w))):
                gw.insert(0, acts[layer].T @ grad)
                gb.insert(0, grad.sum(axis=0, keepdims=True))
                if layer > 0:
                    grad = grad @ self.w[layer].T
                    grad = grad * (zs[layer - 1] > 0)
            self.step += 1
            for i in range(len(self.w)):
                self.mw[i] = beta1 * self.mw[i] + (1 - beta1) * gw[i]
                self.vw[i] = beta2 * self.vw[i] + (1 - beta2) * (gw[i] ** 2)
                self.mb[i] = beta1 * self.mb[i] + (1 - beta1) * gb[i]
                self.vb[i] = beta2 * self.vb[i] + (1 - beta2) * (gb[i] ** 2)
                mw_hat = self.mw[i] / (1 - beta1**self.step)
                vw_hat = self.vw[i] / (1 - beta2**self.step)
                mb_hat = self.mb[i] / (1 - beta1**self.step)
                vb_hat = self.vb[i] / (1 - beta2**self.step)
                self.w[i] -= lr * mw_hat / (np.sqrt(vw_hat) + eps)
                self.b[i] -= lr * mb_hat / (np.sqrt(vb_hat) + eps)


def metrics(y_true: np.ndarray, pred: np.ndarray, red_true: np.ndarray) -> dict[str, float]:
    mse = float(np.mean((y_true - pred) ** 2))
    mae = float(np.mean(np.abs(y_true - pred)))
    corr = []
    for a, b in zip(y_true, pred):
        if np.std(a) < 1e-8 or np.std(b) < 1e-8:
            continue
        corr.append(np.corrcoef(a, b)[0, 1])
    red_score = 1 - pred.min(axis=1)
    return {
        "mse": mse,
        "mae": mae,
        "shape_corr": float(np.mean(corr)) if corr else float("nan"),
        "red_auc": auc_score(red_true, red_score),
    }


def run(repeats: int = 30, seed: int = 123) -> pd.DataFrame:
    X, y, red = build_dataset()
    rng = np.random.default_rng(seed)
    rows = []
    for rep in range(repeats):
        idx = rng.permutation(len(X))
        split = int(0.75 * len(idx))
        tr, te = idx[:split], idx[split:]
        Xtr, Xte = standardize(X[tr], X[te])
        ytr, yte = y[tr], y[te]
        red_te = red[te]

        last_pred = np.repeat(ytr.mean(axis=0, keepdims=True), len(te), axis=0)
        rows.append({"repeat": rep, "model": "Mean trajectory", **metrics(yte, last_pred, red_te)})

        flat_tr = Xtr.reshape(len(Xtr), -1)
        flat_te = Xte.reshape(len(Xte), -1)
        w = ridge_fit(flat_tr, ytr, alpha=4.0)
        ridge_pred = ridge_predict(flat_te, w)
        rows.append({"repeat": rep, "model": "Temporal ridge", **metrics(yte, ridge_pred, red_te)})

        mlp = MLPRegressor(flat_tr.shape[1], ytr.shape[1], np.random.default_rng(seed + rep))
        mlp.fit(flat_tr, ytr)
        mlp_pred = np.clip(mlp.predict(flat_te), 0, 1)
        rows.append({"repeat": rep, "model": "Deep MLP", **metrics(yte, mlp_pred, red_te)})
    return pd.DataFrame(rows)


def make_plot(summary: pd.DataFrame, out: Path) -> None:
    models = ["Mean trajectory", "Temporal ridge", "Deep MLP"]
    width, height = 1100, 620
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    try:
        title = ImageFont.truetype("arialbd.ttf", 28)
        f = ImageFont.truetype("arial.ttf", 18)
        small = ImageFont.truetype("arial.ttf", 15)
    except OSError:
        title = f = small = ImageFont.load_default()
    draw.text((36, 26), "ML prediction test for dynamic TRISS trajectories", fill=(20, 20, 20), font=title)
    left, top, bottom = 110, 95, 500
    max_y = float(summary["mse_mean"].max()) * 1.2
    colors = [(160, 160, 160), (92, 132, 168), (47, 111, 115)]
    for tick in np.linspace(0, max_y, 6):
        y = bottom - tick / max_y * (bottom - top)
        draw.line((left, y, width - 60, y), fill=(230, 230, 230))
        draw.text((45, y - 10), f"{tick:.3f}", fill=(80, 80, 80), font=small)
    group = (width - left - 80) / len(models)
    for i, model in enumerate(models):
        row = summary[summary["model"] == model].iloc[0]
        val = float(row["mse_mean"])
        x = left + i * group + 105
        h = val / max_y * (bottom - top)
        y = bottom - h
        draw.rectangle((x, y, x + 96, bottom), fill=colors[i])
        draw.text((x + 5, y - 26), f"{val:.4f}", fill=(40, 40, 40), font=small)
        draw.text((x - 20, bottom + 24), model, fill=(40, 40, 40), font=f)
        draw.text((x - 16, bottom + 52), f"AUC {row['red_auc_mean']:.3f}", fill=(85, 85, 85), font=small)
    draw.text((36, height - 54), "Lower MSE is better. AUC is computed for red-zone transition detection.", fill=(80, 80, 80), font=small)
    img.save(out)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = run()
    raw.to_csv(OUT / "dynamic_triss_ml_results_raw.csv", index=False)
    summary = raw.groupby("model")[["mse", "mae", "shape_corr", "red_auc"]].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(c).strip("_") for c in summary.columns.to_flat_index()]
    summary.to_csv(OUT / "dynamic_triss_ml_results_summary.csv", index=False)
    make_plot(summary, OUT / "dynamic_triss_ml_prediction.png")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()

