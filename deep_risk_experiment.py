from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
INP = ROOT / "results" / "mimic_patient_risk_time_bins.csv"
OUT = ROOT / "results_mimic"

BINS = ["0-2", "2-4", "4-6", "6-8", "8-12", "12-18", "18-24"]
FEATURES = ["HR", "SBP", "RR", "GCS", "shock_index", "vital_score"]


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


def build_sequences(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    global_median = df[FEATURES].median()
    sequences, y_survival, y_red, stay_ids = [], [], [], []

    for stay_id, group in df.groupby("stay_id"):
        group = group.set_index("time_bin").reindex(BINS)
        values = group[FEATURES].copy()
        values = values.ffill().bfill().fillna(global_median)
        sequences.append(values.to_numpy(dtype=float))
        y_survival.append(float(group["survival_6h"].min()))
        y_red.append(float(group["red_zone"].max() > 0))
        stay_ids.append(stay_id)

    return np.asarray(sequences), np.asarray(y_survival), np.asarray(y_red), np.asarray(stay_ids)


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.reshape(-1, train.shape[-1]).mean(axis=0)
    std = train.reshape(-1, train.shape[-1]).std(axis=0) + 1e-8
    return (train - mean) / std, (test - mean) / std


def ridge_regression_fit(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    Xb = np.column_stack([np.ones(len(X)), X])
    eye = np.eye(Xb.shape[1])
    eye[0, 0] = 0
    return np.linalg.solve(Xb.T @ Xb + alpha * eye, Xb.T @ y)


def ridge_predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    Xb = np.column_stack([np.ones(len(X)), X])
    return np.clip(Xb @ w, 0, 1)


class DeepTemporalMLP:
    def __init__(self, input_dim: int, rng: np.random.Generator) -> None:
        dims = [input_dim, 64, 32, 16, 1]
        self.weights = []
        self.biases = []
        for a, b in zip(dims[:-1], dims[1:]):
            self.weights.append(rng.normal(0, np.sqrt(2 / a), size=(a, b)))
            self.biases.append(np.zeros((1, b)))
        self.mw = [np.zeros_like(w) for w in self.weights]
        self.vw = [np.zeros_like(w) for w in self.weights]
        self.mb = [np.zeros_like(b) for b in self.biases]
        self.vb = [np.zeros_like(b) for b in self.biases]
        self.step = 0

    def forward(self, X: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        activations = [X]
        preacts = []
        a = X
        for idx, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ w + b
            preacts.append(z)
            if idx == len(self.weights) - 1:
                a = sigmoid(z)
            else:
                a = relu(z)
            activations.append(a)
        return activations, preacts

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)[0][-1].ravel()

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 650, lr: float = 0.003) -> None:
        y = y.reshape(-1, 1)
        n = len(X)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for _ in range(epochs):
            activations, preacts = self.forward(X)
            pred = activations[-1]
            grad = 2 * (pred - y) / n
            grad = grad * pred * (1 - pred)

            grad_w = []
            grad_b = []
            for layer in reversed(range(len(self.weights))):
                a_prev = activations[layer]
                grad_w.insert(0, a_prev.T @ grad)
                grad_b.insert(0, grad.sum(axis=0, keepdims=True))
                if layer > 0:
                    grad = grad @ self.weights[layer].T
                    grad = grad * (preacts[layer - 1] > 0)

            self.step += 1
            for i in range(len(self.weights)):
                self.mw[i] = beta1 * self.mw[i] + (1 - beta1) * grad_w[i]
                self.vw[i] = beta2 * self.vw[i] + (1 - beta2) * (grad_w[i] ** 2)
                self.mb[i] = beta1 * self.mb[i] + (1 - beta1) * grad_b[i]
                self.vb[i] = beta2 * self.vb[i] + (1 - beta2) * (grad_b[i] ** 2)

                mw_hat = self.mw[i] / (1 - beta1**self.step)
                vw_hat = self.vw[i] / (1 - beta2**self.step)
                mb_hat = self.mb[i] / (1 - beta1**self.step)
                vb_hat = self.vb[i] / (1 - beta2**self.step)

                self.weights[i] -= lr * mw_hat / (np.sqrt(vw_hat) + eps)
                self.biases[i] -= lr * mb_hat / (np.sqrt(vb_hat) + eps)


def metrics(y: np.ndarray, pred: np.ndarray, y_red: np.ndarray) -> dict[str, float]:
    mse = float(np.mean((pred - y) ** 2))
    mae = float(np.mean(np.abs(pred - y)))
    red_score = 1 - pred
    auc = auc_score(y_red, red_score)
    red_pred = (pred < 0.70).astype(int)
    f1_num = 2 * ((red_pred == 1) & (y_red == 1)).sum()
    f1_den = (red_pred == 1).sum() + (y_red == 1).sum()
    f1 = float(f1_num / f1_den) if f1_den else float("nan")
    return {"mse": mse, "mae": mae, "auc_red_zone": auc, "f1_red_zone": f1}


def run(repeats: int = 30, seed: int = 42) -> pd.DataFrame:
    X, y, y_red, stay_ids = build_sequences(INP)
    rng = np.random.default_rng(seed)
    rows = []

    for rep in range(repeats):
        idx = rng.permutation(len(X))
        split = int(0.75 * len(X))
        train_idx, test_idx = idx[:split], idx[split:]
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        red_test = y_red[test_idx]
        X_train_s, X_test_s = standardize(X_train, X_test)

        # Static baseline: last available 6h survival proxy from the sequence target source.
        static_pred = np.repeat(y_train.mean(), len(y_test))
        rows.append({"repeat": rep, "model": "Mean baseline", **metrics(y_test, static_pred, red_test)})

        Xtr_flat = X_train_s.reshape(len(X_train_s), -1)
        Xte_flat = X_test_s.reshape(len(X_test_s), -1)
        w = ridge_regression_fit(Xtr_flat, y_train, alpha=3.0)
        linear_pred = ridge_predict(Xte_flat, w)
        rows.append({"repeat": rep, "model": "Linear temporal ridge", **metrics(y_test, linear_pred, red_test)})

        model = DeepTemporalMLP(Xtr_flat.shape[1], np.random.default_rng(seed + rep))
        model.fit(Xtr_flat, y_train, epochs=650, lr=0.003)
        deep_pred = np.clip(model.predict(Xte_flat), 0, 1)
        rows.append({"repeat": rep, "model": "Deep temporal MLP", **metrics(y_test, deep_pred, red_test)})

    return pd.DataFrame(rows)


def make_plot(summary: pd.DataFrame, out: Path) -> None:
    models = ["Mean baseline", "Linear temporal ridge", "Deep temporal MLP"]
    width, height = 1100, 620
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    try:
        title_font = ImageFont.truetype("arialbd.ttf", 28)
        font = ImageFont.truetype("arial.ttf", 18)
        small = ImageFont.truetype("arial.ttf", 15)
    except OSError:
        title_font = font = small = ImageFont.load_default()

    draw.text((36, 24), "Temporal risk prediction: neural module vs baselines", fill=(20, 20, 20), font=title_font)
    left, top, bottom = 110, 95, 500
    plot_w = width - left - 70
    group_w = plot_w / len(models)
    colors = [(160, 160, 160), (92, 132, 168), (47, 111, 115)]
    max_y = max(0.25, float(summary["mae_mean"].max()) * 1.2)

    for tick in np.linspace(0, max_y, 6):
        y = bottom - tick / max_y * (bottom - top)
        draw.line((left, y, width - 50, y), fill=(230, 230, 230))
        draw.text((45, y - 10), f"{tick:.2f}", fill=(80, 80, 80), font=small)

    for i, model in enumerate(models):
        val = float(summary[summary["model"] == model]["mae_mean"].iloc[0])
        bar_h = val / max_y * (bottom - top)
        x = left + i * group_w + 110
        y = bottom - bar_h
        draw.rectangle((x, y, x + 95, bottom), fill=colors[i])
        draw.text((x + 10, y - 26), f"{val:.3f}", fill=(40, 40, 40), font=small)
        draw.text((x - 42, bottom + 24), model, fill=(30, 30, 30), font=font)

    draw.text((36, 552), "Lower MAE is better. Target: minimum predicted 6-hour survival over the ICU time sequence.", fill=(80, 80, 80), font=small)
    img.save(out)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = run()
    raw.to_csv(OUT / "deep_risk_results_raw.csv", index=False)
    summary = raw.groupby("model")[["mse", "mae", "auc_red_zone", "f1_red_zone"]].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(c).strip("_") for c in summary.columns.to_flat_index()]
    summary.to_csv(OUT / "deep_risk_results_summary.csv", index=False)
    make_plot(summary, OUT / "deep_risk_mae.png")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
