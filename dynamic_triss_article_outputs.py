from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_triss"
FEATURES = ROOT / "results" / "mimic_dynamic_triss_features.csv"
BINS = ROOT / "results" / "mimic_patient_time_bins.csv"
BIN_LABELS = ["0-2", "2-4", "4-6", "6-8", "8-12", "12-18", "18-24"]
BIN_CENTERS = np.array([1, 3, 5, 7, 10, 15, 21], dtype=float)


def font(size: int, bold: bool = False):
    for name in (["arialbd.ttf", "arial.ttf"] if bold else ["arial.ttf"]):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def build_summary(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stay_id, group in features.groupby("stay_id"):
        g = group.sort_values("time_from_icu_h")
        prob = g["survival_6h"].to_numpy()
        time = g["time_from_icu_h"].to_numpy()
        red = g[g["survival_6h"] < 0.50]
        first_red = float(red["time_from_icu_h"].iloc[0]) if len(red) else np.nan
        rows.append(
            {
                "stay_id": stay_id,
                "subject_id": int(g["subject_id"].iloc[0]),
                "n_points": len(g),
                "start_survival_6h": float(prob[0]),
                "min_survival_6h": float(prob.min()),
                "max_survival_6h": float(prob.max()),
                "dynamic_range": float(prob.max() - prob.min()),
                "first_red_hour": first_red,
                "red_event": int(len(red) > 0),
            }
        )
    return pd.DataFrame(rows)


def text_size(draw: ImageDraw.ImageDraw, text: str, font_obj) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font_obj)
    return int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])


def draw_rotated_text(img: Image.Image, xy: tuple[int, int], text: str, font_obj, fill: tuple[int, int, int]) -> None:
    tmp = Image.new("RGBA", (520, 80), (255, 255, 255, 0))
    d = ImageDraw.Draw(tmp)
    d.text((0, 0), text, fill=fill + (255,), font=font_obj)
    tmp = tmp.crop(tmp.getbbox()).rotate(90, expand=True)
    img.paste(tmp, xy, tmp)


def trajectory_table(binned: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stay_id, group in binned.groupby("stay_id"):
        g = group.set_index("time_bin").reindex(BIN_LABELS)
        observed_bins = int(g["survival_6h"].notna().sum())
        y = g["survival_6h"].ffill().bfill()
        if y.isna().all() or observed_bins < 5:
            continue
        y = y.fillna(float(y.median())).to_numpy(float)
        ranks_y = pd.Series(y).rank(method="average").to_numpy(float)
        spearman = float(np.corrcoef(np.arange(len(y), dtype=float), ranks_y)[0, 1])
        if not np.isfinite(spearman):
            spearman = 0.0
        row = {
            "stay_id": int(stay_id),
            "subject_id": int(group["subject_id"].iloc[0]),
            "observed_bins": observed_bins,
            "start_survival_6h": float(y[0]),
            "end_survival_6h": float(y[-1]),
            "delta_survival_6h": float(y[-1] - y[0]),
            "min_survival_6h": float(y.min()),
            "max_survival_6h": float(y.max()),
            "spearman_time": spearman,
        }
        for label, val in zip(BIN_LABELS, y):
            row[f"P_{label}"] = float(val)
        rows.append(row)
    return pd.DataFrame(rows)


def select_representative_trajectories(binned: pd.DataFrame) -> pd.DataFrame:
    traj = trajectory_table(binned)
    improving = traj[
        (traj["observed_bins"] >= 6)
        & (traj["start_survival_6h"] < 0.75)
        & (traj["delta_survival_6h"] > 0.18)
        & (traj["spearman_time"] >= 0.60)
    ].copy()
    improving = improving.sort_values(["delta_survival_6h", "spearman_time"], ascending=[False, False]).head(3)
    if len(improving) < 3:
        fallback = traj[
            (traj["observed_bins"] >= 5)
            & (traj["start_survival_6h"] < 0.75)
            & (traj["delta_survival_6h"] > 0.12)
        ].sort_values(["delta_survival_6h", "spearman_time"], ascending=[False, False])
        improving = pd.concat([improving, fallback]).drop_duplicates("stay_id").head(3)
    improving["trend"] = "positive"

    critical_worsening = traj[
        (traj["observed_bins"] >= 6)
        & (traj["start_survival_6h"] > 0.45)
        & (traj["end_survival_6h"] < 0.50)
        & (traj["delta_survival_6h"] < -0.10)
    ].sort_values("end_survival_6h", ascending=True).head(1)
    structured_worsening = traj[
        (traj["observed_bins"] >= 6)
        & (traj["start_survival_6h"] > 0.70)
        & (traj["end_survival_6h"] < 0.75)
        & (traj["delta_survival_6h"] < -0.08)
        & (traj["spearman_time"] < -0.55)
    ].sort_values("delta_survival_6h", ascending=True).head(2)
    worsening = pd.concat([structured_worsening, critical_worsening]).drop_duplicates("stay_id")
    if len(worsening) < 3:
        fallback = traj[
            (traj["observed_bins"] >= 6)
            & (traj["delta_survival_6h"] < -0.08)
        ].copy()
        fallback["selection_score"] = (
            -fallback["delta_survival_6h"]
            + 0.35 * (1.0 - fallback["end_survival_6h"])
            + 0.08 * np.maximum(0.0, -fallback["spearman_time"])
        )
        fallback = fallback.sort_values("selection_score", ascending=False)
        worsening = pd.concat([worsening, fallback]).drop_duplicates("stay_id").head(3)
    worsening = worsening.sort_values(
        ["start_survival_6h", "delta_survival_6h"],
        ascending=[False, True],
    )
    worsening["trend"] = "negative"

    selected = pd.concat([improving, worsening], ignore_index=True)
    selected["display_label"] = selected.apply(
        lambda r: f"Stay {int(r['stay_id'])}: {r['start_survival_6h']:.2f} -> {r['end_survival_6h']:.2f}",
        axis=1,
    )
    return selected


def make_curves(binned: pd.DataFrame, out: Path) -> pd.DataFrame:
    selected = select_representative_trajectories(binned)
    scale = 2
    width, height = 1900 * scale, 1010 * scale
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title = font(35 * scale, True)
    subtitle = font(19 * scale)
    label_font = font(20 * scale)
    small = font(16 * scale)
    tick_font = font(15 * scale)
    legend_font = font(15 * scale)

    draw.text((46 * scale, 26 * scale), "Representative dynamic TRISS-inspired trajectories", fill=(20, 20, 20), font=title)
    draw.text(
        (48 * scale, 74 * scale),
        "Aggregated by time windows; patients with pronounced improvement or worsening of the dynamic survival estimate were selected.",
        fill=(78, 78, 78),
        font=subtitle,
    )

    panels = [
        ("positive", "Positive dynamics", (150 * scale, 145 * scale, 890 * scale, 700 * scale)),
        ("negative", "Negative dynamics", (1010 * scale, 145 * scale, 1750 * scale, 700 * scale)),
    ]
    palettes = {
        "positive": [(35, 125, 118), (54, 105, 180), (88, 145, 80)],
        "negative": [(170, 70, 70), (140, 88, 160), (190, 115, 50)],
    }
    zone_defs = [
        (0.75, 1.0, (226, 244, 232), "green zone > 0.75", (45, 105, 65)),
        (0.50, 0.75, (255, 247, 218), "yellow zone 0.50-0.75", (128, 96, 28)),
        (0.0, 0.50, (251, 229, 229), "red zone < 0.50", (150, 55, 55)),
    ]

    def xy(panel_box: tuple[int, int, int, int], x_hour: float, prob: float) -> tuple[int, int]:
        left, top, right, bottom = panel_box
        x = left + (x_hour / 24.0) * (right - left)
        y = bottom - prob * (bottom - top)
        return int(x), int(y)

    for trend, panel_title, box in panels:
        left, top, right, bottom = box
        draw.text((left, top - 54 * scale), panel_title, fill=(30, 30, 30), font=label_font)
        for y0, y1, color, _, _ in zone_defs:
            _, yy1 = xy(box, 0, y1)
            _, yy0 = xy(box, 0, y0)
            draw.rectangle((left, yy1, right, yy0), fill=color)
        for tick in [0.0, 0.25, 0.50, 0.75, 1.0]:
            _, y = xy(box, 0, tick)
            grid_color = (185, 185, 185) if tick in [0.50, 0.75] else (218, 218, 218)
            draw.line((left, y, right, y), fill=grid_color, width=2 * scale if tick in [0.50, 0.75] else scale)
            draw.text((left - 58 * scale, y - 10 * scale), f"{tick:.2f}", fill=(70, 70, 70), font=tick_font)
        for hour, label in zip(BIN_CENTERS, BIN_LABELS):
            x, _ = xy(box, hour, 0)
            draw.line((x, top, x, bottom), fill=(232, 232, 232), width=scale)
            tw, _ = text_size(draw, label, tick_font)
            draw.text((x - tw // 2, bottom + 16 * scale), label, fill=(70, 70, 70), font=tick_font)
        draw.rectangle((left, top, right, bottom), outline=(178, 178, 178), width=2 * scale)

        panel_rows = selected[selected["trend"] == trend].reset_index(drop=True)
        for idx, row in panel_rows.iterrows():
            color = palettes[trend][idx]
            values = [float(row[f"P_{label}"]) for label in BIN_LABELS]
            points = [xy(box, hour, prob) for hour, prob in zip(BIN_CENTERS, values)]
            draw.line(points, fill=color, width=5 * scale)
            for x, y in points:
                r = 5 * scale
                draw.ellipse((x - r, y - r, x + r, y + r), fill="white", outline=color, width=3 * scale)
            end_x, end_y = points[-1]
            label = f"{int(row['stay_id'])}  Δ={row['delta_survival_6h']:+.2f}"
            label_offsets = [-12, -12, -12] if trend == "positive" else [-36, 2, -12]
            label_y = max(top + 10 * scale, min(bottom - 28 * scale, end_y + label_offsets[idx] * scale))
            draw.text((end_x + 12 * scale, label_y), label, fill=color, font=legend_font)

        draw.text((left + 130 * scale, bottom + 60 * scale), "Time window from ICU admission, hours", fill=(45, 45, 45), font=small)

    draw_rotated_text(
        img,
        (58 * scale, 315 * scale),
        "Estimated probability of favorable outcome P_i(t)",
        label_font,
        (45, 45, 45),
    )
    legend_x = 150 * scale
    legend_y = 805 * scale
    for idx, (_, _, color, label, text_color) in enumerate(zone_defs):
        x = legend_x + idx * 420 * scale
        draw.rectangle((x, legend_y, x + 34 * scale, legend_y + 22 * scale), fill=color, outline=(205, 205, 205))
        draw.text((x + 46 * scale, legend_y - 2 * scale), label, fill=text_color, font=small)
    draw.text(
        (150 * scale, 872 * scale),
        "Raw minute-level measurements are not shown; values are aggregated by time window, which makes the overall trend readable and reproducible.",
        fill=(78, 78, 78),
        font=small,
    )
    img = img.resize((width // scale, height // scale), Image.Resampling.LANCZOS)
    img.save(out)
    return selected


def make_zone_chart(summary: pd.DataFrame, out: Path) -> None:
    zones = {
        "green": int((summary["min_survival_6h"] >= 0.75).sum()),
        "yellow": int(((summary["min_survival_6h"] >= 0.50) & (summary["min_survival_6h"] < 0.75)).sum()),
        "red": int((summary["min_survival_6h"] < 0.50).sum()),
    }
    width, height = 900, 560
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title = font(28, True)
    f = font(18)
    small = font(15)
    draw.text((36, 28), "Worst-zone distribution by ICU stay", fill=(20, 20, 20), font=title)
    total = sum(zones.values())
    colors = {"green": (78, 145, 92), "yellow": (210, 160, 55), "red": (180, 70, 70)}
    left, top, bottom = 130, 100, 450
    max_val = max(zones.values()) * 1.15
    for i, (name, value) in enumerate(zones.items()):
        x = left + i * 220
        h = value / max_val * (bottom - top)
        y = bottom - h
        draw.rectangle((x, y, x + 110, bottom), fill=colors[name])
        draw.text((x + 35, y - 30), str(value), fill=(40, 40, 40), font=f)
        draw.text((x + 16, bottom + 20), name, fill=(40, 40, 40), font=f)
        draw.text((x + 14, bottom + 48), f"{value/total:.1%}", fill=(90, 90, 90), font=small)
    draw.text((36, height - 54), "Zone is determined by the minimum 6-hour survival estimate within the first 24h.", fill=(80, 80, 80), font=small)
    img.save(out)
    return zones


def make_model_chart(models: pd.DataFrame, out: Path) -> None:
    width, height = 1050, 600
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title = font(28, True)
    f = font(18)
    small = font(15)
    draw.text((36, 28), "Neural trajectory models: MSE comparison", fill=(20, 20, 20), font=title)
    left, top, bottom = 120, 95, 465
    max_val = float(models["mse"].max()) * 1.25
    colors = [(140, 140, 140), (95, 130, 170), (47, 111, 115)]
    for tick in np.linspace(0, max_val, 5):
        y = bottom - tick / max_val * (bottom - top)
        draw.line((left, y, width - 70, y), fill=(230, 230, 230))
        draw.text((55, y - 10), f"{tick:.3f}", fill=(80, 80, 80), font=small)
    for i, row in models.iterrows():
        x = left + i * 280 + 85
        val = float(row["mse"])
        h = val / max_val * (bottom - top)
        y = bottom - h
        draw.rectangle((x, y, x + 100, bottom), fill=colors[i])
        draw.text((x + 5, y - 28), f"{val:.4f}", fill=(40, 40, 40), font=small)
        draw.text((x - 12, bottom + 22), str(row["model"]), fill=(40, 40, 40), font=f)
        draw.text((x - 10, bottom + 50), f"corr {row['shape_corr']:.3f}", fill=(80, 80, 80), font=small)
    draw.text((36, height - 56), "Lower MSE is better; shape correlation measures trajectory-form similarity.", fill=(80, 80, 80), font=small)
    img.save(out)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(FEATURES)
    summary = build_summary(features)
    summary.to_csv(OUT / "dynamic_triss_trajectory_summary.csv", index=False)

    zones = make_zone_chart(summary, OUT / "dynamic_triss_zone_distribution.png")
    pd.DataFrame([{"zone": k, "stay_count": v} for k, v in zones.items()]).to_csv(
        OUT / "dynamic_triss_zone_distribution.csv", index=False
    )
    binned = pd.read_csv(BINS)
    selected = make_curves(binned, OUT / "dynamic_triss_curves.png")
    selected.to_csv(OUT / "dynamic_triss_representative_trajectories.csv", index=False)

    key = pd.DataFrame(
        [
            {"metric": "icu_stays", "value": int(summary["stay_id"].nunique())},
            {"metric": "red_event_stays", "value": int(summary["red_event"].sum())},
            {"metric": "mean_dynamic_range", "value": float(summary["dynamic_range"].mean())},
            {"metric": "median_min_survival_6h", "value": float(summary["min_survival_6h"].median())},
            {"metric": "mean_first_red_hour", "value": float(summary["first_red_hour"].dropna().mean())},
        ]
    )
    key.to_csv(OUT / "dynamic_triss_key_metrics.csv", index=False)
    print(key.to_string(index=False))


if __name__ == "__main__":
    main()
