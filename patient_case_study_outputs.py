from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from patient_allocation_experiment import (
    make_hospitals,
    make_patients_from_features,
    nearest_assignment,
    optimize_assignment,
    survival_matrix,
    travel_minutes,
)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_mimic"
FEATURES = ROOT / "results" / "mimic_dynamic_risk_features.csv"


def load_font(size: int, bold: bool = False):
    names = ["arialbd.ttf", "arial.ttf"] if bold else ["arial.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def label(policy: str) -> str:
    return {
        "normal": "Normal load",
        "mass_casualty": "Mass casualty",
        "specialty_shortage": "Specialty shortage",
    }.get(policy, policy)


def build_case(seed: int = 20260511, scenario: str = "normal", n: int = 140, h: int = 8) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    features = pd.read_csv(FEATURES)

    patients = make_patients_from_features(features, n, rng, scenario)
    hospitals = make_hospitals(n, h, rng, scenario)
    travel = travel_minutes(patients, hospitals, rng, scenario)

    pred_dynamic = survival_matrix(patients, hospitals, travel, predicted=True, dynamic=True)
    pred_static = survival_matrix(patients, hospitals, travel, predicted=True, dynamic=False)
    true_dynamic = survival_matrix(patients, hospitals, travel, predicted=False, dynamic=True)

    nearest = nearest_assignment(travel, hospitals["capacity"].to_numpy(), np.arange(n))
    dynamic = optimize_assignment(pred_dynamic, hospitals["capacity"].to_numpy())
    static = optimize_assignment(pred_static, hospitals["capacity"].to_numpy())

    gain = true_dynamic[np.arange(n), dynamic] - true_dynamic[np.arange(n), nearest]
    changed = np.where(dynamic != nearest)[0]
    if len(changed) == 0:
        selected = np.argsort(-gain)[:6]
    else:
        selected = changed[np.argsort(-gain[changed])[:6]]

    rows = []
    for rank, i in enumerate(selected, start=1):
        dyn_j = int(dynamic[i])
        near_j = int(nearest[i])
        stat_j = int(static[i])
        top = np.argsort(-pred_dynamic[i])[:3]
        top3 = "; ".join(
            f"H{j} ({hospitals.loc[j, 'hospital_type']}, {pred_dynamic[i, j]:.3f})"
            for j in top
        )
        rows.append(
            {
                "case": rank,
                "patient_id": int(patients.loc[i, "pid"]),
                "age": int(round(patients.loc[i, "age"])),
                "injury_type": patients.loc[i, "injury_type"],
                "risk_score": round(float(patients.loc[i, "vital_score"]), 3),
                "shock_index": round(float(patients.loc[i, "shock_index"]), 2),
                "iss_proxy": round(float(patients.loc[i, "iss_proxy"]), 1),
                "base_survival": round(float(patients.loc[i, "true_base_survival"]), 3),
                "deadline_min": round(float(patients.loc[i, "deadline_min"]), 1),
                "nearest_hospital": f"H{near_j}",
                "nearest_profile": hospitals.loc[near_j, "hospital_type"],
                "nearest_travel_min": round(float(travel[i, near_j]), 1),
                "nearest_survival": round(float(true_dynamic[i, near_j]), 3),
                "static_hospital": f"H{stat_j}",
                "dynamic_hospital": f"H{dyn_j}",
                "dynamic_profile": hospitals.loc[dyn_j, "hospital_type"],
                "dynamic_travel_min": round(float(travel[i, dyn_j]), 1),
                "dynamic_survival": round(float(true_dynamic[i, dyn_j]), 3),
                "survival_gain": round(float(gain[i]), 3),
                "top3_recommendations": top3,
            }
        )

    cases = pd.DataFrame(rows)
    cases.to_csv(OUT / "case_study_patients.csv", index=False)
    hospitals_out = hospitals.copy()
    hospitals_out["label"] = [f"H{i}" for i in range(len(hospitals_out))]
    hospitals_out[["label", "capacity", "hospital_type", "quality", "x", "y"]].to_csv(
        OUT / "case_study_hospitals.csv", index=False
    )

    experiment_design = pd.DataFrame(
        [
            {
                "scenario": "normal",
                "meaning": "Baseline load; total capacity moderately exceeds demand.",
                "patients": 140,
                "hospitals": 8,
                "transport": "Spatial travel-time model with stochastic noise.",
                "purpose": "Check method behavior under routine surge conditions.",
            },
            {
                "scenario": "mass_casualty",
                "meaning": "Patient count increased by 35%; transport times are inflated.",
                "patients": 189,
                "hospitals": 8,
                "transport": "Travel-time multiplier = 1.35.",
                "purpose": "Stress test under simultaneous inflow.",
            },
            {
                "scenario": "specialty_shortage",
                "meaning": "Lower share of trauma/burn-capable hospitals.",
                "patients": 140,
                "hospitals": 8,
                "transport": "Travel-time multiplier = 1.12.",
                "purpose": "Test profile-aware allocation when specialized capacity is scarce.",
            },
        ]
    )
    experiment_design.to_csv(OUT / "experiment_design.csv", index=False)

    make_heatmap(cases, hospitals, pred_dynamic, nearest, dynamic, OUT / "case_study_heatmap.png")
    make_case_bars(cases, OUT / "case_study_patient_bars.png")


def make_heatmap(
    cases: pd.DataFrame,
    hospitals: pd.DataFrame,
    utility: np.ndarray,
    nearest: np.ndarray,
    dynamic: np.ndarray,
    out: Path,
) -> None:
    selected = cases["patient_id"].to_numpy()
    h = len(hospitals)
    cell_w, cell_h = 120, 58
    left, top = 220, 96
    width = left + h * cell_w + 40
    height = top + len(selected) * cell_h + 105

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font_title = load_font(28, bold=True)
    font = load_font(18)
    font_small = load_font(15)
    font_bold = load_font(17, bold=True)

    draw.text((28, 24), "Patient-hospital utility matrix for selected cases", fill=(20, 20, 20), font=font_title)
    draw.text((28, 58), "D = dynamic assignment, N = nearest assignment", fill=(85, 85, 85), font=font_small)

    for j in range(h):
        x = left + j * cell_w
        draw.text((x + 35, top - 32), f"H{j}", fill=(30, 30, 30), font=font_bold)
        draw.text((x + 18, top - 12), str(hospitals.loc[j, "hospital_type"])[:10], fill=(90, 90, 90), font=font_small)

    vals = utility[selected, :]
    vmin, vmax = float(vals.min()), float(vals.max())
    span = max(vmax - vmin, 1e-6)
    for r, pid in enumerate(selected):
        y = top + r * cell_h
        row = cases[cases["patient_id"] == pid].iloc[0]
        draw.text((28, y + 8), f"P{int(pid)}", fill=(20, 20, 20), font=font_bold)
        draw.text((82, y + 8), f"{row['injury_type']}, score {row['risk_score']}", fill=(80, 80, 80), font=font_small)
        for j in range(h):
            x = left + j * cell_w
            val = float(utility[pid, j])
            alpha = (val - vmin) / span
            color = (
                int(238 - 130 * alpha),
                int(244 - 80 * alpha),
                int(247 - 40 * alpha),
            )
            draw.rectangle((x, y, x + cell_w - 4, y + cell_h - 4), fill=color, outline=(210, 210, 210))
            marker = ""
            if int(dynamic[pid]) == j:
                marker += "D"
            if int(nearest[pid]) == j:
                marker += "N"
            draw.text((x + 13, y + 10), f"{val:.3f}", fill=(20, 20, 20), font=font)
            if marker:
                draw.text((x + 75, y + 10), marker, fill=(150, 40, 35), font=font_bold)

    draw.text((28, height - 54), "Darker cells indicate higher personalized utility S_ij.", fill=(80, 80, 80), font=font_small)
    img.save(out)


def make_case_bars(cases: pd.DataFrame, out: Path) -> None:
    width, height = 1300, 620
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title = load_font(28, bold=True)
    font = load_font(18)
    small = load_font(15)
    bold = load_font(17, bold=True)

    draw.text((36, 24), "Nearest vs dynamic assignment for selected patients", fill=(20, 20, 20), font=title)
    left, top, bottom = 115, 95, 500
    plot_w = width - left - 80
    group_w = plot_w / len(cases)
    max_val = 1.0
    for tick in np.linspace(0, 1, 6):
        y = bottom - tick * (bottom - top)
        draw.line((left, y, width - 50, y), fill=(230, 230, 230))
        draw.text((58, y - 10), f"{tick:.1f}", fill=(80, 80, 80), font=small)

    for idx, row in cases.iterrows():
        gx = left + idx * group_w + 25
        values = [
            ("Nearest", row["nearest_survival"], (169, 169, 169)),
            ("Dynamic", row["dynamic_survival"], (47, 111, 115)),
        ]
        for k, (_, value, color) in enumerate(values):
            bar_h = float(value) / max_val * (bottom - top)
            x = gx + k * 38
            y = bottom - bar_h
            draw.rectangle((x, y, x + 30, bottom), fill=color)
            draw.text((x - 3, y - 24), f"{value:.2f}", fill=(40, 40, 40), font=small)
        draw.text((gx - 7, bottom + 18), f"P{int(row['patient_id'])}", fill=(20, 20, 20), font=bold)
        draw.text((gx - 12, bottom + 42), f"+{row['survival_gain']:.3f}", fill=(150, 40, 35), font=small)

    draw.rectangle((930, 38, 950, 58), fill=(169, 169, 169))
    draw.text((960, 35), "Nearest", fill=(50, 50, 50), font=font)
    draw.rectangle((1060, 38, 1080, 58), fill=(47, 111, 115))
    draw.text((1090, 35), "Dynamic", fill=(50, 50, 50), font=font)
    draw.text((36, 552), "The label below each patient is the gain in expected survival after dynamic reassignment.", fill=(80, 80, 80), font=small)
    img.save(out)


if __name__ == "__main__":
    build_case()
