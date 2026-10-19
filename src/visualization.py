"""Visualization module for bike rental demand patterns (postponed deliverable)."""

import os
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.eda import SEASON_NAMES, WEATHER_NAMES, compute_feature_correlations


def generate_eda_charts(df: pd.DataFrame, output_dir: str = "reports/figures") -> Dict[str, str]:
    """Create and save core EDA visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    generated_files = {}

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Chart 1: Target Variable (cnt) Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["cnt"], bins=50, color="#3b82f6", edgecolor="white", alpha=0.85)
    mean_val = df["cnt"].mean()
    median_val = df["cnt"].median()
    upper_fence = df["cnt"].quantile(0.75) + 1.5 * (df["cnt"].quantile(0.75) - df["cnt"].quantile(0.25))

    ax.axvline(mean_val, color="#ef4444", linestyle="--", linewidth=2, label=f"Mean ({mean_val:.1f})")
    ax.axvline(median_val, color="#10b981", linestyle="-", linewidth=2, label=f"Median ({median_val:.1f})")
    ax.axvline(upper_fence, color="#f59e0b", linestyle=":", linewidth=2, label=f"IQR Fence ({upper_fence:.1f})")

    ax.set_title("Rental Demand (cnt) Distribution — Positively Skewed (+1.28)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Hourly Rental Count (cnt)", fontsize=11)
    ax.set_ylabel("Frequency (Hours)", fontsize=11)
    ax.legend(frameon=True)
    fig.tight_layout()
    p1 = os.path.join(output_dir, "01_target_distribution.png")
    fig.savefig(p1, dpi=200)
    plt.close(fig)
    generated_files["target_distribution"] = p1

    # Chart 2: Hourly Demand Patterns (Commute vs Leisure)
    fig, ax = plt.subplots(figsize=(9, 5))
    hrs = list(range(24))
    wk_mean = [df[(df["workingday"] == 1) & (df["hr"] == h)]["cnt"].mean() for h in hrs]
    nwk_mean = [df[(df["workingday"] == 0) & (df["hr"] == h)]["cnt"].mean() for h in hrs]

    ax.plot(hrs, wk_mean, marker="o", color="#ef4444", linewidth=2.2, label="Working Days (Commute Peaks: 8 AM & 5 PM)")
    ax.plot(hrs, nwk_mean, marker="s", color="#3b82f6", linewidth=2.2, linestyle="--", label="Weekends/Holidays (Afternoon Leisure Curve)")

    ax.set_title("Hourly Demand: Commuter Commute vs Weekend Leisure Patterns", fontsize=13, fontweight="bold")
    ax.set_xlabel("Hour of Day (0 to 23)", fontsize=11)
    ax.set_ylabel("Average Hourly Rentals", fontsize=11)
    ax.set_xticks(hrs)
    ax.legend(frameon=True)
    fig.tight_layout()
    p2 = os.path.join(output_dir, "02_hourly_demand_pattern.png")
    fig.savefig(p2, dpi=200)
    plt.close(fig)
    generated_files["hourly_demand"] = p2

    # Chart 3: Season-wise Demand (SRS v1.5 Mapping)
    fig, ax = plt.subplots(figsize=(7, 5))
    season_order = [1, 2, 3, 4]
    season_labels = [SEASON_NAMES[s] for s in season_order]
    season_means = [df[df["season"] == s]["cnt"].mean() for s in season_order]
    colors = ["#60a5fa", "#34d399", "#f59e0b", "#fb923c"]

    bars = ax.bar(season_labels, season_means, color=colors, edgecolor="black", linewidth=0.8, alpha=0.9)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_title("Average Hourly Demand by Season (SRS v1.5 Mapping)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Season", fontsize=11)
    ax.set_ylabel("Average Hourly Rentals", fontsize=11)
    ax.set_ylim(0, max(season_means) * 1.18)
    fig.tight_layout()
    p3 = os.path.join(output_dir, "03_seasonal_demand.png")
    fig.savefig(p3, dpi=200)
    plt.close(fig)
    generated_files["seasonal_demand"] = p3

    # Chart 4: Weather Severity Impact
    fig, ax = plt.subplots(figsize=(8, 5))
    weather_order = [1, 2, 3, 4]
    weather_labels = [WEATHER_NAMES[w] for w in weather_order]
    weather_means = [df[df["weathersit"] == w]["cnt"].mean() if len(df[df["weathersit"] == w]) > 0 else 0 for w in weather_order]
    weather_counts = [len(df[df["weathersit"] == w]) for w in weather_order]
    w_colors = ["#10b981", "#64748b", "#3b82f6", "#ef4444"]

    bars = ax.bar(weather_labels, weather_means, color=w_colors, edgecolor="black", linewidth=0.8, alpha=0.9)
    for bar, count in zip(bars, weather_counts):
        h = bar.get_height()
        ax.annotate(f"{h:.1f} bikes\n(n={count})", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_title("Average Hourly Demand by Weather Severity", fontsize=13, fontweight="bold")
    ax.set_xlabel("Weather Condition", fontsize=11)
    ax.set_ylabel("Average Hourly Rentals", fontsize=11)
    ax.set_ylim(0, max(weather_means) * 1.22)
    fig.tight_layout()
    p4 = os.path.join(output_dir, "04_weather_demand.png")
    fig.savefig(p4, dpi=200)
    plt.close(fig)
    generated_files["weather_demand"] = p4

    # Chart 5: Feature Correlations with cnt
    fig, ax = plt.subplots(figsize=(8, 5))
    corrs = compute_feature_correlations(df)
    sorted_corrs = sorted(corrs.items(), key=lambda x: x[1])
    feat_names = [x[0] for x in sorted_corrs]
    feat_vals = [x[1] for x in sorted_corrs]
    bar_colors = ["#ef4444" if v < 0 else "#10b981" for v in feat_vals]

    bars = ax.barh(feat_names, feat_vals, color=bar_colors, edgecolor="black", linewidth=0.8, alpha=0.9)
    for bar in bars:
        w = bar.get_width()
        offset = 5 if w >= 0 else -30
        ax.annotate(f"{w:+.3f}", xy=(w, bar.get_y() + bar.get_height() / 2), xytext=(offset, -3),
                    textcoords="offset points", ha="left" if w >= 0 else "right", va="center", fontsize=9, fontweight="bold")

    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Feature Correlations with Rental Demand (cnt)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Pearson Correlation Coefficient (r)", fontsize=11)
    ax.set_xlim(-0.45, 0.55)
    fig.tight_layout()
    p5 = os.path.join(output_dir, "05_feature_correlations.png")
    fig.savefig(p5, dpi=200)
    plt.close(fig)
    generated_files["feature_correlations"] = p5

    return generated_files
