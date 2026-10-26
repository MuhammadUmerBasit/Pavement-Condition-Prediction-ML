"""
Benchmarking & Visualization Script for Pavement Roughness ML Models.
Generates evaluation figures and metrics summary for README and reports.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from model import train_and_benchmark


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "dataset", "pavement_traffic_dataset.csv")
    media_dir = os.path.join(base_dir, "media")
    os.makedirs(media_dir, exist_ok=True)

    benchmark_data = train_and_benchmark(csv_path)
    df = benchmark_data["df"]
    results = benchmark_data["results"]
    preds = benchmark_data["predictions"]

    y_actual = df["IRI_in_per_mi"].values

    print("\n" + "=" * 65)
    print("           MODEL BENCHMARK RESULTS (N=75 SAMPLES)")
    print("=" * 65)
    print(f"{'Model Name':<30} | {'MAE (in/mi)':<12} | {'RMSE':<10} | {'R2 Score':<8}")
    print("-" * 65)
    for model_name, metrics in results.items():
        print(f"{model_name:<30} | {metrics['MAE']:<12} | {metrics['RMSE']:<10} | {metrics['R2']:<8}")
    print("=" * 65 + "\n")

    # Set aesthetic style
    sns.set_theme(style="whitegrid", font="sans-serif")

    # 1. Plot Actual vs. Predicted IRI
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Domain Piecewise Model
    y_pred_domain = preds["Domain Piecewise Model"]
    r2_domain = results["Domain Piecewise Model"]["R2"]
    rmse_domain = results["Domain Piecewise Model"]["RMSE"]

    axes[0].scatter(y_actual, y_pred_domain, color="#1f77b4", edgecolors="k", s=60, alpha=0.8, label="Segments")
    min_val = min(y_actual.min(), y_pred_domain.min()) - 10
    max_val = max(y_actual.max(), y_pred_domain.max()) + 10
    axes[0].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Ideal Line (y = x)")
    axes[0].set_title(f"Domain Piecewise Model\n(R² = {r2_domain:.3f}, RMSE = {rmse_domain:.2f} in/mi)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Actual Measured IRI (in/mi)", fontsize=11)
    axes[0].set_ylabel("Predicted IRI (in/mi)", fontsize=11)
    axes[0].set_xlim(min_val, max_val)
    axes[0].set_ylim(min_val, max_val)
    axes[0].legend(loc="upper left")

    # Random Forest Regressor
    y_pred_rf = preds["Random Forest Regressor"]
    r2_rf = results["Random Forest Regressor"]["R2"]
    rmse_rf = results["Random Forest Regressor"]["RMSE"]

    axes[1].scatter(y_actual, y_pred_rf, color="#2ca02c", edgecolors="k", s=60, alpha=0.8, label="Segments")
    min_val2 = min(y_actual.min(), y_pred_rf.min()) - 10
    max_val2 = max(y_actual.max(), y_pred_rf.max()) + 10
    axes[1].plot([min_val2, max_val2], [min_val2, max_val2], "r--", linewidth=2, label="Ideal Line (y = x)")
    axes[1].set_title(f"Random Forest Regressor\n(R² = {r2_rf:.3f}, RMSE = {rmse_rf:.2f} in/mi)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Actual Measured IRI (in/mi)", fontsize=11)
    axes[1].set_ylabel("Predicted IRI (in/mi)", fontsize=11)
    axes[1].set_xlim(min_val2, max_val2)
    axes[1].set_ylim(min_val2, max_val2)
    axes[1].legend(loc="upper left")

    plt.tight_layout()
    pred_fig_path = os.path.join(media_dir, "actual_vs_predicted_iri.png")
    plt.savefig(pred_fig_path, dpi=300)
    plt.close()
    print(f"Saved: {pred_fig_path}")

    # 2. Plot Feature Correlation & Pavement Condition Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Scatter: IRI vs Structural Number by Highway Section
    palette = sns.color_palette("tab10", len(df["Highway_Section"].unique()))
    sns.scatterplot(
        data=df,
        x="Structural_Number_SN",
        y="IRI_in_per_mi",
        hue="Highway_Section",
        style="Highway_Section",
        s=80,
        palette=palette,
        ax=axes[0],
    )
    # FHWA Threshold line for Poor pavement (> 170 in/mi)
    axes[0].axhline(170, color="red", linestyle=":", linewidth=1.5, label="FHWA Poor Threshold (>170 in/mi)")
    axes[0].axhline(95, color="green", linestyle=":", linewidth=1.5, label="FHWA Good Threshold (<95 in/mi)")
    axes[0].set_title("Pavement Roughness vs. Structural Number", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Effective Structural Number (SN)", fontsize=11)
    axes[0].set_ylabel("International Roughness Index (IRI in/mi)", fontsize=11)
    axes[0].legend(fontsize=8, loc="upper right")

    # Bar chart: Condition breakdown across sections
    condition_counts = pd.crosstab(df["Highway_Section"], df["Condition_Rating"])
    condition_counts.plot(
        kind="barh",
        stacked=True,
        color={"Good": "#2ca02c", "Fair": "#ff7f0e", "Poor": "#d62728"},
        ax=axes[1],
    )
    axes[1].set_title("Pavement Condition Rating by Highway Section", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Number of Highway Segments (1 km segments)", fontsize=11)
    axes[1].set_ylabel("N-5 Highway Section", fontsize=11)
    axes[1].legend(title="Rating", loc="lower right")

    plt.tight_layout()
    corr_fig_path = os.path.join(media_dir, "pavement_condition_breakdown.png")
    plt.savefig(corr_fig_path, dpi=300)
    plt.close()
    print(f"Saved: {corr_fig_path}")

    # 3. Model Benchmark Bar Chart
    fig, ax = plt.subplots(figsize=(10, 5))
    model_names = list(results.keys())
    r2_scores = [results[m]["R2"] for m in model_names]
    rmses = [results[m]["RMSE"] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    ax.bar(x - width/2, r2_scores, width, label="R² Score (Higher is better)", color="#3498db")
    ax.set_ylabel("R² Score", color="#3498db", fontsize=11)
    ax.set_ylim(0, 1.0)

    ax2 = ax.twinx()
    ax2.bar(x + width/2, rmses, width, label="RMSE in/mi (Lower is better)", color="#e74c3c")
    ax2.set_ylabel("RMSE (in/mi)", color="#e74c3c", fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right", fontsize=10)
    ax.set_title("Machine Learning Model Performance Comparison (N-5 Highway Dataset)", fontsize=12, fontweight="bold")

    # Combine legends
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left")

    plt.tight_layout()
    comp_fig_path = os.path.join(media_dir, "model_benchmark_comparison.png")
    plt.savefig(comp_fig_path, dpi=300)
    plt.close()
    print(f"Saved: {comp_fig_path}")


if __name__ == "__main__":
    main()
