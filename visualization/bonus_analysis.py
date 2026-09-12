"""
BONUS: 2.4GHz vs 5GHz PERFORMANCE COMPARISON
================================================
Aggregate, dataset-level comparison (independent of any single placement scenario) --
useful for the report's "real-world interpretation" section.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

sns.set_theme(style="whitegrid")
OUT_DIR = PROJECT_ROOT / "outputs"


def main():
    df = pd.read_csv(PROJECT_ROOT / "data" / "wifi_dataset.csv").dropna()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = ["signal_strength_dbm", "throughput_mbps", "latency_ms"]
    titles = ["Signal Strength (dBm)", "Throughput (Mbps)", "Latency (ms)"]

    for ax, metric, title in zip(axes, metrics, titles):
        sns.boxplot(data=df, x="frequency_band", y=metric, hue="frequency_band", ax=ax,
                    palette={"2.4GHz": "#2563eb", "5GHz": "#dc2626"}, legend=False)
        ax.set_title(title)
        ax.set_xlabel("")

    plt.suptitle("2.4GHz vs 5GHz — Aggregate Performance Comparison Across Dataset", y=1.03)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/09_band_comparison_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close()

    summary = df.groupby("frequency_band")[metrics].mean().round(2)
    summary.to_csv(f"{OUT_DIR}/band_comparison_summary.csv")

    print("Saved: 09_band_comparison_boxplots.png")
    print("\nMean performance by band:")
    print(summary)


if __name__ == "__main__":
    main()
