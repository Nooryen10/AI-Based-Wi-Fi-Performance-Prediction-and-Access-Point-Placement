"""
STEP 4: VISUALIZATION
========================
Generates and saves (as PNG, for direct inclusion in the report):
  1. Signal strength vs distance (scatter + trend, split by frequency band)
  2. Throughput vs number of users (scatter + trend, split by interference)
  3. Correlation heatmap of numeric features + targets
  4. A grid-based Wi-Fi coverage heatmap for a single router placement
     (this feeds conceptually into Step 5's placement optimizer)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

sns.set_theme(style="whitegrid")
OUT_DIR = PROJECT_ROOT / "outputs"


def load_data():
    return pd.read_csv(PROJECT_ROOT / "data" / "wifi_dataset.csv").dropna()


def plot_signal_vs_distance(df):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df.sample(2000, random_state=1), x="distance_m", y="signal_strength_dbm",
        hue="frequency_band", alpha=0.4, s=15, palette={"2.4GHz": "#2563eb", "5GHz": "#dc2626"}
    )
    sns.regplot(
        data=df[df.frequency_band == "2.4GHz"], x="distance_m", y="signal_strength_dbm",
        scatter=False, color="#1e3a8a", lowess=True, label="2.4GHz trend"
    )
    sns.regplot(
        data=df[df.frequency_band == "5GHz"], x="distance_m", y="signal_strength_dbm",
        scatter=False, color="#7f1d1d", lowess=True, label="5GHz trend"
    )
    plt.title("Signal Strength vs Distance from Router (by Frequency Band)")
    plt.xlabel("Distance from Router (m)")
    plt.ylabel("Signal Strength (dBm)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/01_signal_vs_distance.png", dpi=150)
    plt.close()


def plot_throughput_vs_users(df):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df.sample(2000, random_state=1), x="num_users", y="throughput_mbps",
        hue="interference_level", alpha=0.5, s=15,
        palette={"low": "#16a34a", "medium": "#ca8a04", "high": "#dc2626"}
    )
    plt.title("Throughput vs Number of Connected Users (by Interference Level)")
    plt.xlabel("Number of Users")
    plt.ylabel("Throughput (Mbps)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/02_throughput_vs_users.png", dpi=150)
    plt.close()


def plot_correlation_heatmap(df):
    numeric_df = df.copy()
    numeric_df["interference_score"] = numeric_df["interference_level"].map(
        {"low": 0, "medium": 1, "high": 2}
    )
    numeric_df["band_5ghz"] = (numeric_df["frequency_band"] == "5GHz").astype(int)
    cols = ["distance_m", "num_walls", "num_users", "interference_score", "band_5ghz",
            "signal_strength_dbm", "throughput_mbps", "latency_ms"]

    plt.figure(figsize=(9, 7))
    corr = numeric_df[cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True,
                linewidths=0.5, cbar_kws={"label": "Pearson correlation"})
    plt.title("Correlation Heatmap: Features vs Wi-Fi Performance Metrics")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/03_correlation_heatmap.png", dpi=150)
    plt.close()


def path_loss_signal(distance, walls, band="2.4GHz", interference_db=0.0):
    """Same physics model as the dataset generator -- reused for coverage simulation."""
    distance = np.maximum(distance, 0.1)
    if band == "5GHz":
        n, waf, tx = 3.2, 5.5, 23.0
    else:
        n, waf, tx = 2.7, 3.5, 20.0
    pl0, d0 = 40.0, 1.0
    path_loss = pl0 + 10 * n * np.log10(distance / d0) + walls * waf
    return tx - path_loss - 0.6 * interference_db


def plot_coverage_heatmap(room_w=20, room_h=15, router_pos=(10, 7.5), walls_grid=None,
                           band="2.4GHz", save_name="04_coverage_heatmap.png"):
    """
    Simulates signal strength across a 2D grid room for a SINGLE router position.
    walls_grid: optional 2D array (same shape as grid) giving obstacle count per cell region.
    """
    resolution = 0.5  # meters per grid cell
    x = np.arange(0, room_w, resolution)
    y = np.arange(0, room_h, resolution)
    X, Y = np.meshgrid(x, y)

    rx, ry = router_pos
    dist_grid = np.sqrt((X - rx) ** 2 + (Y - ry) ** 2)

    if walls_grid is None:
        # default: assume walls increase with distance bands (simple room layout proxy)
        walls_grid = np.clip((dist_grid // 4).astype(int), 0, 5)

    signal_grid = path_loss_signal(dist_grid, walls_grid, band=band)

    plt.figure(figsize=(9, 7))
    hm = plt.pcolormesh(X, Y, signal_grid, cmap="RdYlGn", vmin=-90, vmax=-30, shading="auto")
    plt.colorbar(hm, label="Signal Strength (dBm)")
    plt.scatter(*router_pos, marker="*", s=400, c="black", edgecolors="white", label="Router (AP)")

    # Overlay the "usable coverage" contour at -65 dBm threshold
    plt.contour(X, Y, signal_grid, levels=[-65], colors="blue", linewidths=2)
    plt.title(f"Wi-Fi Coverage Heatmap ({band}) — Router at {router_pos}\n"
              f"Blue contour = -65 dBm usable-coverage threshold")
    plt.xlabel("Room Width (m)")
    plt.ylabel("Room Height (m)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{save_name}", dpi=150)
    plt.close()

    coverage_pct = 100 * np.mean(signal_grid >= -65)
    return coverage_pct


if __name__ == "__main__":
    df = load_data()

    plot_signal_vs_distance(df)
    print("Saved: 01_signal_vs_distance.png")

    plot_throughput_vs_users(df)
    print("Saved: 02_throughput_vs_users.png")

    plot_correlation_heatmap(df)
    print("Saved: 03_correlation_heatmap.png")

    coverage = plot_coverage_heatmap(band="2.4GHz")
    print(f"Saved: 04_coverage_heatmap.png (coverage @ -65dBm threshold: {coverage:.1f}%)")

    print(f"\nAll visualizations generated in {OUT_DIR}/")
