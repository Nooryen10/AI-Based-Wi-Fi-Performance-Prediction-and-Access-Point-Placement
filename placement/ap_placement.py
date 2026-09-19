"""
STEP 5: ACCESS POINT PLACEMENT (CORE PART OF THE PROJECT)
=============================================================
Unlike the simple Step-4 demo heatmap (which faked "walls" as a function of distance),
this module builds an actual 2D FLOOR PLAN with real internal wall segments, and computes
signal strength at every grid cell using RAY-BASED wall counting: for a candidate router
position and a target cell, we count how many wall segments the straight line between them
actually crosses (classic 2D computational-geometry segment-intersection test), then plug
that wall count into the same log-distance + wall-attenuation path loss model used in the
dataset generator. This is what real Wi-Fi planning tools (e.g. Ekahau, NetSpot) do at a
simplified level.

Pipeline:
  1. Define a floor plan (room boundary + internal walls) as line segments
  2. For a given router position, compute signal strength at every grid cell
     (accounting for the real walls crossed by each router->cell ray)
  3. Define "coverage" = % of grid cells with signal >= threshold (-65 dBm default)
  4. Search candidate router positions (grid search, walls-aware) and pick the one
     that MAXIMIZES coverage  ->  this is the placement recommendation
  5. Compare "naive" placement (e.g., corner, common real-world mistake) vs "optimized"
     placement, and 2.4GHz vs 5GHz for the same floor plan
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"

# ---------------------------------------------------------------------------
# 1. FLOOR PLAN DEFINITION
# ---------------------------------------------------------------------------
# A 35m x 25m apartment-style floor plan (a realistic 2-3BHK flat): outer boundary +
# internal walls creating 5 rooms around a central hallway, with doorway gaps
# (deliberately left open in the wall segments, so it's not a fully sealed box).
# Sized so that far rooms genuinely fall into a weak-signal / dead-zone regime with a
# corner-placed router -- which is what makes the "before vs after" comparison meaningful.
ROOM_W, ROOM_H = 35.0, 25.0

WALLS = [
    # Outer boundary
    ((0, 0), (ROOM_W, 0)),
    ((ROOM_W, 0), (ROOM_W, ROOM_H)),
    ((ROOM_W, ROOM_H), (0, ROOM_H)),
    ((0, ROOM_H), (0, 0)),

    # Central horizontal hallway wall (separates bottom rooms from top rooms)
    # with two doorway gaps: [12-14] and [23-25]
    ((0, 12.5), (12, 12.5)),
    ((14, 12.5), (23, 12.5)),
    ((25, 12.5), (ROOM_W, 12.5)),

    # Bottom section: split into 3 rooms (each with a doorway gap into the hallway zone)
    ((12, 0), (12, 10)),          # divider 1 (gap 10-12.5 = doorway)
    ((23, 0), (23, 10)),          # divider 2 (gap 10-12.5 = doorway)

    # Top section: split into 2 larger rooms (bedrooms)
    ((17.5, 12.5), (17.5, 22)),   # divider (gap 22-25 = doorway)

    # Small bathroom block in the top-right corner
    ((28, 15), (28, ROOM_H)),
    ((28, 15), (ROOM_W, 15)),
]


class interference_db:
    """Represent interference in dB and convert it to signal attenuation.

    ``value`` may be a scalar or a NumPy array, which makes the class useful
    for both a uniform interference floor and a spatial interference map.
    """

    def __init__(self, value=0.0, attenuation_factor=0.6):
        self.value = np.asarray(value, dtype=float)
        self.attenuation_factor = float(attenuation_factor)
        if self.attenuation_factor < 0:
            raise ValueError("attenuation_factor must be non-negative")

    def attenuation(self, shape=None):
        """Return the equivalent signal penalty in dB."""
        penalty = self.attenuation_factor * self.value
        return np.broadcast_to(penalty, shape) if shape is not None else penalty

    def __float__(self):
        if self.value.size != 1:
            raise TypeError("an interference map cannot be converted to a scalar")
        return float(self.attenuation())

    def __array__(self, dtype=None):
        return np.asarray(self.attenuation(), dtype=dtype)


def path_loss_signal(distance, walls, band="2.4GHz", interference_db=0.0):
    """Same physics as data/generate_dataset.py -- kept identical so ML predictions
    and the geometric simulation are consistent with each other."""
    distance = np.maximum(distance, 0.1)
    if band == "5GHz":
        n, waf, tx = 3.2, 5.5, 23.0
    else:
        n, waf, tx = 2.7, 3.5, 20.0
    pl0, d0 = 40.0, 1.0
    path_loss = pl0 + 10 * n * np.log10(distance / d0) + walls * waf
    return tx - path_loss - 0.6 * interference_db


def _cross(o, a, b):
    """2D cross product of (a-o) and (b-o), vectorized over arrays of points."""
    return (a[..., 0] - o[0]) * (b[..., 1] - o[1]) - (a[..., 1] - o[1]) * (b[..., 0] - o[0])


def count_wall_crossings(router_xy, cell_points, walls=None):
    """
    Vectorized ray-based wall counting.
    router_xy: (2,) fixed point
    cell_points: (N, 2) array of target grid points
    walls: list of ((x1,y1),(x2,y2)) segments. IMPORTANT: defaults to None (not the module-level
        WALLS list directly) and is resolved to the CURRENT module-level WALLS at call time --
        this matters because app/interface.py rescales the floor plan for custom room sizes by
        reassigning ap_placement.WALLS at runtime; binding a mutable default at function-definition
        time would silently keep using the ORIGINAL 35x25 floor plan geometry after that rescale.
    Returns: (N,) array -- number of walls crossed on the straight line router->each cell
    """
    if walls is None:
        walls = WALLS
    p = np.asarray(router_xy, dtype=float)
    q = cell_points  # (N, 2)
    crossing_count = np.zeros(q.shape[0], dtype=int)

    for (r, s) in walls:
        r = np.asarray(r, dtype=float)
        s = np.asarray(s, dtype=float)

        # Standard orientation-based segment intersection test (router-cell line vs wall):
        # line (r,s) straddles line (p,q) AND line (p,q) straddles line (r,s)
        d1 = _cross(r, s, np.broadcast_to(p, q.shape))
        d2 = _cross(r, s, q)
        d3 = _cross(p, q, np.broadcast_to(r, q.shape))
        d4 = _cross(p, q, np.broadcast_to(s, q.shape))

        seg_intersects = (np.sign(d1) != np.sign(d2)) & (np.sign(d3) != np.sign(d4))
        crossing_count += seg_intersects.astype(int)

    return crossing_count


def build_grid(resolution=0.5):
    x = np.arange(resolution / 2, ROOM_W, resolution)
    y = np.arange(resolution / 2, ROOM_H, resolution)
    X, Y = np.meshgrid(x, y)
    points = np.column_stack([X.ravel(), Y.ravel()])
    return X, Y, points


def simulate_coverage(router_xy, band="2.4GHz", resolution=0.5, threshold=-65.0,
            interference_db=0.0):
    """Returns (X, Y, signal_grid_2d, coverage_percent) for a given router position."""
    X, Y, points = build_grid(resolution)
    dist = np.linalg.norm(points - np.asarray(router_xy), axis=1)
    walls_crossed = count_wall_crossings(router_xy, points)
    signal = path_loss_signal(dist, walls_crossed, band=band, interference_db=interference_db)
    signal_2d = signal.reshape(X.shape)
    coverage_pct = 100.0 * np.mean(signal >= threshold)
    return X, Y, signal_2d, coverage_pct


NOISE_FLOOR_DBM = -95.0   # typical Wi-Fi receiver noise floor


def _dbm_to_mw(dbm):
    return 10.0 ** (np.asarray(dbm, dtype=float) / 10.0)


def _mw_to_dbm(mw):
    return 10.0 * np.log10(np.maximum(mw, 1e-12))


def simulate_coverage_multi(router_list, band="2.4GHz", resolution=0.5, threshold=-65.0,
                        interference_db=0.0, interference_aware=False, sinr_threshold_db=9.0,
                        noise_floor_dbm=NOISE_FLOOR_DBM):
    """
    Multi-AP version of simulate_coverage(). Each grid cell is served by whichever AP
    gives it the STRONGEST signal (best-server association -- the same assumption real
    Wi-Fi clients use when roaming between APs on the same SSID).

    SECOND RESEARCH-GAP EXTENSION -- interference_aware=True:
    By default (interference_aware=False, unchanged from before) a cell counts as "covered"
    purely because SOME AP's signal there clears the absolute RSSI threshold. That is exactly
    the gap most coverage-only multi-AP placement tools (including this project's own first
    extension, find_optimal_multi_ap_placement) have: nothing stops the algorithm from
    happily stacking two APs close together, because from a pure "is anyone loud enough here"
    view that looks fine -- even though in real deployments those two nearby APs would very
    likely share the same Wi-Fi channel (2.4GHz has only 3 non-overlapping channels; dense
    deployments run out of clean channels fast) and so would CO-CHANNEL INTERFERE with each
    other, degrading real-world throughput at exactly the point where their coverage overlaps.

    When interference_aware=True, every placed AP is conservatively treated as being on the
    SAME channel as every other one (the simple, worst-case assumption -- this project doesn't
    do channel planning). At each grid cell we compute:
        SINR(dB) = 10*log10( P_serving_mW / (sum of P_other_APs_mW + P_noise_floor_mW) )
    and a cell only counts as covered if it ALSO clears `sinr_threshold_db` (9 dB by default,
    a reasonable minimum for a good modulation/coding rate), on top of the usual absolute RSSI
    threshold. This makes the placement search naturally spread APs out instead of clustering
    them, since clustering now measurably lowers SINR without buying any extra coverage.

    Returns: (X, Y, best_signal_2d, serving_ap_index_2d, coverage_percent, sinr_2d_or_None)
    """
    X, Y, points = build_grid(resolution)
    n_points = points.shape[0]
    n_aps = len(router_list)
    signals_mw = np.zeros((n_aps, n_points))

    for k, router_xy in enumerate(router_list):
        dist = np.linalg.norm(points - np.asarray(router_xy), axis=1)
        walls_crossed = count_wall_crossings(router_xy, points)
        signal_dbm = path_loss_signal(dist, walls_crossed, band=band, interference_db=interference_db)
        signals_mw[k] = _dbm_to_mw(signal_dbm)

    serving_ap = np.argmax(signals_mw, axis=0)
    best_signal_mw = signals_mw[serving_ap, np.arange(n_points)]
    best_signal_dbm = _mw_to_dbm(best_signal_mw)

    sinr_2d = None
    if interference_aware and n_aps > 1:
        total_mw = signals_mw.sum(axis=0)
        co_channel_interference_mw = total_mw - best_signal_mw + _dbm_to_mw(noise_floor_dbm)
        sinr_db = 10.0 * np.log10(best_signal_mw / co_channel_interference_mw)
        covered = (best_signal_dbm >= threshold) & (sinr_db >= sinr_threshold_db)
        sinr_2d = sinr_db.reshape(X.shape)
    else:
        covered = best_signal_dbm >= threshold

    signal_2d = best_signal_dbm.reshape(X.shape)
    serving_2d = serving_ap.reshape(X.shape)
    coverage_pct = 100.0 * np.mean(covered)
    return X, Y, signal_2d, serving_2d, coverage_pct, sinr_2d


def find_optimal_multi_ap_placement(num_aps=2, band="2.4GHz", resolution=0.5,
                                candidate_step=1.0, threshold=-65.0, interference_aware=False,
                                sinr_threshold_db=9.0):
    """
    RESEARCH-GAP EXTENSION: multi-AP joint placement optimization.

    Limitation being addressed: the single-AP exhaustive search above (find_optimal_placement)
    is the standard approach used by most "AI Wi-Fi placement" course projects, but it cannot
    help once a floor plan is too large / has too many walls for ONE access point to cover
    adequately (e.g. 5GHz on this floor plan tops out at ~94% coverage no matter where the
    single AP goes -- see the BAND COMPARISON output). Real deployments (mesh systems,
    enterprise WLAN) always use multiple coordinated APs. Jointly optimizing K AP positions
    by brute force is combinatorially expensive (O(candidates^K)), so this implements the
    standard, well-studied approximation for the "maximal coverage location problem":
    GREEDY SEQUENTIAL PLACEMENT with re-evaluation --

    1. Place AP #1 at the position that maximizes coverage alone (exactly the Step-5 search).
    2. Place AP #2 at the position that maximizes the MARGINAL coverage gain given AP #1 is
    already there (i.e. considers best-server signal = max(AP1, AP2) at every cell, so it
    naturally targets the dead zones AP #1 couldn't reach instead of overlapping it).
    3. Repeat for AP #3, #4, ... up to num_aps.

    This greedy strategy is a widely-used, provably-reasonable heuristic (submodular set-cover
    style guarantee: greedy achieves >= (1 - 1/e) ~= 63% of the true joint optimum for coverage
    maximization) and stays fast enough to run interactively, unlike full joint brute force.

    interference_aware=True switches the coverage metric used at every greedy step to the
    SINR-aware definition in simulate_coverage_multi (see its docstring for the full
    rationale) -- this is the SECOND research-gap extension: it stops the greedy search from
    treating two overlapping/adjacent APs as "free" extra coverage when they would actually
    co-channel-interfere with each other in a real deployment.

    Returns: (list_of_ap_positions, final_coverage_pct, per_ap_marginal_gain_list)
    """
    candidates_x = np.arange(1.0, ROOM_W, candidate_step)
    candidates_y = np.arange(1.0, ROOM_H, candidate_step)
    candidate_positions = [(cx, cy) for cy in candidates_y for cx in candidates_x]

    placed_aps = []
    coverage_history = []
    prev_coverage = 0.0

    for ap_index in range(num_aps):
        best_gain = -np.inf
        best_candidate = None
        best_coverage_with_candidate = prev_coverage

        for cand in candidate_positions:
            trial_list = placed_aps + [cand]
            _, _, _, _, cov, _ = simulate_coverage_multi(
                trial_list, band=band, resolution=resolution, threshold=threshold,
                interference_aware=interference_aware, sinr_threshold_db=sinr_threshold_db)
            gain = cov - prev_coverage
            if gain > best_gain:
                best_gain = gain
                best_candidate = cand
                best_coverage_with_candidate = cov

        # Stop adding APs once even the best remaining candidate no longer helps (this can
        # happen with interference_aware=True: cramming in another AP can lower SINR for
        # everyone without gaining any new area, so its marginal "gain" goes negative).
        if best_candidate is None or best_gain <= 0:
            break

        best_candidate = (float(best_candidate[0]), float(best_candidate[1]))
        placed_aps.append(best_candidate)
        coverage_history.append(round(best_coverage_with_candidate, 1))
        prev_coverage = best_coverage_with_candidate

        # Early stop once we've already hit (near) full coverage -- no point adding more APs
        if prev_coverage >= 99.9:
            break

    marginal_gains = [coverage_history[0]] + [round(coverage_history[i] - coverage_history[i - 1], 1)
                                            for i in range(1, len(coverage_history))]
    return placed_aps, coverage_history[-1], marginal_gains


def plot_coverage_multi(router_list, band, title, save_name, threshold=-65.0,
                         interference_aware=False, sinr_threshold_db=9.0):
    """Multi-AP coverage plot -- shades each grid cell by its best-server signal and marks
    every AP with a distinct colored star so overlapping coverage cells are visible."""
    X, Y, signal_2d, serving_2d, coverage_pct, sinr_2d = simulate_coverage_multi(
        router_list, band=band, threshold=threshold, interference_aware=interference_aware,
        sinr_threshold_db=sinr_threshold_db)

    fig, ax = plt.subplots(figsize=(9, 7))
    hm = ax.pcolormesh(X, Y, signal_2d, cmap="RdYlGn", vmin=-90, vmax=-30, shading="auto")
    plt.colorbar(hm, ax=ax, label="Best-Server Signal Strength (dBm)")
    ax.contour(X, Y, signal_2d, levels=[threshold], colors="blue", linewidths=2)
    if interference_aware and sinr_2d is not None:
        ax.contour(X, Y, sinr_2d, levels=[sinr_threshold_db], colors="red", linewidths=2,
                   linestyles="dashed")
    plot_floorplan(ax)

    colors = ["black", "purple", "darkorange", "deeppink", "navy"]
    for i, pos in enumerate(router_list):
        ax.scatter(*pos, marker="*", s=550, c=colors[i % len(colors)], edgecolors="white",
                zorder=5, label=f"AP {i + 1} {tuple(round(v, 1) for v in pos)}")

    mode_label = " (interference-aware)" if interference_aware else ""
    ax.set_title(f"{title}\nJoint Coverage{mode_label} @ {threshold:.0f} dBm threshold: "
                f"{coverage_pct:.1f}% ({len(router_list)} AP{'s' if len(router_list) != 1 else ''})")
    ax.set_xlabel("Room Width (m)")
    ax.set_ylabel("Room Height (m)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{save_name}", dpi=150)
    plt.close()
    return coverage_pct


def find_optimal_placement(band="2.4GHz", resolution=0.5, candidate_step=1.0, threshold=-65.0):
    """
    Grid-search optimization: evaluate coverage for router positions spaced `candidate_step`
    meters apart across the whole floor plan (skipping positions that fall exactly ON a wall),
    and return the position that maximizes % coverage at the given signal threshold.

    This is a brute-force / exhaustive-grid-search approach -- appropriate here because the
    search space is small (candidate positions are typically a few hundred), it's guaranteed
    to find the best position ON the evaluated grid (unlike a greedy hill-climb which can get
    stuck in local optima), and it doubles as the basis for a heatmap of "coverage achievable
    from every possible AP position" which is itself a useful diagnostic visualization.
    """
    candidates_x = np.arange(1.0, ROOM_W, candidate_step)
    candidates_y = np.arange(1.0, ROOM_H, candidate_step)

    best_coverage = -1
    best_pos = None
    coverage_map = np.zeros((len(candidates_y), len(candidates_x)))

    for i, cy in enumerate(candidates_y):
        for j, cx in enumerate(candidates_x):
            _, _, _, coverage = simulate_coverage((cx, cy), band=band, resolution=resolution,
                                                    threshold=threshold)
            coverage_map[i, j] = coverage
            if coverage > best_coverage:
                best_coverage = coverage
                best_pos = (cx, cy)

    return best_pos, best_coverage, (candidates_x, candidates_y, coverage_map)


def plot_floorplan(ax):
    for (x1, y1), (x2, y2) in WALLS:
        ax.plot([x1, x2], [y1, y2], color="black", linewidth=3, solid_capstyle="round")


def plot_coverage(router_xy, band, title, save_name, threshold=-65.0):
    X, Y, signal_2d, coverage_pct = simulate_coverage(router_xy, band=band, threshold=threshold)

    fig, ax = plt.subplots(figsize=(9, 7))
    hm = ax.pcolormesh(X, Y, signal_2d, cmap="RdYlGn", vmin=-90, vmax=-30, shading="auto")
    plt.colorbar(hm, ax=ax, label="Signal Strength (dBm)")
    ax.contour(X, Y, signal_2d, levels=[threshold], colors="blue", linewidths=2)
    plot_floorplan(ax)
    ax.scatter(*router_xy, marker="*", s=500, c="black", edgecolors="white", zorder=5,
            label="Router (AP)")
    ax.set_title(f"{title}\nCoverage @ {threshold:.0f} dBm threshold: {coverage_pct:.1f}%")
    ax.set_xlabel("Room Width (m)")
    ax.set_ylabel("Room Height (m)")
    ax.legend(loc="upper right")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{save_name}", dpi=150)
    plt.close()
    return coverage_pct


def plot_optimization_surface(candidates_x, candidates_y, coverage_map, best_pos, band,
                            save_name):
    fig, ax = plt.subplots(figsize=(9, 7))
    hm = ax.pcolormesh(candidates_x, candidates_y, coverage_map, cmap="viridis", shading="auto")
    plt.colorbar(hm, ax=ax, label="Achievable Coverage (%)")
    plot_floorplan(ax)
    ax.scatter(*best_pos, marker="*", s=500, c="red", edgecolors="white", zorder=5,
            label=f"Optimal AP position {tuple(round(v,1) for v in best_pos)}")
    ax.set_title(f"AP Placement Search Surface ({band})\n"
                  f"Color = coverage% achievable if router placed at that point")
    ax.set_xlabel("Room Width (m)")
    ax.set_ylabel("Room Height (m)")
    ax.legend(loc="upper right")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{save_name}", dpi=150)
    plt.close()

def plot_before_after_comparison(naive_pos, naive_cov, best_pos, best_cov,
                                   band="2.4GHz", threshold=-65.0,
                                   save_name="11_before_after_comparison.png"):
    """Create a side-by-side BEFORE vs AFTER coverage visualization."""

    X1, Y1, signal_before, _ = simulate_coverage(
        naive_pos, band=band, threshold=threshold
    )
    X2, Y2, signal_after, _ = simulate_coverage(
        best_pos, band=band, threshold=threshold
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    hm1 = axes[0].pcolormesh(
        X1, Y1, signal_before, cmap="RdYlGn",
        vmin=-90, vmax=-30, shading="auto"
    )
    axes[0].contour(
        X1, Y1, signal_before, levels=[threshold],
        colors="blue", linewidths=2
    )
    plot_floorplan(axes[0])
    axes[0].scatter(
        *naive_pos, marker="*", s=500, c="black",
        edgecolors="white", zorder=5
    )
    axes[0].set_title(
        f"BEFORE: Naive Placement\n"
        f"AP: {naive_pos} | Coverage: {naive_cov:.1f}%"
    )
    axes[0].set_xlabel("Room Width (m)")
    axes[0].set_ylabel("Room Height (m)")
    axes[0].set_aspect("equal")

    hm2 = axes[1].pcolormesh(
        X2, Y2, signal_after, cmap="RdYlGn",
        vmin=-90, vmax=-30, shading="auto"
    )
    axes[1].contour(
        X2, Y2, signal_after, levels=[threshold],
        colors="blue", linewidths=2
    )
    plot_floorplan(axes[1])
    axes[1].scatter(
        *best_pos, marker="*", s=500, c="black",
        edgecolors="white", zorder=5
    )
    axes[1].set_title(
        f"AFTER: Optimized Placement\n"
        f"AP: {best_pos} | Coverage: {best_cov:.1f}%"
    )
    axes[1].set_xlabel("Room Width (m)")
    axes[1].set_ylabel("Room Height (m)")
    axes[1].set_aspect("equal")

    fig.colorbar(
        hm2, ax=axes, label="Signal Strength (dBm)", shrink=0.85
    )

    improvement = best_cov - naive_cov
    fig.suptitle(
        f"AP Placement Before vs After ({band})\n"
        f"Coverage Improvement: +{improvement:.1f} percentage points "
        f"at {threshold:.0f} dBm threshold",
        fontsize=14
    )

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{save_name}", dpi=150)
    plt.close()
    return save_name


def generate_placement_report(naive_pos, naive_cov, best_pos, best_cov,
                            band="2.4GHz", threshold=-65.0):
    """
    Generate a clear before-vs-after AP placement report.
    """

    improvement = best_cov - naive_cov

    print("\n" + "=" * 70)
    print("AP PLACEMENT RECOMMENDATION REPORT")
    print("=" * 70)

    print(f"Frequency Band       : {band}")
    print(f"Signal Threshold     : {threshold:.0f} dBm")
    print(f"Naive AP Position    : {naive_pos}")
    print(f"Naive Coverage       : {naive_cov:.1f}%")
    print(f"Recommended Position : {best_pos}")
    print(f"Optimized Coverage   : {best_cov:.1f}%")
    print(f"Coverage Improvement : +{improvement:.1f} percentage points")

    print("-" * 70)

    if improvement > 0:
        print(
            f"RECOMMENDATION: Place the AP near {best_pos} "
            f"to maximize usable Wi-Fi coverage."
        )
    else:
        print("RECOMMENDATION: Current AP placement is already optimal.")

    print("=" * 70)

    # Save the report as a text file
    report_path = OUT_DIR / "ap_placement_recommendation.txt"

    with open(report_path, "w") as f:
        f.write("AP PLACEMENT RECOMMENDATION REPORT\n")
        f.write("=" * 50 + "\n")
        f.write(f"Frequency Band: {band}\n")
        f.write(f"Signal Threshold: {threshold:.0f} dBm\n")
        f.write(f"Naive AP Position: {naive_pos}\n")
        f.write(f"Naive Coverage: {naive_cov:.1f}%\n")
        f.write(f"Recommended AP Position: {best_pos}\n")
        f.write(f"Optimized Coverage: {best_cov:.1f}%\n")
        f.write(f"Coverage Improvement: +{improvement:.1f} percentage points\n")
        f.write("\nRecommendation:\n")
        f.write(
            f"Place the AP near {best_pos} to maximize "
            f"usable Wi-Fi coverage.\n"
        )

    print(f"Saved: {report_path.name}")

if __name__ == "__main__":
    THRESHOLD = -65.0

    print("=" * 70)
    print("STEP 5: ACCESS POINT PLACEMENT OPTIMIZATION")
    print("=" * 70)

    naive_pos = (2.0, 2.0)
    naive_cov = plot_coverage(
        naive_pos, "2.4GHz",
        f"BEFORE: Naive Corner Placement — Router at {naive_pos}",
        "05_before_naive_placement.png", threshold=THRESHOLD)
    print(f"\n[BEFORE] Naive corner placement {naive_pos}: {naive_cov:.1f}% coverage (2.4GHz)")

    best_pos, best_cov, (cx, cy, cov_map) = find_optimal_placement(
        band="2.4GHz", candidate_step=1.0, threshold=THRESHOLD)
    best_pos = (float(best_pos[0]), float(best_pos[1]))
    plot_coverage(
        best_pos, "2.4GHz",
        f"AFTER: Optimized Placement — Router at {tuple(round(v, 2) for v in best_pos)}",
        "06_after_optimized_placement.png", threshold=THRESHOLD)
    print(f"[AFTER]  Optimized placement {tuple(round(v,2) for v in best_pos)}: {best_cov:.1f}% coverage (2.4GHz)")
    print(f"[IMPROVEMENT] +{best_cov - naive_cov:.1f} percentage points of coverage")

    generate_placement_report(
        naive_pos, naive_cov, best_pos, best_cov,
        band="2.4GHz", threshold=THRESHOLD)

    # ---- NEW: side-by-side BEFORE vs AFTER visualization ----
    comparison_file = plot_before_after_comparison(
        naive_pos,
        naive_cov,
        best_pos,
        best_cov,
        band="2.4GHz",
        threshold=THRESHOLD
    )
    print(f"Saved: {comparison_file} (before vs after coverage comparison)")

    plot_optimization_surface(
        cx, cy, cov_map, best_pos, "2.4GHz",
        "07_placement_search_surface.png")
    print("Saved: 07_placement_search_surface.png (coverage achievable from every candidate AP spot)")

    best_pos_5g, best_cov_5g, _ = find_optimal_placement(
        band="5GHz", candidate_step=1.0, threshold=THRESHOLD)
    best_pos_5g = (float(best_pos_5g[0]), float(best_pos_5g[1]))
    plot_coverage(
        best_pos_5g, "5GHz",
        f"AFTER: Optimized Placement — Router at {tuple(round(v,2) for v in best_pos_5g)}",
        "08_optimized_placement_5ghz.png", threshold=THRESHOLD)
    print("\n[BAND COMPARISON] Best achievable coverage:")
    print(f"   2.4GHz -> {best_cov:.1f}% at optimal position {tuple(round(v,2) for v in best_pos)}")
    print(f"   5GHz   -> {best_cov_5g:.1f}% at optimal position {tuple(round(v,2) for v in best_pos_5g)}")
    print("   (5GHz has shorter range/more wall loss -> typically needs the AP more centrally")
    print("    located, or multiple APs, to match 2.4GHz's coverage footprint)")

    with open(f"{OUT_DIR}/placement_summary.txt", "w") as f:
        f.write("ACCESS POINT PLACEMENT OPTIMIZATION SUMMARY\n")
        f.write("=" * 50 + "\n")
        f.write(f"Floor plan: {ROOM_W}m x {ROOM_H}m, 4 rooms + corridor, {len(WALLS)} wall segments\n")
        f.write(f"Coverage threshold: {THRESHOLD} dBm\n\n")
        f.write(f"BEFORE (naive corner placement {naive_pos}): {naive_cov:.1f}% coverage\n")
        f.write(f"AFTER  (optimized placement {tuple(round(v,2) for v in best_pos)}): {best_cov:.1f}% coverage\n")
        f.write(f"Improvement: +{best_cov - naive_cov:.1f} percentage points\n\n")
        f.write(f"2.4GHz optimal position: {tuple(round(v,2) for v in best_pos)} -> {best_cov:.1f}% coverage\n")
        f.write(f"5GHz optimal position:   {tuple(round(v,2) for v in best_pos_5g)} -> {best_cov_5g:.1f}% coverage\n")
    print("\nSaved: placement_summary.txt")

    print("\n" + "=" * 70)
    print("RESEARCH-GAP EXTENSION: MULTI-AP GREEDY JOINT PLACEMENT (5GHz)")
    print("=" * 70)
    multi_positions, multi_cov, marginal_gains = find_optimal_multi_ap_placement(
        num_aps=2, band="5GHz", candidate_step=1.0, threshold=THRESHOLD)
    plot_coverage_multi(
        multi_positions, "5GHz",
        f"Multi-AP Greedy Placement (5GHz, {len(multi_positions)} APs)",
        "10_multi_ap_placement_5ghz.png", threshold=THRESHOLD)
    for i, (pos, gain) in enumerate(zip(multi_positions, marginal_gains)):
        print(f"  AP {i + 1} -> position {tuple(round(v, 2) for v in pos)}  (+{gain:.1f} pts marginal coverage gain)")
    print(f"  Joint coverage with {len(multi_positions)} APs: {multi_cov:.1f}%  (vs {best_cov_5g:.1f}% with a single 5GHz AP -> +{multi_cov - best_cov_5g:.1f} points)")
    print("Saved: 10_multi_ap_placement_5ghz.png")
    with open(f"{OUT_DIR}/placement_summary.txt", "a") as f:
        f.write("\nRESEARCH-GAP EXTENSION: MULTI-AP GREEDY JOINT PLACEMENT (5GHz)\n")
        f.write("-" * 50 + "\n")
        for i, (pos, gain) in enumerate(zip(multi_positions, marginal_gains)):
            f.write(f"AP {i + 1}: {tuple(round(v, 2) for v in pos)} (+{gain:.1f} pts)\n")
        f.write(f"Joint coverage: {multi_cov:.1f}% vs single-AP 5GHz {best_cov_5g:.1f}% (+{multi_cov - best_cov_5g:.1f} points)\n")

    # ---- SECOND RESEARCH-GAP EXTENSION: interference-aware multi-AP placement ----
    # With only 2 APs on this floor plan there isn't much room for them to sit close enough
    # to meaningfully interfere, so this comparison uses 4 APs (more APs -> greedy has more
    # opportunity to cluster them if nothing is penalizing that, which is exactly the failure
    # mode interference-awareness is meant to catch).
    print("\n" + "=" * 70)
    print("SECOND RESEARCH-GAP EXTENSION: INTERFERENCE-AWARE MULTI-AP PLACEMENT (5GHz, 4 APs)")
    print("=" * 70)
    coverage_only_positions, coverage_only_cov, _ = find_optimal_multi_ap_placement(
        num_aps=4, band="5GHz", candidate_step=1.0, threshold=THRESHOLD, interference_aware=False)
    # Using a relaxed 3 dB minimum SINR bar for this demo (rather than the function's default
    # 9 dB "good/reliable" bar) -- this represents accepting a weaker but still usable link,
    # and produces a more informative comparison: instead of just concluding "use fewer APs",
    # it shows the greedy search actively choosing to SPREAD the APs apart to keep interference
    # manageable, which is the more general and more interesting behavior change to demonstrate.
    demo_sinr_threshold = 3.0
    interference_positions, interference_cov, _ = find_optimal_multi_ap_placement(
        num_aps=4, band="5GHz", candidate_step=1.0, threshold=THRESHOLD, interference_aware=True,
        sinr_threshold_db=demo_sinr_threshold)

    # Re-measure BOTH placements under the interference-aware metric so they're compared
    # on the same yardstick (a coverage-only placement can look great on paper and still
    # perform worse once real co-channel interference between nearby APs is accounted for).
    _, _, _, _, coverage_only_cov_under_sinr, _ = simulate_coverage_multi(
        coverage_only_positions, band="5GHz", threshold=THRESHOLD, interference_aware=True,
        sinr_threshold_db=demo_sinr_threshold)

    plot_coverage_multi(coverage_only_positions, "5GHz",
                         "Coverage-Only Placement (ignores AP-to-AP interference)",
                         "12_coverage_only_vs_interference_aware_A_coverage_only.png",
                         threshold=THRESHOLD, interference_aware=True,
                         sinr_threshold_db=demo_sinr_threshold)
    plot_coverage_multi(interference_positions, "5GHz",
                         "Interference-Aware Placement (penalizes co-channel overlap)",
                         "12_coverage_only_vs_interference_aware_B_interference_aware.png",
                         threshold=THRESHOLD, interference_aware=True,
                         sinr_threshold_db=demo_sinr_threshold)

    print(f"  Coverage-only search's own metric:        {coverage_only_cov:.1f}% "
          f"(4 APs placed at {[tuple(round(v,1) for v in p) for p in coverage_only_positions]})")
    print(f"  ...but under a real SINR/interference check, that same placement only achieves: "
          f"{coverage_only_cov_under_sinr:.1f}%")
    print(f"  Interference-aware search (optimizes for SINR from the start): "
          f"{interference_cov:.1f}% (4 APs placed at "
          f"{[tuple(round(v,1) for v in p) for p in interference_positions]})")
    print(f"  Gain from being interference-aware: +{interference_cov - coverage_only_cov_under_sinr:.1f} "
          f"points of REAL (interference-limited) coverage")
    print("Saved: 12_coverage_only_vs_interference_aware_{A_coverage_only,B_interference_aware}.png")

    with open(f"{OUT_DIR}/placement_summary.txt", "a") as f:
        f.write("\nSECOND RESEARCH-GAP EXTENSION: INTERFERENCE-AWARE MULTI-AP PLACEMENT (5GHz, 4 APs)\n")
        f.write("-" * 50 + "\n")
        f.write(f"Coverage-only placement:      {coverage_only_cov:.1f}% by its own (interference-blind) "
                f"metric, but only {coverage_only_cov_under_sinr:.1f}% once real co-channel SINR is checked\n")
        f.write(f"Interference-aware placement: {interference_cov:.1f}% real (SINR-checked) coverage\n")
        f.write(f"Gain from interference-awareness: +{interference_cov - coverage_only_cov_under_sinr:.1f} points\n")
