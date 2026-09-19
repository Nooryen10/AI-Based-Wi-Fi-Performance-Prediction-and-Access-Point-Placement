"""
STEP 6: USER INPUT SYSTEM (CLI)
==================================
Command-line tool that ties the whole pipeline together for an end user:

  INPUT:  room width/height (m), number of walls between AP and a point of interest,
          number of concurrent users, interference level, frequency band
  OUTPUT: - ML-predicted signal strength / throughput / latency at that point
          - Recommended AP (x, y) position for the given room, computed by running
            the Step-5 placement optimizer on a floor plan sized to the given room

Usage (interactive):
    python3 app/interface.py

Usage (non-interactive, for scripting/demo):
    python3 app/interface.py --room_w 30 --room_h 20 --walls 2 --users 15 \
        --interference medium --band 2.4GHz

Usage (research-gap extension, multi-AP joint placement):
    python3 app/interface.py --room_w 30 --room_h 20 --walls 2 --users 15 \
        --interference medium --band 5GHz --num_aps 2
"""

import argparse
import sys
import numpy as np
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from preprocessing.preprocess import NUMERIC_FEATURES, CATEGORICAL_FEATURES
from placement.ap_placement import (find_optimal_placement, find_optimal_multi_ap_placement,
                                     WALLS as DEFAULT_WALLS)

MODEL_DIR = PROJECT_ROOT / "models"
PREP_ARTIFACTS_PATH = PROJECT_ROOT / "preprocessing" / "preprocessing_artifacts.pkl"

TARGETS = ["signal_strength_dbm", "throughput_mbps", "latency_ms"]


def load_artifacts():
    artifacts = joblib.load(PREP_ARTIFACTS_PATH)
    models = {t: joblib.load(f"{MODEL_DIR}/best_model_{t}.pkl") for t in TARGETS}
    return artifacts, models


def predict_performance(distance_m, num_walls, num_users, interference_level, frequency_band,
                         artifacts, models):
    import pandas as pd
    row = pd.DataFrame([{
        "distance_m": distance_m,
        "num_walls": num_walls,
        "num_users": num_users,
        "interference_level": interference_level,
        "frequency_band": frequency_band,
    }])
    X_processed = artifacts["preprocessor"].transform(row)
    predictions = {t: float(models[t].predict(X_processed)[0]) for t in TARGETS}
    return predictions


def _apply_scaled_floorplan(room_w, room_h):
    """Re-scales the default floor plan's internal wall layout proportionally to the user's
    given room dimensions and monkey-patches the module-level ROOM_W/ROOM_H/WALLS used by
    the placement search functions. Returns the placement module for convenience."""
    scale_x = room_w / 35.0
    scale_y = room_h / 25.0
    scaled_walls = [((x1 * scale_x, y1 * scale_y), (x2 * scale_x, y2 * scale_y))
                     for (x1, y1), (x2, y2) in DEFAULT_WALLS]

    import placement.ap_placement as ap
    ap.ROOM_W, ap.ROOM_H, ap.WALLS = room_w, room_h, scaled_walls
    return ap


def recommend_placement(room_w, room_h, band="2.4GHz", threshold=-65.0):
    """Single-AP placement recommendation (original Step-5 behaviour)."""
    _apply_scaled_floorplan(room_w, room_h)
    best_pos, best_cov, _ = find_optimal_placement(band=band, candidate_step=max(1.0, room_w / 25),
                                                     threshold=threshold)
    return (round(float(best_pos[0]), 2), round(float(best_pos[1]), 2)), round(best_cov, 1)


def recommend_multi_ap_placement(room_w, room_h, num_aps=2, band="2.4GHz", threshold=-65.0,
                                  interference_aware=False, sinr_threshold_db=9.0):
    """
    RESEARCH-GAP EXTENSION: multi-AP placement recommendation.
    Greedily places `num_aps` access points to jointly maximize coverage -- see the detailed
    rationale in placement/ap_placement.py::find_optimal_multi_ap_placement.

    interference_aware=True switches on the SECOND research-gap extension: it additionally
    penalizes placing APs so close together that they would co-channel-interfere with each
    other in a real deployment (see find_optimal_multi_ap_placement's docstring). With this
    on, the search may recommend fewer or more spread-out APs than the coverage-only mode.
    """
    _apply_scaled_floorplan(room_w, room_h)
    positions, final_cov, marginal_gains = find_optimal_multi_ap_placement(
        num_aps=num_aps, band=band, candidate_step=max(1.0, room_w / 25), threshold=threshold,
        interference_aware=interference_aware, sinr_threshold_db=sinr_threshold_db)
    positions = [(round(p[0], 2), round(p[1], 2)) for p in positions]
    return positions, round(final_cov, 1), marginal_gains


def interpret_results(preds):
    """Plain-English interpretation -- useful both for the CLI user and for the viva."""
    sig, thr, lat = preds["signal_strength_dbm"], preds["throughput_mbps"], preds["latency_ms"]
    lines = []

    if sig >= -50:
        lines.append(f"Signal ({sig:.1f} dBm): Excellent -- near the router.")
    elif sig >= -65:
        lines.append(f"Signal ({sig:.1f} dBm): Good -- reliable for browsing, calls, streaming.")
    elif sig >= -75:
        lines.append(f"Signal ({sig:.1f} dBm): Fair -- may see occasional buffering / slower speeds.")
    else:
        lines.append(f"Signal ({sig:.1f} dBm): Weak -- likely to experience drops or very low speed.")

    if thr >= 100:
        lines.append(f"Throughput ({thr:.1f} Mbps): High -- suitable for 4K streaming, large downloads.")
    elif thr >= 25:
        lines.append(f"Throughput ({thr:.1f} Mbps): Moderate -- fine for HD video, video calls, browsing.")
    else:
        lines.append(f"Throughput ({thr:.1f} Mbps): Low -- may struggle with HD video or multiple devices.")

    if lat <= 30:
        lines.append(f"Latency ({lat:.1f} ms): Low -- good for gaming/video calls.")
    elif lat <= 80:
        lines.append(f"Latency ({lat:.1f} ms): Moderate -- acceptable for most everyday use.")
    else:
        lines.append(f"Latency ({lat:.1f} ms): High -- noticeable lag in real-time applications.")

    return lines


def main():
    parser = argparse.ArgumentParser(description="Wi-Fi Performance Prediction + AP Placement CLI")
    parser.add_argument("--room_w", type=float, help="Room width in meters")
    parser.add_argument("--room_h", type=float, help="Room height in meters")
    parser.add_argument("--walls", type=int, help="Number of walls between router and device")
    parser.add_argument("--users", type=int, help="Number of concurrent connected users")
    parser.add_argument("--interference", choices=["low", "medium", "high"], help="Interference level")
    parser.add_argument("--band", choices=["2.4GHz", "5GHz"], help="Frequency band")
    parser.add_argument("--distance", type=float, help="Distance from router in meters (optional; "
                                                          "defaults to half the room diagonal)")
    parser.add_argument("--num_aps", type=int, default=1,
                         help="Number of access points to jointly place (research-gap extension; "
                              "default 1 = original single-AP behaviour, use 2+ for multi-AP "
                              "greedy coverage optimization)")
    parser.add_argument("--interference_aware", action="store_true",
                         help="SECOND research-gap extension: when placing 2+ APs, also penalize "
                              "placing them close enough to co-channel-interfere with each other "
                              "(see placement/ap_placement.py::find_optimal_multi_ap_placement). "
                              "No effect with --num_aps 1.")
    parser.add_argument("--sinr_threshold_db", type=float, default=9.0,
                         help="Minimum acceptable signal-to-interference ratio in dB when "
                              "--interference_aware is set (default 9.0)")
    args = parser.parse_args()

    # Interactive fallback for any missing arguments -- with range validation, so both
    # --flag misuse (e.g. --room_w -5) AND interactive typos are rejected consistently
    # with the same rules the web UI (app/flask_app.py::validate_form) enforces.
    def ask(prompt, cast=float, choices=None, min_val=None, max_val=None):
        while True:
            val = input(prompt).strip()
            try:
                v = cast(val)
                if choices and v not in choices:
                    print(f"  Please choose one of {choices}")
                    continue
                if min_val is not None and v < min_val:
                    print(f"  Please enter a value >= {min_val}")
                    continue
                if max_val is not None and v > max_val:
                    print(f"  Please enter a value <= {max_val}")
                    continue
                return v
            except ValueError:
                print("  Invalid input, try again.")

    def require(value, prompt, cast=float, choices=None, min_val=None, max_val=None):
        """Returns `value` if it was actually supplied AND passes validation; otherwise
        prompts interactively. Uses `is not None` (not truthy) checks throughout so an
        explicitly-passed 0 (e.g. --walls 0) is never mistaken for "not supplied"."""
        if value is not None:
            ok = True
            if choices and value not in choices:
                print(f"Invalid --{prompt.split()[0].lower()}: must be one of {choices}. "
                      f"Falling back to interactive input.")
                ok = False
            if min_val is not None and value < min_val:
                print(f"Invalid value {value}: must be >= {min_val}. Falling back to interactive input.")
                ok = False
            if max_val is not None and value > max_val:
                print(f"Invalid value {value}: must be <= {max_val}. Falling back to interactive input.")
                ok = False
            if ok:
                return value
        return ask(prompt, cast, choices=choices, min_val=min_val, max_val=max_val)

    room_w = require(args.room_w, "Room width (m): ", float, min_val=0.1, max_val=200)
    room_h = require(args.room_h, "Room height (m): ", float, min_val=0.1, max_val=200)
    walls = require(args.walls, "Number of walls to device: ", int, min_val=0, max_val=20)
    users = require(args.users, "Number of concurrent users: ", int, min_val=1, max_val=500)
    interference = require(args.interference, "Interference (low/medium/high): ", str,
                            choices=["low", "medium", "high"])
    band = require(args.band, "Frequency band (2.4GHz/5GHz): ", str, choices=["2.4GHz", "5GHz"])
    distance = args.distance if args.distance is not None else (np.hypot(room_w, room_h) / 2)
    if args.num_aps < 1 or args.num_aps > 4:
        print(f"--num_aps must be between 1 and 4 (got {args.num_aps}); using 1 instead.")
        args.num_aps = 1

    print("\nLoading trained models...")
    artifacts, models = load_artifacts()

    print("\n" + "=" * 60)
    print("PREDICTED WI-FI PERFORMANCE")
    print("=" * 60)
    preds = predict_performance(distance, walls, users, interference, band, artifacts, models)
    print(f"Signal Strength : {preds['signal_strength_dbm']:.2f} dBm")
    print(f"Throughput      : {preds['throughput_mbps']:.2f} Mbps")
    print(f"Latency         : {preds['latency_ms']:.2f} ms")

    print("\nInterpretation:")
    for line in interpret_results(preds):
        print(f"  - {line}")

    print("\n" + "=" * 60)
    print("RECOMMENDED ACCESS POINT PLACEMENT")
    print("=" * 60)

    num_aps = max(1, args.num_aps)
    if num_aps == 1:
        print(f"Computing optimal router position for a {room_w}m x {room_h}m room...")
        best_pos, best_cov = recommend_placement(room_w, room_h, band=band)
        print(f"Recommended position: x={best_pos[0]}m, y={best_pos[1]}m (measured from bottom-left corner)")
        print(f"Expected coverage at -65 dBm threshold: {best_cov}%")
    else:
        mode_note = " with interference-awareness (2nd research-gap extension)" if args.interference_aware else ""
        print(f"Computing optimal {num_aps}-AP joint placement for a {room_w}m x {room_h}m room "
              f"(research-gap extension: greedy multi-AP coverage optimization{mode_note})...")
        positions, final_cov, gains = recommend_multi_ap_placement(
            room_w, room_h, num_aps=num_aps, band=band,
            interference_aware=args.interference_aware, sinr_threshold_db=args.sinr_threshold_db)
        for i, (pos, gain) in enumerate(zip(positions, gains)):
            print(f"  AP {i + 1}: x={pos[0]}m, y={pos[1]}m  (+{gain}% marginal coverage gain)")
        if len(positions) < num_aps:
            print(f"  (stopped after {len(positions)} APs -- coverage already reached {final_cov}%, "
                  f"additional APs would add negligible value)")
        print(f"Joint expected coverage at -65 dBm threshold: {final_cov}%")


if __name__ == "__main__":
    main()
