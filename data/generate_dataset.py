"""
STEP 1: DATASET CREATION
=========================
Generates a synthetic but PHYSICS-INFORMED Wi-Fi performance dataset.

Why synthetic instead of a "found" real dataset?
--------------------------------------------------
Public real-world Wi-Fi RSSI/throughput datasets (e.g. UJIIndoorLoc) exist but are built
for indoor *localization*, not for controlled study of (distance, walls, users, interference,
band) -> (signal, throughput, latency). For a project whose entire second half is about
*causally* reasoning about placement, we need ground-truth control over the input variables.
So we simulate the physics ourselves using well-known wireless propagation models:

1. Log-distance Path Loss Model (used in real RF planning tools):
       PL(d) = PL0 + 10 * n * log10(d / d0)
   where n = path loss exponent (~2 for free space, ~2.7-3.5 for indoor office/home)

2. Wall Attenuation Factor (WAF) model (ITU / COST231-style):
       Extra loss = num_walls * attenuation_per_wall (dB)

3. Interference is modeled as an additional stochastic dB penalty + throughput/latency
   penalty, following how co-channel interference degrades effective SNR in real Wi-Fi.

4. Throughput is derived from an approximate Shannon-capacity-style relationship between
   SNR/RSSI and achievable rate, then de-rated for MAC-layer contention among num_users
   (this mimics real 802.11 behaviour where airtime is shared, not bandwidth).

5. Latency is modeled as a function of contention (users), interference, and queuing delay
   that grows non-linearly as the channel approaches saturation.

This gives us realistic, non-linear, noisy relationships -- exactly what makes the ML
step meaningful (a lookup table would NOT need Random Forest / XGBoost; noisy physics does).
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_SAMPLES = 25000

rng = np.random.default_rng(RANDOM_SEED)


def generate_dataset(n_samples: int = N_SAMPLES) -> pd.DataFrame:
    # ---------------------------------------------------------------
    # 1. Sample the raw input features
    # ---------------------------------------------------------------
    distance = rng.uniform(0.5, 50.0, n_samples)                     # meters from AP
    num_walls = rng.integers(0, 7, n_samples)                        # 0-6 walls/obstacles
    num_users = rng.integers(1, 51, n_samples)                       # concurrent users on AP
    interference_level = rng.choice(
        ["low", "medium", "high"], size=n_samples, p=[0.5, 0.35, 0.15]
    )
    frequency_band = rng.choice(["2.4GHz", "5GHz"], size=n_samples, p=[0.55, 0.45])

    # ---------------------------------------------------------------
    # 2. Band-dependent RF constants (based on real-world 802.11 behaviour)
    # ---------------------------------------------------------------
    # 5GHz attenuates faster with distance and walls but has less ambient interference
    # and higher max PHY rate (wider channels, more spatial streams typically available).
    path_loss_exponent = np.where(frequency_band == "5GHz", 3.2, 2.7)
    wall_attenuation_db = np.where(frequency_band == "5GHz", 5.5, 3.5)   # dB loss per wall
    tx_power_dbm = np.where(frequency_band == "5GHz", 23.0, 20.0)         # typical AP TX power
    d0 = 1.0     # reference distance (m)
    pl0 = 40.0   # path loss at reference distance (dB), free-space-ish at 1m

    interference_db_penalty = {"low": 0.0, "medium": 4.0, "high": 9.0}
    interference_db = np.array([interference_db_penalty[i] for i in interference_level])

    # ---------------------------------------------------------------
    # 3. SIGNAL STRENGTH (dBm) via log-distance path loss + walls + interference + noise
    # ---------------------------------------------------------------
    path_loss = pl0 + 10 * path_loss_exponent * np.log10(distance / d0) \
                + num_walls * wall_attenuation_db

    noise = rng.normal(0, 2.0, n_samples)   # measurement/multipath noise
    signal_strength_dbm = tx_power_dbm - path_loss - (0.6 * interference_db) + noise
    signal_strength_dbm = np.clip(signal_strength_dbm, -95, -20)

    # ---------------------------------------------------------------
    # 4. THROUGHPUT (Mbps): SNR-driven ceiling, de-rated for contention
    # ---------------------------------------------------------------
    # Approximate "SNR" proxy from signal strength (assume noise floor ~ -90 dBm)
    noise_floor = -90.0
    snr_db = signal_strength_dbm - noise_floor

    max_phy_mbps = np.where(frequency_band == "5GHz", 866.0, 300.0)  # typical PHY ceiling

    # Shannon-style saturating curve mapped onto realistic Wi-Fi throughput range
    snr_linear = 10 ** (snr_db / 10)
    shannon_factor = np.log2(1 + snr_linear)
    shannon_factor_norm = shannon_factor / np.max(shannon_factor)   # normalize 0-1 ceiling shape

    interference_throughput_penalty = {"low": 1.0, "medium": 0.85, "high": 0.65}
    interf_factor = np.array([interference_throughput_penalty[i] for i in interference_level])

    # Airtime/contention de-rating: throughput per user drops roughly ~1/sqrt(users)
    # (real 802.11 MAC contention is sub-linear, not a hard divide-by-N)
    contention_factor = 1.0 / np.sqrt(num_users)

    raw_throughput = max_phy_mbps * shannon_factor_norm * interf_factor * contention_factor
    throughput_noise = rng.normal(0, 3.0, n_samples)
    throughput_mbps = np.clip(raw_throughput + throughput_noise, 0.5, None)

    # ---------------------------------------------------------------
    # 5. LATENCY (ms): base propagation/processing + queuing delay under contention
    # ---------------------------------------------------------------
    base_latency = 2.0 + (0.15 * num_walls) + (0.05 * distance)
    interference_latency_penalty = {"low": 0.0, "medium": 5.0, "high": 15.0}
    interf_latency = np.array([interference_latency_penalty[i] for i in interference_level])

    # Queuing delay grows non-linearly as users approach channel saturation
    utilization = np.clip(num_users / 50.0, 0.01, 0.98)
    queuing_delay = 20 * (utilization / (1 - utilization))   # M/M/1-like queue blow-up

    weak_signal_penalty = np.clip((-signal_strength_dbm - 60) * 0.5, 0, None)  # penalty below -60dBm

    latency_noise = rng.normal(0, 2.0, n_samples)
    latency_ms = base_latency + interf_latency + queuing_delay + weak_signal_penalty + latency_noise
    latency_ms = np.clip(latency_ms, 1.0, 500.0)

    # ---------------------------------------------------------------
    # 6. Assemble DataFrame
    # ---------------------------------------------------------------
    df = pd.DataFrame({
        "distance_m": np.round(distance, 2),
        "num_walls": num_walls,
        "num_users": num_users,
        "interference_level": interference_level,
        "frequency_band": frequency_band,
        "signal_strength_dbm": np.round(signal_strength_dbm, 2),
        "throughput_mbps": np.round(throughput_mbps, 2),
        "latency_ms": np.round(latency_ms, 2),
    })

    # ---------------------------------------------------------------
    # 7. Inject a small % of missing values to make preprocessing step meaningful
    #    (mirrors real-world sensor logs where some readings are dropped)
    # ---------------------------------------------------------------
    for col in ["signal_strength_dbm", "throughput_mbps", "num_walls"]:
        missing_idx = rng.choice(df.index, size=int(0.01 * n_samples), replace=False)
        df.loc[missing_idx, col] = np.nan

    return df


if __name__ == "__main__":
    dataset = generate_dataset()
    from pathlib import Path
    out_path = Path(__file__).resolve().parent / "wifi_dataset.csv"
    dataset.to_csv(out_path, index=False)
    print(f"Generated {len(dataset)} samples -> {out_path}")
    print("\nPreview:")
    print(dataset.head())
    print("\nMissing values per column:")
    print(dataset.isna().sum())
    print("\nSummary stats:")
    print(dataset.describe(include="all"))
