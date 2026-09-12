"""
STEP 3: MACHINE LEARNING MODELS
==================================
We train a SEPARATE regressor per target (signal_strength_dbm, throughput_mbps, latency_ms)
rather than one multi-output model, because:
  - Each target has a different noise structure and physical relationship to the inputs
  - It lets us report per-target R2/RMSE/MAE, which is far more informative for a viva
  - Tree ensembles in sklearn support MultiOutputRegressor, but per-target models tend to
    perform better and are easier to explain/defend

Models compared per target:
  1. Linear Regression       -> baseline, interpretable
  2. Random Forest Regressor -> handles non-linearity + feature interactions
  3. XGBoost Regressor       -> gradient boosting, usually best for this kind of tabular data

Metrics: R2, RMSE, MAE (computed on the held-out test set)
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

TARGETS = ["signal_strength_dbm", "throughput_mbps", "latency_ms"]
PREP_DIR = PROJECT_ROOT / "preprocessing"
MODEL_DIR = PROJECT_ROOT / "models"


def load_splits():
    X_train = np.load(f"{PREP_DIR}/X_train.npy")
    X_test = np.load(f"{PREP_DIR}/X_test.npy")
    y_train = pd.read_csv(f"{PREP_DIR}/y_train.csv")
    y_test = pd.read_csv(f"{PREP_DIR}/y_test.csv")
    return X_train, X_test, y_train, y_test


def get_model_zoo():
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=14, random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1
        ),
    }


def evaluate(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


def train_and_compare():
    X_train, X_test, y_train, y_test = load_splits()
    results_rows = []
    best_models = {}

    for target in TARGETS:
        y_tr = y_train[target].values
        y_te = y_test[target].values

        best_r2 = -np.inf
        best_model_name = None
        best_model_obj = None

        for model_name, model in get_model_zoo().items():
            model.fit(X_train, y_tr)
            preds = model.predict(X_test)
            metrics = evaluate(y_te, preds)

            results_rows.append({
                "Target": target,
                "Model": model_name,
                "R2": round(metrics["R2"], 4),
                "RMSE": round(metrics["RMSE"], 4),
                "MAE": round(metrics["MAE"], 4),
            })

            if metrics["R2"] > best_r2:
                best_r2 = metrics["R2"]
                best_model_name = model_name
                best_model_obj = model

        best_models[target] = {"model_name": best_model_name, "model": best_model_obj, "r2": best_r2}
        joblib.dump(best_model_obj, f"{MODEL_DIR}/best_model_{target}.pkl")

    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(f"{MODEL_DIR}/model_comparison_results.csv", index=False)
    return results_df, best_models


if __name__ == "__main__":
    results_df, best_models = train_and_compare()

    print("=" * 70)
    print("MODEL COMPARISON TABLE (per target)")
    print("=" * 70)
    for target in TARGETS:
        print(f"\n--- Target: {target} ---")
        print(results_df[results_df["Target"] == target].to_string(index=False))

    print("\n" + "=" * 70)
    print("BEST MODEL PER TARGET")
    print("=" * 70)
    for target, info in best_models.items():
        print(f"{target:25s} -> {info['model_name']:20s} (R2 = {info['r2']:.4f})")

    print("\nSaved: model_comparison_results.csv and best_model_<target>.pkl files")
