"""
STEP 2: DATA PREPROCESSING
============================
- Handle missing values (median imputation for numeric columns)
- Encode categorical features (One-Hot Encoding for interference_level, frequency_band)
- Scale numeric features (StandardScaler) -- fit ONLY on train data to avoid leakage
- Train/test split (80/20)

Returns reusable objects (imputer, encoder, scaler) so the SAME transformation can be
applied later to a brand-new user input in Step 6 (the CLI interface).
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent

NUMERIC_FEATURES = ["distance_m", "num_walls", "num_users"]
CATEGORICAL_FEATURES = ["interference_level", "frequency_band"]
TARGETS = ["signal_strength_dbm", "throughput_mbps", "latency_ms"]


def load_raw_data(path=None) -> pd.DataFrame:
    if path is None:
        path = PROJECT_ROOT / "data" / "wifi_dataset.csv"
    return pd.read_csv(path)


def build_preprocessing_pipeline():
    """
    ColumnTransformer that:
      - median-imputes + scales numeric columns
      - imputes (most frequent) + one-hot-encodes categorical columns
    """
    numeric_pipeline = ColumnTransformer(transformers=[
        ("num_impute_scale", "passthrough", NUMERIC_FEATURES)
    ])  # placeholder, real pipeline built inline below for clarity

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    return preprocessor


def preprocess_and_split(df: pd.DataFrame, test_size=0.2, random_state=42):
    # -----------------------------------------------------------
    # 1. Handle missing values
    #    - Numeric feature columns -> median
    #    - Numeric target columns  -> drop rows (never impute a target! that would fabricate
    #      ground truth). Missing targets are dropped BEFORE splitting.
    # -----------------------------------------------------------
    df = df.copy()
    df = df.dropna(subset=TARGETS)  # never train on a fabricated label

    num_imputer = SimpleImputer(strategy="median")
    df[NUMERIC_FEATURES] = num_imputer.fit_transform(df[NUMERIC_FEATURES])

    # -----------------------------------------------------------
    # 2. Train/test split (do this before fitting scaler/encoder -> avoid data leakage)
    # -----------------------------------------------------------
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGETS]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # -----------------------------------------------------------
    # 3. Fit encoder + scaler on TRAIN ONLY, transform both
    # -----------------------------------------------------------
    preprocessor = build_preprocessing_pipeline()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = (
        NUMERIC_FEATURES +
        list(preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES))
    )

    artifacts = {
        "preprocessor": preprocessor,
        "num_imputer": num_imputer,
        "feature_names": feature_names,
    }

    return X_train_processed, X_test_processed, y_train, y_test, artifacts


if __name__ == "__main__":
    df = load_raw_data()
    print(f"Loaded {len(df)} raw rows, {df.isna().sum().sum()} total missing cells")

    X_train, X_test, y_train, y_test, artifacts = preprocess_and_split(df)

    print(f"\nAfter dropping rows with missing targets: usable rows = {len(df.dropna(subset=TARGETS))}")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Feature names after encoding: {artifacts['feature_names']}")

    # Persist artifacts + splits for the modeling step
    prep_dir = PROJECT_ROOT / "preprocessing"
    joblib.dump(artifacts, prep_dir / "preprocessing_artifacts.pkl")
    np.save(prep_dir / "X_train.npy", X_train)
    np.save(prep_dir / "X_test.npy", X_test)
    y_train.to_csv(prep_dir / "y_train.csv", index=False)
    y_test.to_csv(prep_dir / "y_test.csv", index=False)
    print("\nSaved preprocessing artifacts and train/test splits.")
