# train_model.py
import json
from pathlib import Path
import platform
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "data" / "loan_data.csv"
MODEL_DIR = BASE / "models"
MODEL_PATH = MODEL_DIR / "loan_approval_model.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

# The app's FEATURE_COLUMNS (must match exactly for compatibility)
FEATURE_COLUMNS = [
    "loan_id",
    " no_of_dependents",
    " education",
    " self_employed",
    " income_annum",
    " loan_amount",
    " loan_term",
    " cibil_score",
    " residential_assets_value",
    " commercial_assets_value",
    " luxury_assets_value",
    " bank_asset_value",
    "Totalasset",
]

# Encodings used by the app
education_map = {"Graduate": 1, "Not Graduate": 0}
self_employed_map = {"Yes": 1, "No": 0}


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected dataset at: {path}")
    df = pd.read_csv(path)
    return df


def prepare_dataframe(df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    # Ensure required columns exist.
    required = [
        "no_of_dependents",
        "education",
        "self_employed",
        "income_annum",
        "loan_amount",
        "loan_term",
        "cibil_score",
        "residential_assets_value",
        "commercial_assets_value",
        "luxury_assets_value",
        "bank_asset_value",
        "loan_status",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"The following required columns are missing from the CSV: {missing}. CSV columns: {list(df.columns)}")

    # Compute Totalasset as the SUM of the four asset fields (Residential + Commercial + Luxury + Bank)
    df["Totalasset"] = (
        df["residential_assets_value"].fillna(0).astype(float)
        + df["commercial_assets_value"].fillna(0).astype(float)
        + df["luxury_assets_value"].fillna(0).astype(float)
        + df["bank_asset_value"].fillna(0).astype(float)
    )

    # Map encodings
    if df["education"].dtype == object:
        df["education_mapped"] = df["education"].map(education_map)
    else:
        df["education_mapped"] = df["education"].astype(int)

    if df["self_employed"].dtype == object:
        df["self_employed_mapped"] = df["self_employed"].map(self_employed_map)
    else:
        df["self_employed_mapped"] = df["self_employed"].astype(int)

    # target mapping: assume loan_status already encoded as 0/1; if strings, map
    if df["loan_status"].dtype == object:
        mapping = {"Approved": 1, "Rejected": 0}
        df["loan_status_mapped"] = df["loan_status"].map(mapping)
    else:
        df["loan_status_mapped"] = df["loan_status"].astype(int)

    # Build feature DataFrame with the exact column names (including leading spaces)
    X = pd.DataFrame(index=df.index)
    # loan_id
    if "loan_id" in df.columns:
        X["loan_id"] = df["loan_id"].values
    else:
        X["loan_id"] = 0

    X[" no_of_dependents"] = df["no_of_dependents"].astype(int)
    X[" education"] = df["education_mapped"].astype(int)
    X[" self_employed"] = df["self_employed_mapped"].astype(int)
    X[" income_annum"] = df["income_annum"].astype(float)
    X[" loan_amount"] = df["loan_amount"].astype(float)
    X[" loan_term"] = df["loan_term"].astype(int)
    X[" cibil_score"] = df["cibil_score"].astype(float)
    X[" residential_assets_value"] = df["residential_assets_value"].astype(float)
    X[" commercial_assets_value"] = df["commercial_assets_value"].astype(float)
    X[" luxury_assets_value"] = df["luxury_assets_value"].astype(float)
    X[" bank_asset_value"] = df["bank_asset_value"].astype(float)
    X["Totalasset"] = df["Totalasset"].astype(float)

    y = df["loan_status_mapped"].astype(int)

    # ensure column order
    X = X[FEATURE_COLUMNS]

    return X, y


def train_and_save(X, y):
    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    # Save a small test slice for quick local verification
    X_test.head(5).to_csv(MODEL_DIR / "sanity_X_test_head5.csv", index=False)
    (MODEL_DIR / "sanity_y_test_head5.csv").write_text(
        ",".join(map(str, y_test.head(5).tolist()))
    )

    # Save metadata
    metadata = {
        "python_version": platform.python_version(),
        "scikit_learn_version": __import__("sklearn").__version__,
        "numpy_version": __import__("numpy").__version__,
        "pandas_version": __import__("pandas").__version__,
        "joblib_version": __import__("joblib").__version__,
        "model_type": "GradientBoostingClassifier",
        "feature_names": FEATURE_COLUMNS,
        "target_mapping": {"Approved": 1, "Rejected": 0},
        "education_mapping": education_map,
        "self_employed_mapping": self_employed_map,
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Model saved to:", MODEL_PATH)
    print("Metadata saved to:", METADATA_PATH)
    return model, X_test, y_test


def main():
    print("Loading data from:", DATA_PATH)
    df = load_data(DATA_PATH)
    X, y = prepare_dataframe(df)
    model, X_test, y_test = train_and_save(X, y)
    print("Training complete. Test sample shape:", X_test.shape)

if __name__ == "__main__":
    main()
