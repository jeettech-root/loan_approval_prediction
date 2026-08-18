# verify_model.py
"""Simple verification script that loads the saved model and runs predict/predict_proba
on the saved sanity test rows created by train_model.py
"""
import joblib
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "loan_approval_model.pkl"
X_TEST_CSV = BASE / "models" / "sanity_X_test_head5.csv"

if not MODEL_PATH.exists():
    print("Model file not found:", MODEL_PATH)
    raise SystemExit(1)

print("Loading model from:", MODEL_PATH)
model = joblib.load(MODEL_PATH)
print("Model loaded:", type(model), model.__class__)

if not X_TEST_CSV.exists():
    print("Sanity X_test CSV not found:", X_TEST_CSV)
    raise SystemExit(1)

X_test = pd.read_csv(X_TEST_CSV)
print("X_test shape:", X_test.shape)

pred = model.predict(X_test)
prob = model.predict_proba(X_test)

print("prediction:", pred)
print("probability:", prob)
