import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "loan_approval_model.pkl"
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
FEATURE_NAME_MAP = {
    "loan_id": "Loan ID",
    " no_of_dependents": "Dependents",
    " education": "Education",
    " self_employed": "Self Employment",
    " income_annum": "Annual Income",
    " loan_amount": "Loan Amount",
    " loan_term": "Loan Term",
    " cibil_score": "CIBIL Score",
    " residential_assets_value": "Residential Assets",
    " commercial_assets_value": "Commercial Assets",
    " luxury_assets_value": "Luxury Assets",
    " bank_asset_value": "Bank Assets",
    "Totalasset": "Total Assets",
}


def load_model():
    return joblib.load(MODEL_PATH)


def education_code(value: str) -> int:
    mapping = {"Graduate": 1, "Not Graduate": 0}
    return mapping[value]


def self_employed_code(value: str) -> int:
    mapping = {"Yes": 1, "No": 0}
    return mapping[value]


def prepare_feature_vector(data: dict) -> pd.DataFrame:
    # The trained model expects the original feature order.
    # loan_id is kept at 0 because it is not a predictive feature for the decision itself.
    feature_row = {
        "loan_id": 0,
        " no_of_dependents": int(data["no_of_dependents"]),
        " education": education_code(data["education"]),
        " self_employed": self_employed_code(data["self_employed"]),
        " income_annum": float(data["income_annum"]),
        " loan_amount": float(data["loan_amount"]),
        " loan_term": int(data["loan_term"]),
        " cibil_score": int(data["cibil_score"]),
        " residential_assets_value": float(data["residential_assets_value"]),
        " commercial_assets_value": float(data["commercial_assets_value"]),
        " luxury_assets_value": float(data["luxury_assets_value"]),
        " bank_asset_value": float(data["bank_asset_value"]),
        "Totalasset": float(data["total_asset"]),
    }
    return pd.DataFrame([feature_row], columns=FEATURE_COLUMNS)


def predict_loan_approval(model, form_data: dict):
    features = prepare_feature_vector(form_data)
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    positive_index = int(np.where(model.classes_ == 1)[0][0])
    approved_probability = float(probabilities[positive_index]) * 100
    return int(prediction), approved_probability, probabilities


def get_shap_feature_contributions(model, feature_df: pd.DataFrame):
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(feature_df)
        values = np.asarray(shap_values)

        if values.ndim == 3:
            positive_index = int(np.where(model.classes_ == 1)[0][0]) if hasattr(model, "classes_") else 1
            values = values[positive_index]

        values = np.asarray(values).reshape(-1, feature_df.shape[1])[0]
        feature_names = list(feature_df.columns)

        contributions = []
        total_abs = sum(abs(v) for v in values)

        for feature_name, value in zip(feature_names, values):
            contribution = float(value)
            rel = (abs(contribution) / total_abs * 100.0) if total_abs else 0.0
            contributions.append(
                {
                    "name": FEATURE_NAME_MAP.get(feature_name, feature_name.strip()),
                    "value": contribution,
                    "relative": rel,
                    "direction": "Positive" if contribution >= 0 else "Negative",
                }
            )

        return sorted(contributions, key=lambda item: abs(item["value"]), reverse=True)[:8]
    except Exception:
        return None


def validate_inputs(data: dict):
    """Validate form inputs. Returns a list of human-readable error messages.
    The caller should not attempt prediction if this returns any errors."""
    errors = []

    # no_of_dependents must be non-negative
    if data.get("no_of_dependents", 0) is None or int(data.get("no_of_dependents", 0)) < 0:
        errors.append("No. of Dependents must be 0 or a positive integer.")

    # income_annum must be > 0
    if data.get("income_annum", 0) is None or float(data.get("income_annum", 0)) <= 0:
        errors.append("Annual Income must be greater than 0.")

    # loan_amount must be > 0
    if data.get("loan_amount", 0) is None or float(data.get("loan_amount", 0)) <= 0:
        errors.append("Loan Amount must be greater than 0.")

    # loan_term must be > 0
    if data.get("loan_term", 0) is None or int(data.get("loan_term", 0)) <= 0:
        errors.append("Loan Term must be greater than 0 months.")

    # cibil_score between 0 and 900
    try:
        cibil = float(data.get("cibil_score", 0))
        if not (0 <= cibil <= 900):
            errors.append("CIBIL Score must be between 0 and 900.")
    except Exception:
        errors.append("CIBIL Score must be a number between 0 and 900.")

    # asset values non-negative
    asset_fields = {
        "residential_assets_value": "Residential Assets Value",
        "commercial_assets_value": "Commercial Assets Value",
        "luxury_assets_value": "Luxury Assets Value",
        "bank_asset_value": "Bank Asset Value",
        "total_asset": "Total Asset",
    }

    for key, label in asset_fields.items():
        try:
            if float(data.get(key, 0)) < 0:
                errors.append(f"{label} must be non-negative.")
        except Exception:
            errors.append(f"{label} must be a number (0 or positive).")

    return errors


def main():
    st.set_page_config(page_title="Loan Approval Prediction", page_icon="💰", layout="wide")

    st.markdown(
        """
        <style>
        :root {
            --bg: #f4f7fb;
            --panel: #ffffff;
            --panel-alt: #f8faff;
            --line: #dfe7f4;
            --text: #122033;
            --muted: #53677d;
            --primary: #4f46e5;
            --primary-soft: #eef2ff;
            --primary-dark: #312e81;
            --success: #0f9d6e;
            --error: #d14343;
            --shadow: 0 12px 28px rgba(79, 70, 229, 0.08);
        }

        html, body {
            background: var(--bg);
            overflow-x: hidden;
        }

        .stApp {
            background: var(--bg);
            color: var(--text);
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        div[data-testid="stForm"] {
            background: transparent;
            padding: 0;
            border: none;
        }

        .header-panel {
            background: linear-gradient(135deg, rgba(79, 70, 229, 0.06), rgba(59, 130, 246, 0.04));
            border: 1px solid var(--line);
            border-radius: 18px;
            box-shadow: var(--shadow);
            padding: 1rem 1.2rem 0.8rem;
            margin-bottom: 0.7rem;
            position: relative;
            overflow: hidden;
        }

        .header-pill {
            display: inline-flex;
            align-items: center;
            background: var(--primary-soft);
            color: var(--primary-dark);
            border: 1px solid rgba(79,70,229,0.12);
            border-radius: 999px;
            padding: 0.25rem 0.7rem;
            font-size: 0.66rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.5rem;
        }

        .header-panel h1 {
            margin: 0;
            font-size: clamp(1.8rem, 2.8vw, 2.5rem);
            line-height: 1.1;
            color: var(--text);
            font-weight: 800;
        }

        .header-panel p {
            margin: 0.4rem 0 0;
            color: var(--muted);
            font-size: 0.95rem;
        }

        .section-wrap {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 16px;
            box-shadow: 0 8px 16px rgba(15, 23, 42, 0.025);
            padding: 0.6rem 0.8rem 0.7rem;
            margin-top: 0.7rem;
            margin-bottom: 0.7rem;
            transition: transform 0.18s ease, box-shadow 0.18s ease;
        }

        .section-wrap:hover {
            box-shadow: 0 10px 18px rgba(79, 70, 229, 0.08);
            transform: translateY(-1px);
        }

        .section-title {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            font-size: 1rem;
            font-weight: 700;
            color: var(--text);
            margin: 0.1rem 0 0.5rem;
        }

        .section-title .icon {
            display: inline-flex;
            width: 1.8rem;
            height: 1.8rem;
            align-items: center;
            justify-content: center;
            border-radius: 8px;
            background: var(--primary-soft);
            color: var(--primary);
            font-size: 0.9rem;
        }

        [data-testid="stHorizontalBlock"] {
            gap: 0.85rem;
        }

        [data-testid="stVerticalBlock"] > div {
            margin-bottom: 0.2rem;
        }

        .stNumberInput, .stSelectbox, .stSlider {
            background: transparent;
        }

        div[data-testid="stFormSubmitButton"] > button {
            background: linear-gradient(135deg, var(--primary), #3b82f6);
            color: #ffffff;
            border: none;
            border-radius: 12px;
            font-weight: 700;
            font-size: 1rem;
            padding: 0.8rem 1.2rem;
            box-shadow: 0 12px 24px rgba(79, 70, 229, 0.2);
            transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease;
        }

        div[data-testid="stFormSubmitButton"] > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 14px 26px rgba(79, 70, 229, 0.24);
            filter: brightness(1.02);
        }

        .loading-card {
            display: flex;
            align-items: center;
            gap: 0.9rem;
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            box-shadow: var(--shadow);
            margin-top: 1rem;
            animation: fadeIn 0.25s ease;
        }

        .loader {
            width: 22px;
            height: 22px;
            border: 3px solid rgba(79, 70, 229, 0.15);
            border-top-color: var(--primary);
            border-radius: 50%;
            animation: spin 0.9s linear infinite;
        }

        .loading-text {
            color: var(--text);
            font-weight: 600;
            font-size: 1rem;
        }

        .result-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 18px;
            box-shadow: var(--shadow);
            padding: 1.2rem 1.25rem;
            margin-top: 1rem;
            animation: slideUp 0.35s ease;
        }

        .result-card.success-card {
            border-color: rgba(15, 157, 110, 0.25);
            background: rgba(15, 157, 110, 0.04);
        }

        .result-card.warning-card {
            border-color: rgba(209, 67, 67, 0.25);
            background: rgba(209, 67, 67, 0.04);
        }

        .result-header {
            display: flex;
            align-items: center;
            gap: 0.9rem;
        }

        .result-icon {
            width: 54px;
            height: 54px;
            border-radius: 15px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.7rem;
            font-weight: 800;
            animation: fadeIn 0.3s ease;
        }

        .result-card.success-card .result-icon {
            background: rgba(15, 157, 110, 0.12);
            color: var(--success);
        }

        .result-card.warning-card .result-icon {
            background: rgba(209, 67, 67, 0.1);
            color: var(--error);
        }

        .result-status {
            font-size: 1.8rem;
            font-weight: 800;
            margin: 0 0 0.2rem;
            letter-spacing: -0.03em;
        }

        .result-status.success {
            color: var(--success);
        }

        .result-status.warning {
            color: var(--error);
        }

        .result-explanation {
            color: var(--muted);
            margin-top: 0.45rem;
            line-height: 1.5;
        }

        .probability-panel {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 0.9rem 1rem 0.8rem;
            margin-top: 1rem;
            box-shadow: 0 8px 16px rgba(15, 23, 42, 0.02);
            animation: fadeIn 0.35s ease;
        }

        .probability-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 0.8rem;
            margin-bottom: 0.5rem;
        }

        .probability-label {
            font-size: 0.8rem;
            font-weight: 700;
            color: var(--muted);
            letter-spacing: 0.03em;
            text-transform: uppercase;
        }

        .probability-value {
            font-size: 1.5rem;
            font-weight: 800;
            color: var(--text);
        }

        .probability-meta {
            font-size: 0.74rem;
            color: var(--muted);
            margin-top: 0.2rem;
        }

        .probability-track {
            width: 100%;
            height: 12px;
            background: #edf2ff;
            border-radius: 999px;
            overflow: hidden;
            border: 1px solid rgba(79, 70, 229, 0.08);
            margin-top: 0.65rem;
        }

        .probability-fill {
            height: 100%;
            width: 0;
            border-radius: 999px;
            background: linear-gradient(90deg, #4f46e5 0%, #3b82f6 100%);
            animation: progressGrow 1.2s ease-out forwards;
            box-shadow: inset 0 0 0 1px rgba(255,255,255,0.2);
        }

        @keyframes progressGrow {
            from { width: 0; }
            to { width: var(--fill-width, 50%); }
        }

        .explain-panel {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 18px;
            box-shadow: var(--shadow);
            padding: 1.1rem 1.2rem;
            margin-top: 1.1rem;
        }

        .explain-title {
            font-size: 1.15rem;
            font-weight: 800;
            color: var(--text);
            margin-bottom: 0.9rem;
        }

        .shap-row {
            display: grid;
            grid-template-columns: minmax(150px, 1.4fr) 110px 1.5fr 70px;
            gap: 0.7rem;
            align-items: center;
            padding: 0.55rem 0.2rem;
            border-bottom: 1px solid #edf2f7;
        }

        .shap-row:last-child {
            border-bottom: none;
        }

        .shap-name {
            font-weight: 600;
            color: var(--text);
        }

        .shap-direction {
            font-size: 0.76rem;
            font-weight: 700;
            border-radius: 999px;
            padding: 0.28rem 0.5rem;
            text-align: center;
        }

        .shap-direction.positive {
            background: rgba(15, 157, 110, 0.08);
            color: var(--success);
        }

        .shap-direction.negative {
            background: rgba(209, 67, 67, 0.08);
            color: var(--error);
        }

        .shap-bar-wrap {
            width: 100%;
            height: 10px;
            background: #edf2ff;
            border-radius: 999px;
            overflow: hidden;
            border: 1px solid rgba(79, 70, 229, 0.08);
        }

        .shap-bar {
            height: 100%;
            border-radius: 999px;
            transition: width 1s ease-in-out;
        }

        .shap-bar.positive {
            background: linear-gradient(90deg, #10b981, #34d399);
        }

        .shap-bar.negative {
            background: linear-gradient(90deg, #ef4444, #f87171);
        }

        .shap-value {
            text-align: right;
            font-size: 0.78rem;
            font-weight: 700;
            color: var(--muted);
        }

        .disclaimer {
            margin-top: 0.9rem;
            font-size: 0.8rem;
            color: var(--muted);
            line-height: 1.5;
        }

        .result-explanation {
            color: var(--muted);
            margin-top: 0.4rem;
            line-height: 1.5;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(4px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(12px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        div[data-testid="stMetric"] > div {
            background: var(--panel-alt);
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 0.8rem 1rem;
        }

        .stSuccess, .stError {
            border-radius: 14px;
            padding: 0.8rem 1rem;
            border: 1px solid transparent;
        }

        .stSuccess {
            background: rgba(15, 157, 110, 0.08);
            border-color: rgba(15, 157, 110, 0.2);
        }

        .stError {
            background: rgba(209, 67, 67, 0.06);
            border-color: rgba(209, 67, 67, 0.2);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="header-panel">
            <div class="header-pill">AI assessment</div>
            <h1>Loan Approval Prediction</h1>
            <p>AI-powered assessment based on applicant financial information.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # initialize persistent session state safely
    if "prediction" not in st.session_state:
        st.session_state.prediction = None
    if "probability" not in st.session_state:
        st.session_state.probability = None
    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None
    if "show_result" not in st.session_state:
        st.session_state.show_result = False
    if "asset_mode" not in st.session_state:
        st.session_state.asset_mode = "Automatic"
    if "manual_total_assets" not in st.session_state:
        st.session_state.manual_total_assets = 0.0

    # sensible defaults for reset
    defaults = {
        "no_of_dependents": 0,
        "education": "Graduate",
        "self_employed": "No",
        "income_annum": 5000000,
        "loan_amount": 2000000,
        "loan_term": 180,
        "cibil_score": 650,
        "residential_assets_value": 1000000,
        "commercial_assets_value": 1000000,
        "luxury_assets_value": 1000000,
        "bank_asset_value": 1000000,
        "manual_total_assets": 5000000,
        "asset_mode": "Automatic",
    }

    # initialize session state with defaults when first run
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Reset button (clears input state and any stored outputs)
    reset_col, _ = st.columns([1, 6])
    with reset_col:
        if st.button("Reset Inputs", key="reset_button"):
            # reset input values to defaults
            for k, v in defaults.items():
                st.session_state[k] = v

            # reset prediction state
            st.session_state.prediction = None
            st.session_state.probability = None
            st.session_state.prediction_result = None
            st.session_state.show_result = False
            st.session_state.asset_mode = "Automatic"
            st.session_state.manual_total_assets = defaults.get("manual_total_assets", 0.0)

            # clear explanation if present
            if "explanation_features" in st.session_state:
                del st.session_state["explanation_features"]

            # re-run to update UI immediately
            st.rerun()

    with st.form("loan_form"):
        st.markdown('<div class="section-wrap"><div class="section-title"><span class="icon">👤</span>Applicant Information</div>', unsafe_allow_html=True)
        applicant_col1, applicant_col2 = st.columns(2)
        with applicant_col1:
            no_of_dependents = st.number_input("No. of Dependents", min_value=0, max_value=20, value=st.session_state["no_of_dependents"], step=1, key="no_of_dependents", help="Number of people dependent on applicant. More dependents can increase repayment burden.")
            education = st.selectbox("Education", ["Graduate", "Not Graduate"], index=0 if st.session_state["education"] == "Graduate" else 1, key="education", help="Highest education level. Higher education often correlates with income and job stability.")
        with applicant_col2:
            self_employed = st.selectbox("Self Employed", ["No", "Yes"], index=0 if st.session_state["self_employed"] == "No" else 1, key="self_employed", help="Self-employment status. Employment type can affect income stability.")
            income_annum = st.number_input("Annual Income", min_value=0, value=st.session_state["income_annum"], step=100000, key="income_annum", help="Applicant's annual income. Higher income can improve repayment capacity.")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-wrap"><div class="section-title"><span class="icon">💰</span>Financial Information</div>', unsafe_allow_html=True)
        financial_col1, financial_col2 = st.columns(2)
        with financial_col1:
            loan_amount = st.number_input("Loan Amount", min_value=0, value=st.session_state["loan_amount"], step=100000, key="loan_amount", help="Requested loan amount. Larger loans may increase monthly repayment burden.")
            loan_term = st.slider("Loan Term (months)", min_value=1, max_value=360, value=st.session_state["loan_term"], key="loan_term", help="Loan duration in months. Longer terms lower monthly payments.")
        with financial_col2:
            cibil_score = st.slider("CIBIL Score", min_value=0, max_value=900, value=st.session_state["cibil_score"], key="cibil_score", help="Applicant credit score (0–900). Higher scores indicate stronger credit history.")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-wrap"><div class="section-title"><span class="icon">🏠</span>Asset Information</div>', unsafe_allow_html=True)
        asset_col1, asset_col2 = st.columns(2)
        with asset_col1:
            residential_assets_value = st.number_input("Residential Assets Value", min_value=0, value=st.session_state["residential_assets_value"], step=100000, key="residential_assets_value", help="Value of residential assets. Higher asset value can strengthen financial position.")
            commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0, value=st.session_state["commercial_assets_value"], step=100000, key="commercial_assets_value", help="Value of commercial assets. Higher commercial assets can improve collateral strength.")
        with asset_col2:
            luxury_assets_value = st.number_input("Luxury Assets Value", min_value=0, value=st.session_state["luxury_assets_value"], step=100000, key="luxury_assets_value", help="Value of luxury assets (e.g., vehicles). Additional assets may indicate wealth.")
            bank_asset_value = st.number_input("Bank Asset Value", min_value=0, value=st.session_state["bank_asset_value"], step=100000, key="bank_asset_value", help="Liquid bank assets (savings). More bank assets improve available repayment funds.")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-wrap"><div class="section-title"><span class="icon">📄</span>Loan Information</div>', unsafe_allow_html=True)

        # Calculate Totalasset from the four asset fields using current widget values
        try:
            total_computed = (
                float(residential_assets_value)
                + float(commercial_assets_value)
                + float(luxury_assets_value)
                + float(bank_asset_value)
            )
        except Exception:
            total_computed = 0.0

        # Asset Calculation Mode: Automatic (default) or Manual
        st.markdown("**Asset Calculation Mode**")
        asset_mode = st.radio(
            "",
            ["Automatic", "Manual"],
            index=0 if st.session_state.get("asset_mode", "Automatic") == "Automatic" else 1,
            key="asset_mode",
            help="Choose whether Total Assets are calculated automatically or entered manually.",
        )

        total_cols = st.columns([2, 3])
        with total_cols[0]:
            st.markdown("**Total Assets**")
            if asset_mode == "Manual":
                # manual input - stable key 'manual_total_assets'
                manual_total = st.number_input(
                    "Total Assets",
                    min_value=0,
                    value=int(st.session_state.get("manual_total_assets", int(total_computed))),
                    step=100000,
                    key="manual_total_assets",
                    help="Enter total assets manually. Must be non-negative.",
                )
                final_total = float(manual_total)
                st.session_state["total_asset"] = final_total
            else:
                final_total = float(total_computed)
                # update session state total_asset for submission
                st.session_state["total_asset"] = final_total
                st.markdown(f"### ₹{int(final_total):,}")
        with total_cols[1]:
            st.markdown("Calculated automatically from:  Residential + Commercial + Luxury + Bank Assets")
            st.info("Automatically calculated as the sum of residential, commercial, luxury, and bank assets.", icon="ℹ️")

        st.markdown('</div>', unsafe_allow_html=True)

        submitted = st.form_submit_button("Predict Loan Approval", use_container_width=True)

    if submitted:
        # Build the form_data dict first (used for validation and prediction)
        form_data = {
            "no_of_dependents": int(st.session_state.get("no_of_dependents", 0)),
            "education": st.session_state.get("education", "Graduate"),
            "self_employed": st.session_state.get("self_employed", "No"),
            "income_annum": float(st.session_state.get("income_annum", 0.0)),
            "loan_amount": float(st.session_state.get("loan_amount", 0.0)),
            "loan_term": int(st.session_state.get("loan_term", 0)),
            "cibil_score": float(st.session_state.get("cibil_score", 0.0)),
            "residential_assets_value": float(st.session_state.get("residential_assets_value", 0.0)),
            "commercial_assets_value": float(st.session_state.get("commercial_assets_value", 0.0)),
            "luxury_assets_value": float(st.session_state.get("luxury_assets_value", 0.0)),
            "bank_asset_value": float(st.session_state.get("bank_asset_value", 0.0)),
            "total_asset": float(st.session_state.get("total_asset", 0.0)),
        }

        # Run validation and show clear messages if anything is invalid
        errors = validate_inputs(form_data)
        if errors:
            st.markdown('<div class="section-wrap">', unsafe_allow_html=True)
            st.error("Please fix the following input issues before running the prediction:")
            for err in errors:
                st.markdown(f"<div style='color:#b00020;margin-left:8px;'>• {err}</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Valid inputs — show the professional short loading animation and run prediction
            loading_placeholder = st.empty()
            loading_placeholder.markdown(
                """
                <div class="loading-card">
                    <div class="loader"></div>
                    <div class="loading-text">Analyzing application...</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            time.sleep(1.3)
            loading_placeholder.empty()

            try:
                model = load_model()
            except Exception as e:
                st.error(f"Failed to load model: {e}")
                model = None

            if model is not None:
                try:
                    prediction, approved_probability, probabilities = predict_loan_approval(model, form_data)
                except Exception as e:
                    st.error("Prediction failed. Please check the input values and try again.")
                    st.exception(e)
                    prediction = None

                if prediction is not None:
                    classes = model.classes_.tolist()
                    # approval and rejection probabilities (percent)
                    rejected_probability = float(probabilities[classes.index(0)]) * 100

                    # store results in session_state for persistence
                    st.session_state["prediction"] = int(prediction)
                    st.session_state["probability_vector"] = probabilities.tolist()
                    st.session_state["probability"] = float(approved_probability) if int(prediction) == 1 else float(rejected_probability)
                    st.session_state["prediction_result"] = "Approved" if int(prediction) == 1 else "Rejected"
                    st.session_state["show_result"] = True

                    # render result card
                    if int(prediction) == 1:
                        result_html = f"""
                        <div class="result-card success-card">
                            <div class="result-header">
                                <div class="result-icon">✓</div>
                                <div>
                                    <div class="result-status success">Loan Approved</div>
                                    <div class="result-explanation">Based on the information provided, the model predicts a higher probability of approval.</div>
                                </div>
                            </div>
                        </div>
                        <div class="probability-panel">
                            <div class="probability-header">
                                <div class="probability-label">Approval Probability</div>
                                <div class="probability-value">{approved_probability:.2f}%</div>
                            </div>
                            <div class="probability-meta">Model confidence estimate</div>
                            <div class="probability-track">
                                <div class="probability-fill" style="--fill-width: {approved_probability:.2f}%; width: {approved_probability:.2f}%;"></div>
                            </div>
                        </div>
                        """
                        st.markdown(result_html, unsafe_allow_html=True)
                    else:
                        result_html = f"""
                        <div class="result-card warning-card">
                            <div class="result-header">
                                <div class="result-icon">!</div>
                                <div>
                                    <div class="result-status warning">Loan Rejected</div>
                                    <div class="result-explanation">Based on the information provided, the model predicts a higher probability of rejection.</div>
                                </div>
                            </div>
                        </div>
                        <div class="probability-panel">
                            <div class="probability-header">
                                <div class="probability-label">Rejection Probability</div>
                                <div class="probability-value">{rejected_probability:.2f}%</div>
                            </div>
                            <div class="probability-meta">Model confidence estimate</div>
                            <div class="probability-track">
                                <div class="probability-fill" style="--fill-width: {rejected_probability:.2f}%; width: {rejected_probability:.2f}%;"></div>
                            </div>
                        </div>
                        """
                        st.markdown(result_html, unsafe_allow_html=True)

                    # SHAP explanation (unchanged) — compute and render explanations for this valid input
                    feature_df = prepare_feature_vector(form_data)
                    try:
                        explanation_features = get_shap_feature_contributions(model, feature_df)
                        st.session_state["explanation_features"] = explanation_features
                    except Exception:
                        explanation_features = None
                        st.session_state["explanation_features"] = None

                    st.markdown('<div class="explain-panel">', unsafe_allow_html=True)
                    st.markdown('<div class="explain-title">Why this prediction?</div>', unsafe_allow_html=True)

                    if explanation_features is None:
                        st.warning(
                            "SHAP explanation is unavailable for this saved model. To enable feature-based explanations, the model must be saved from a tree-based estimator compatible with TreeExplainer or exported with a compatible model artifact."
                        )
                    else:
                        max_abs = max(abs(item["value"]) for item in explanation_features) if explanation_features else 1.0
                        for item in explanation_features:
                            width = (abs(item["value"]) / max_abs) * 100 if max_abs else 0.0
                            direction_class = "positive" if item["value"] >= 0 else "negative"
                            direction_label = "Positive" if item["value"] >= 0 else "Negative"
                            st.markdown(
                                f"""
                                <div class="shap-row">
                                    <div class="shap-name">{item['name']}</div>
                                    <div class="shap-direction {direction_class}">{direction_label}</div>
                                    <div class="shap-bar-wrap"><div class="shap-bar {direction_class}" style="width: {width:.1f}%;"></div></div>
                                    <div class="shap-value">{abs(item['value']):.2f}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                    st.markdown(
                        '<div class="disclaimer">These explanations describe how the model used the provided features. They are not financial advice or a bank\'s official decision criteria.</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
