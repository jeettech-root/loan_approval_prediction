# Loan Approval Prediction

## Project overview

This repository contains an educational machine learning project that demonstrates a complete workflow for predicting loan approval using tabular applicant and financial data. The project includes exploratory data analysis (EDA), model training (notebook), a saved Gradient Boosting model, and a Streamlit-based web application for inference and local experimentation.

**IMPORTANT:** This is an educational prototype — it is not a real banking decision system and should not be used as one.

---

## Problem statement

Lenders must decide whether to approve or reject loan applications. The goal of this project is to build a model that predicts loan approval (Approved = 1, Rejected = 0) from applicant and financial features, and to provide an interactive interface for inspecting predictions and model explanations.

## Objective

- Train and compare several classification models on loan application data.
- Select a final model based on relevant evaluation metrics.
- Provide a Streamlit application that loads the saved model and:
  - Accepts applicant input in the same feature order used during training
  - Produces a prediction and approval probability using `model.predict_proba()`
  - Offers SHAP-based explanations for the individual prediction (when compatible)

---

## Repository structure

- `data/`
  - `loan_data.csv` — original dataset used during development (if present)
- `models/`
  - `loan_approval_model.pkl` — final trained Gradient Boosting model used for inference
- `notebooks/`
  - `EDA_and_training.ipynb` — exploratory data analysis and training workflow
- `app.py` — Streamlit app: loads the model, validates inputs, shows prediction and explainability
- `requirements.txt` — Python dependencies

---

## Dataset description

The dataset contains historical loan applications with features describing applicants, loan terms, credit score, and asset values. Each row corresponds to a single loan application and includes the binary target `loan_status` (Approved = 1, Rejected = 0).

Refer to `data/loan_data.csv` for the exact schema and sample rows (if that file is included).

---

## Features used (in the exact order used for model training)

The model was trained using the following features in this exact order. The Streamlit app preserves this order for inference.

1. no_of_dependents
2. education (encoded: Graduate = 1, Not Graduate = 0)
3. self_employed (encoded: Yes = 1, No = 0)
4. income_annum
5. loan_amount
6. loan_term
7. cibil_score
8. residential_assets_value
9. commercial_assets_value
10. luxury_assets_value
11. bank_asset_value
12. Totalasset

Human-friendly labels used in the application map these fields to readable names (e.g. `no_of_dependents` → "Dependents").

---

## Data preprocessing

Typical preprocessing steps applied during model development include (see the notebook for exact code):

- Handling missing values (imputation or row removal depending on the column and missingness)
- Converting types to numeric where required
- Encoding categorical flags (education, self_employed) to integers matching the training encoding
- Feature scaling when required by particular models (tree models often do not require scaling)
- Train/test split and cross-validation for model selection and hyperparameter tuning

**Note:** The saved model expects inputs with the same column order and numeric encodings used during training.

---

## Exploratory data analysis (EDA) — summary

See `notebooks/EDA_and_training.ipynb` for full EDA. Summary highlights (replace placeholders with your actual findings):

- Class balance: <APPROVED_REJECTED_DISTRIBUTION_PLACEHOLDER>
- Income and loan amount ranges: <INCOME_LOAN_RANGE_PLACEHOLDER>
- Correlations of interest (e.g., CIBIL score vs approval): <CORRELATION_SUMMARY_PLACEHOLDER>
- Missing values and outlier handling decisions: <MISSING_OUTLIER_NOTES_PLACEHOLDER>

---

## Feature engineering

Examples of feature engineering that were considered or applied (see the notebook for details):

- Aggregation of asset components into a `Totalasset` feature
- Ratios such as `loan_amount / income_annum` (debt-to-income proxy)
- Encoding and bucketing where appropriate

---

## Models compared

Models trained and compared during development (see the notebook for code and full results):

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Gradient Boosting (final selected model artifact present in `models/`)

---

## Evaluation metrics

The following metrics were used to compare models. Replace placeholders below with the actual values from your evaluation runs.

- Accuracy: <ACCURACY_PLACEHOLDER>
- Precision: <PRECISION_PLACEHOLDER>
- Recall: <RECALL_PLACEHOLDER>
- F1 Score: <F1_PLACEHOLDER>
- ROC-AUC: <ROC_AUC_PLACEHOLDER>

---

## Final model selection

Final model artifact: `models/loan_approval_model.pkl` — Gradient Boosting model.

Selection rationale (example placeholders — adjust to your results):

- Competitive ROC-AUC and a balanced precision/recall tradeoff on validation sets
- Stable cross-validation performance
- Compatibility with SHAP TreeExplainer for explainability

---

## SHAP explainability

- The Streamlit app computes SHAP values for an individual prediction using TreeExplainer when compatible with the loaded model.
- The app displays the most influential features, their direction (pushing toward approval or rejection), and their relative contribution in a horizontal bar chart.
- If SHAP is incompatible with the saved model artifact, the app shows a clear, factual fallback message and does not fabricate explanations.

**Disclaimer:** SHAP values describe how the model used the provided features for a given prediction; they are not financial advice or an official bank decision.

---

## Streamlit application

`app.py` provides:

- A light-themed, dashboard-style UI with grouped input sections (Applicant Information, Financial Information, Asset Information, Loan Information)
- Input validation that prevents prediction on invalid inputs
- A short loading animation during prediction
- Predictions derived from the saved model using `joblib` and `model.predict_proba()` — the app uses the model's real probability outputs
- SHAP-based "Why this prediction?" explanations when available
- A "Reset Inputs" button to restore sensible defaults and clear previous prediction and explanations

---

## Project architecture

- Notebook (`notebooks/EDA_and_training.ipynb`) — EDA, preprocessing, model training, evaluation
- Model serialization (`models/loan_approval_model.pkl`) — final trained Gradient Boosting model used by the app
- Streamlit app (`app.py`) — inference, validation, UI, SHAP explainability

---

## Installation instructions

1. Clone or download this repository.

2. Create and activate a Python virtual environment (recommended):

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

3. Install Python dependencies:

```bash
pip install -r requirements.txt
```

Recommended Python version: 3.8+ (confirm compatibility with your environment and the packages in `requirements.txt`).

---

## How to run locally

1. Activate the virtual environment and ensure dependencies are installed.
2. Start the Streamlit app:

```bash
streamlit run app.py
```

3. Open http://localhost:8501 in your browser if it does not open automatically.

Notes:
- The app loads the model from `models/loan_approval_model.pkl`. Do not modify or delete that file if you want to reproduce the saved model's behavior.
- Predictions and probability outputs are taken directly from `model.predict_proba()`.

---

## Deployment

Typical deployment options:

- Deploy to Streamlit Cloud or other platform-as-a-service that supports Python and Streamlit
- Containerize the app (Docker) and deploy to any cloud provider (AWS, Azure, GCP, etc.)

Deployment tips:
- Pin dependency versions in `requirements.txt`.
- Secure the model artifact and any sensitive data.
- Add logging, monitoring, and access control for production use.

---

## Screenshots (placeholders)

Add screenshots to `screenshots/` and update these file names:

- `screenshots/01-form.png` — input form
- `screenshots/02-result.png` — prediction result card
- `screenshots/03-shap.png` — SHAP explanation chart

---

## Limitations

- This is an educational project and is not production-grade for real banking or credit decisioning.
- Model performance depends on the training dataset and any biases present in the data will be reflected in predictions.
- The app does not perform identity verification, anti-fraud checks, or regulatory compliance checks.

---

## Future improvements

- Add robust per-field inline error indicators and more advanced client-side validation
- Implement model monitoring, data drift detection, and periodic re-training pipeline
- Expand feature engineering (temporal features, external credit data if available)
- Add test coverage and CI/CD for the app and model training pipeline
- Secure deployment with authentication, logging, and monitoring

---

## License & attribution

This repository is provided for educational purposes. Add or replace with a license of your choice if you intend to redistribute.


If you want, I can also:
- Insert actual evaluation numbers into the metrics section if you provide them
- Add a Dockerfile and minimal deployment guide
- Add real screenshots into `screenshots/` and reference them in this README

---

Generated by: AI assistant (Copilot CLI runtime in VS Code) for an educational loan approval prediction demo.
#   l o a n _ a p p r o v a l _ p r e d i c t i o n  
 