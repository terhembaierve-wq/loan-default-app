# Loan Default Prediction — Project Pipeline

A six-stage pipeline that takes the raw `Loan_default.csv` dataset from
inspection through to a deployable Streamlit prototype for predicting
loan default risk.

**Live demo:** https://loan-default-app-hm9xoyppanp7n5yfnfbths.streamlit.app/

## Dataset

`Loan_default.csv` — 255,347 rows, 18 columns, no missing values, no
duplicates. Target column: `Default` (1 = defaulted, 0 = did not
default), imbalanced at roughly 88% / 12%.

## Setup

```bash
pip install -r requirements.txt
```

Place `Loan_default.csv` in the same folder as the scripts before running
any of them.

## Pipeline stages

Run in order — each stage saves the files the next stage needs.

| # | Script | What it does | Produces |
|---|--------|---------------|----------|
| 1 | `01_data_understanding.py` | Inspects shape, dtypes, missing values, duplicates, summary stats, target balance | Console report |
| 2 | `02_eda.py` | Visualises distributions, class balance, correlations, and default drivers | 6 PNG charts |
| 3 | `03_data_preparation.py` | Drops `LoanID`, splits train/test (stratified), builds a scaling + one-hot encoding pipeline | `preprocessor.pkl`, `raw_splits.pkl`, `processed_splits.pkl` |
| 4 | `04_model_development.py` | Trains Logistic Regression and Random Forest, tunes RF with `GridSearchCV`, compares by CV ROC-AUC | `model_logistic_regression.pkl`, `model_random_forest_default.pkl`, `model_random_forest_tuned.pkl` |
| 5 | `05_model_evaluation.py` | Compares models on test ROC-AUC, produces confusion matrix and threshold-sensitivity analysis, selects the best model | `eval_roc_curve.png`, `eval_confusion_matrix.png`, `eval_threshold_sensitivity.png`, `final_model.pkl` |
| 6 | `06_streamlit_app.py` | Loads `preprocessor.pkl` + `final_model.pkl` and serves a form-based prediction UI | Live app |

```bash
python 01_data_understanding.py
python 02_eda.py
python 03_data_preparation.py
python 04_model_development.py
python 05_model_evaluation.py
streamlit run 06_streamlit_app.py
```

## Key findings (from EDA)

- Higher `InterestRate` and `DTIRatio` associate with higher default rates.
- Lower `CreditScore` and `MonthsEmployed` associate with higher default rates.
- `EmploymentType = Unemployed` shows a notably higher default rate than other groups.
- The target is imbalanced (~88% / ~12%), so models are trained with
  `class_weight="balanced"` and evaluated with ROC-AUC rather than raw accuracy.

## Notes on model selection

Stage 5 compares models on test-set ROC-AUC (threshold-independent) and
then inspects the confusion matrix and precision/recall trade-off at
different thresholds for the better model. The 0.5 vs 0.3 threshold
comparison illustrates the trade-off between catching more defaulters
(higher recall) and avoiding false alarms (higher precision) — the
actual production threshold should be chosen based on the relative
business cost of a missed defaulter vs. a wrongly flagged good loan.
