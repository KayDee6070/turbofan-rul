import shap
from joblib import load
import pandas as pd

# Load model and test data
rf = load("models/FD001_rf.joblib")
df = pd.read_csv("data/processed/FD001_test.csv")

# Drop non-feature columns
sample = df.sample(100, random_state=42).drop(columns=["unit", "cycle", "RUL"], errors="ignore")

# SHAP test
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(sample)

print("✅ SHAP worked, shape:", len(shap_values), "x", sample.shape[1])
