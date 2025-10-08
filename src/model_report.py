import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from joblib import load

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
MODELS_DIR = "models"
DATA_DIR = "data/processed"
FD = "FD001"

# -------------------------------------------------
# LOAD MODELS
# -------------------------------------------------
print("[INFO] Loading models...")

rf_model = load(os.path.join(MODELS_DIR, f"{FD}_rf.joblib"))
xgb_model = load(os.path.join(MODELS_DIR, f"{FD}_xgb.joblib"))
lstm_model = load_model(os.path.join(MODELS_DIR, f"{FD}_lstm.h5"))

# -------------------------------------------------
# LOAD METRICS
# -------------------------------------------------
results_df = pd.read_csv(os.path.join(MODELS_DIR, f"{FD}_baseline_metrics.csv"))
results_df.columns = ["model", "MAE", "RMSE"]

print("\n=== Model Comparison Results ===")
print(results_df)

# -------------------------------------------------
# PLOT: MODEL RMSE COMPARISON
# -------------------------------------------------
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.barplot(
    x="model",
    y="RMSE",
    data=results_df,
    palette=["gray" if m != "XGBoost (all cycles)" else "steelblue" for m in results_df["model"]]
)

plt.title("Model RMSE Comparison (lower = better)", fontsize=14, weight="bold")
plt.ylabel("RMSE", fontsize=12)
plt.xlabel("")
plt.xticks(rotation=25, ha="right", fontsize=10)
plt.tight_layout()

comparison_path = os.path.join(MODELS_DIR, f"{FD}_comparison.png")
plt.savefig(comparison_path, dpi=300)
plt.close()

print(f"[INFO] Saved RMSE comparison chart to {comparison_path}")

# -------------------------------------------------
# SAVE REPORT TO PDF (OPTIONAL)
# -------------------------------------------------
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
    from reportlab.lib.styles import getSampleStyleSheet

    report_path = os.path.join(MODELS_DIR, f"{FD}_model_report.pdf")
    doc = SimpleDocTemplate(report_path, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Model Comparison Report", styles["Title"]))
    elements.append(Spacer(1, 20))

    # Add table data
    table_html = results_df.to_html(index=False)
    elements.append(Paragraph("Model Performance Summary:", styles["Heading2"]))
    elements.append(Paragraph(table_html, styles["Normal"]))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("Model RMSE Comparison (lower = better):", styles["Heading2"]))
    elements.append(Image(comparison_path, width=450, height=300))
    elements.append(Spacer(1, 12))

    doc.build(elements)
    print(f"[INFO] PDF report saved to {report_path}")

except ImportError:
    print("[WARNING] reportlab not installed. Skipping PDF generation.")

# -------------------------------------------------
# END
# -------------------------------------------------
print("\n[INFO] Model report generation complete.")
