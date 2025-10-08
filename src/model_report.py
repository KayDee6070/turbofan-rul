import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
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
from tensorflow.keras.losses import MeanSquaredError

try:
    lstm_model = load_model(
        os.path.join(MODELS_DIR, f"{FD}_lstm.h5"),
        compile=False  # skip recompiling to avoid legacy issues
    )
    print("[INFO] LSTM model loaded (compile=False).")
except Exception as e:
    print(f"[WARNING] Could not load LSTM model normally: {e}")
    print("[INFO] Attempting custom deserialization fix...")
    lstm_model = load_model(
        os.path.join(MODELS_DIR, f"{FD}_lstm.h5"),
        custom_objects={"mse": MeanSquaredError()},
        compile=False
    )
    print("[INFO] LSTM model loaded successfully with custom deserialization.")


# -------------------------------------------------
# LOAD METRICS
# -------------------------------------------------
results_df = pd.read_csv(os.path.join(MODELS_DIR, f"{FD}_baseline_metrics.csv"))
results_df.columns = ["model", "MAE", "RMSE"]

print("\n=== Model Comparison Results ===")
print(results_df)

# -------------------------------------------------
# PLOT: MODEL RMSE COMPARISON (FINAL VERSION)
# -------------------------------------------------
sns.set_theme(style="whitegrid", font_scale=1.2)

plt.figure(figsize=(14, 7))

# Wrap long labels cleanly across multiple lines
wrapped_labels = [
    '\n'.join(textwrap.wrap(label, width=12)) for label in results_df["model"]
]

# Bar plot with highlight for best model
bars = sns.barplot(
    x=wrapped_labels,
    y="RMSE",
    data=results_df,
    palette=["#888888" if m != "XGBoost (all cycles)" else "#1f77b4" for m in results_df["model"]],
)

# Add RMSE values above bars
for i, v in enumerate(results_df["RMSE"]):
    bars.text(i, v + 0.8, f"{v:.2f}", color="black", ha="center", fontsize=10, fontweight="medium")

# Styling
plt.title("Model RMSE Comparison (lower = better)", fontsize=16, weight="bold", pad=20)
plt.ylabel("RMSE", fontsize=13, labelpad=12)
plt.xlabel("", fontsize=12)
plt.xticks(rotation=0, ha="center", fontsize=11, linespacing=1.5)
plt.yticks(fontsize=11)
plt.grid(True, axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()

# Save figure in high resolution
comparison_path = os.path.join(MODELS_DIR, f"{FD}_comparison.png")
plt.savefig(comparison_path, dpi=400, bbox_inches="tight")
plt.close()

print(f"[INFO] Saved cleaned RMSE comparison chart to {comparison_path}")

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

    # Add table
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
