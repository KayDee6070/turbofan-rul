import os
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model

# === Configuration ===
FD = "FD001"
MODELS_DIR = "models"
OUT_PATH = os.path.join(MODELS_DIR, f"{FD}_model_report.pdf")

# === Styling for professional plots ===
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 16,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "axes.labelcolor": "black",
    "text.color": "black"
})

# === Load metrics data ===
metrics_path = os.path.join(MODELS_DIR, f"{FD}_baseline_metrics.csv")
df_metrics = pd.read_csv(metrics_path)
print(df_metrics)

# === Display model comparison results ===
print("\n=== Model Comparison Results ===")
print(df_metrics[["model", "MAE", "RMSE"]])

# === Plot RMSE Comparison ===
plt.figure(figsize=(8, 5))
bars = plt.bar(df_metrics["model"], df_metrics["RMSE"], color="steelblue", alpha=0.8)
plt.title("Model RMSE Comparison (lower = better)", pad=15, weight="bold")
plt.xlabel("Model", labelpad=10)
plt.ylabel("RMSE", labelpad=10)
plt.xticks(rotation=20, ha="right")

# Highlight the best model
min_rmse_idx = df_metrics["RMSE"].idxmin()
bars[min_rmse_idx].set_color("dodgerblue")

# Save high-resolution comparison chart
plot_path = os.path.join(MODELS_DIR, f"{FD}_comparison.png")
plt.tight_layout()
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()

# === Optionally load LSTM for verification ===
try:
    lstm_model = load_model(os.path.join(MODELS_DIR, f"{FD}_lstm.h5"))
    print("[INFO] LSTM model loaded successfully.")
except Exception as e:
    print(f"[WARN] Could not load LSTM model: {e}")

# === Export results summary to PDF ===
from matplotlib.backends.backend_pdf import PdfPages

pdf_path = os.path.join(MODELS_DIR, f"{FD}_model_report.pdf")
with PdfPages(pdf_path) as pdf:
    # Summary Table
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(
        cellText=df_metrics.round(2).values,
        colLabels=df_metrics.columns,
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    plt.title("Model Comparison Results", fontsize=14, weight="bold", pad=10)
    pdf.savefig(fig, bbox_inches="tight")

    # RMSE Comparison Chart
    img = plt.imread(plot_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(img)
    ax.axis("off")
    pdf.savefig(fig, bbox_inches="tight")

print(f"[INFO] Report saved to {pdf_path}")
