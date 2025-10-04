# src/model_report.py
import os
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from joblib import load
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

FD = "FD001"
DATA_DIR = "data/processed"
MODELS_DIR = "models"
REPORT_PATH = f"{MODELS_DIR}/{FD}_model_report.pdf"

# --------- Load metrics from baselines ----------
metrics_csv = os.path.join(MODELS_DIR, f"{FD}_baseline_metrics.csv")
baseline_df = pd.read_csv(metrics_csv)

# --------- Evaluate LSTM again on test ----------
import numpy as np
test = pd.read_csv(os.path.join(DATA_DIR, f"{FD}_test_features.csv"))
selected_feats = ["s4_mean5","s9_mean5","s11_mean5","s9_mean20","s14_mean20"]

# load scaler if you saved it; otherwise re-scale inline
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
test[selected_feats] = scaler.fit_transform(test[selected_feats])

def make_sequences(df, seq_len=30):
    X, y = [], []
    for _, g in df.groupby("unit"):
        data = g[selected_feats + ["RUL"]].values
        for i in range(len(data) - seq_len):
            X.append(data[i:i+seq_len, :-1])
            y.append(data[i+seq_len, -1])
    return np.array(X), np.array(y)

X_test, y_test = make_sequences(test)

lstm_model = load_model(os.path.join(MODELS_DIR, f"{FD}_lstm.h5"), compile=False)
preds = lstm_model.predict(X_test).flatten()
mae = mean_absolute_error(y_test, preds)
rmse = sqrt(mean_squared_error(y_test, preds))

lstm_df = pd.DataFrame([{
    "model": "LSTM",
    "MAE": mae,
    "RMSE": rmse
}])

# Combine all
all_results = pd.concat([baseline_df, lstm_df], ignore_index=True)
print(all_results)

# --------- Plot comparison ---------
plt.figure(figsize=(6,4))
plt.bar(all_results["model"], all_results["RMSE"], color=["gray","gray","steelblue"])
plt.title("Model RMSE Comparison (lower = better)")
plt.ylabel("RMSE")
plt.tight_layout()
plot_path = os.path.join(MODELS_DIR, f"{FD}_comparison.png")
plt.savefig(plot_path)
plt.close()

# --------- Generate PDF ---------
styles = getSampleStyleSheet()
doc = SimpleDocTemplate(REPORT_PATH)
story = []

story.append(Paragraph("<b>NASA Turbofan Engine RUL Prediction</b>", styles["Title"]))
story.append(Spacer(1,12))
story.append(Paragraph("Project Summary:", styles["Heading2"]))
story.append(Paragraph(
    "This project predicts the Remaining Useful Life (RUL) of aircraft engines using NASA's CMAPSS dataset (FD001). "
    "Three models were compared: Random Forest, XGBoost, and an LSTM sequence model. "
    "Data preprocessing included rolling/lag feature engineering and RUL labeling.",
    styles["Normal"]
))
story.append(Spacer(1,12))

story.append(Paragraph("Model Performance Summary:", styles["Heading2"]))
table_data = [["Model","MAE","RMSE"]] + all_results.round(3).values.tolist()
t = Table(table_data, hAlign='LEFT')
t.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0),colors.grey),
    ("TEXTCOLOR",(0,0),(-1,0),colors.whitesmoke),
    ("ALIGN",(0,0),(-1,-1),"CENTER"),
    ("GRID",(0,0),(-1,-1),0.5,colors.black),
]))
story.append(t)
story.append(Spacer(1,12))

story.append(Paragraph("RMSE Comparison Chart:", styles["Heading2"]))
story.append(Image(plot_path, width=400, height=300))

story.append(Spacer(1,12))
story.append(Paragraph(
    "The LSTM model captures temporal degradation trends, while tree-based models provide strong baselines. "
    "Future work could include multi-sensor attention models or multi-fleet domain adaptation.",
    styles["Normal"]
))
doc.build(story)
print(f"[INFO] Report saved to {REPORT_PATH}")