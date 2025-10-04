# src/eda_feature_importance.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load

sns.set(style="whitegrid")

DATA_DIR = "data/processed"
MODELS_DIR = "models"
FD = "FD001"

# ---- Load processed data ----
train = pd.read_csv(os.path.join(DATA_DIR, f"{FD}_train_features.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, f"{FD}_test_features.csv"))

print("[INFO] Train shape:", train.shape)
print("[INFO] Test shape:", test.shape)
print("[INFO] Columns:", len(train.columns))

# ---- Quick look at one engine ----
engine_id = 1
sample = train[train["unit"] == engine_id]
plt.figure(figsize=(10,4))
plt.plot(sample["cycle"], sample["RUL"])
plt.gca().invert_yaxis()
plt.title(f"Engine {engine_id}: RUL vs. Cycle (Degrades over time)")
plt.xlabel("Cycle")
plt.ylabel("Remaining Useful Life")
plt.tight_layout()
plt.show()

# ---- Correlation heatmap (top sensors) ----
sensor_cols = [c for c in train.columns if c.startswith("s")]
corr = train[sensor_cols + ["RUL"]].corr()["RUL"].abs().sort_values(ascending=False)
top_sensors = corr.head(10).index.tolist()

plt.figure(figsize=(8,6))
sns.heatmap(train[top_sensors].corr(), cmap="coolwarm", annot=False)
plt.title("Top Sensor Correlations (RUL strongest at top)")
plt.tight_layout()
plt.show()

# ---- Feature importance: Random Forest ----
rf_path = os.path.join(MODELS_DIR, f"{FD}_rf.joblib")
rf_model = load(rf_path)

drop_cols = {"RUL", "unit", "cycle"}
feature_cols = [c for c in train.columns if c not in drop_cols]
importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=False)

top = importances.head(15)
plt.figure(figsize=(8,5))
sns.barplot(x=top.values, y=top.index)
plt.title("Random Forest Feature Importance (Top 15)")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

print("[INFO] Top 10 important features:")
print(top.head(10))
