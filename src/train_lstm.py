# src/train_lstm.py
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error

FD = "FD001"
DATA_DIR = "data/processed"
MODEL_PATH = f"models/{FD}_lstm.h5"

# ---------- 1. Load processed features ----------
train = pd.read_csv(os.path.join(DATA_DIR, f"{FD}_train_features.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, f"{FD}_test_features.csv"))

# Focus on most important features (from your EDA)
selected_feats = ["s4_mean5","s9_mean5","s11_mean5","s9_mean20","s14_mean20"]

# ---------- 2. Normalise ----------
scaler = StandardScaler()
train[selected_feats] = scaler.fit_transform(train[selected_feats])
test[selected_feats]  = scaler.transform(test[selected_feats])

# ---------- 3. Convert to sequences ----------
def make_sequences(df, seq_len=30):
    X, y = [], []
    for _, group in df.groupby("unit"):
        data = group[selected_feats + ["RUL"]].values
        for i in range(len(data) - seq_len):
            X.append(data[i:i+seq_len, :-1])
            y.append(data[i+seq_len, -1])
    return np.array(X), np.array(y)

SEQ_LEN = 30
X_train, y_train = make_sequences(train, SEQ_LEN)
X_test,  y_test  = make_sequences(test, SEQ_LEN)

print(f"[INFO] X_train: {X_train.shape}, X_test: {X_test.shape}")

# ---------- 4. Build model ----------
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, len(selected_feats))),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse")

# ---------- 5. Train ----------
es = EarlyStopping(patience=5, restore_best_weights=True)
hist = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=40,
    batch_size=64,
    callbacks=[es],
    verbose=1
)

# ---------- 6. Evaluate ----------
preds = model.predict(X_test).flatten()
mae = mean_absolute_error(y_test, preds)
rmse = sqrt(mean_squared_error(y_test, preds))
print(f"[RESULT] LSTM MAE={mae:.3f}, RMSE={rmse:.3f}")

# ---------- 7. Save model ----------
model.save(MODEL_PATH)
print(f"[INFO] Model saved to {MODEL_PATH}")
