# src/data_prep.py
import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List

# ----- Basic column names -----
COLS = (
    ["unit", "cycle", "setting1", "setting2", "setting3"]
    + [f"s{i}" for i in range(1, 22)]
)

# ----- Step 1: Read dataset -----
def _read_cmapss_split(fd: str, data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    train_p = os.path.join(data_dir, f"train_{fd}.txt")
    test_p  = os.path.join(data_dir, f"test_{fd}.txt")
    rul_p   = os.path.join(data_dir, f"RUL_{fd}.txt")

    train = pd.read_csv(train_p, sep=r"\s+", header=None, names=COLS)
    test  = pd.read_csv(test_p,  sep=r"\s+", header=None, names=COLS)
    rul   = pd.read_csv(rul_p,   sep=r"\s+", header=None).iloc[:, 0].values
    return train, test, rul

# ----- Step 2: Compute RUL labels -----
def add_rul_labels(train: pd.DataFrame, test: pd.DataFrame, test_last_rul: np.ndarray):
    train = train.copy()
    max_cycle = train.groupby("unit")["cycle"].transform("max")
    train["RUL"] = (max_cycle - train["cycle"]).astype(int)

    test = test.copy()
    last_cycle = test.groupby("unit")["cycle"].transform("max")
    mapping = {u: int(test_last_rul[u - 1]) for u in range(1, test["unit"].nunique() + 1)}
    test["rul_last"] = test["unit"].map(mapping)
    test["RUL"] = (last_cycle + test["rul_last"] - test["cycle"]).astype(int)
    test.drop(columns="rul_last", inplace=True)
    return train, test

# ----- Step 3: Optional cap -----
def cap_rul(df: pd.DataFrame, cap: Optional[int]):
    if cap is None:
        return df
    df = df.copy()
    df["RUL"] = df["RUL"].clip(upper=int(cap))
    return df

# ----- Step 4: Rolling + lag features -----
def engineer_features(df: pd.DataFrame, windows=[5, 20], lags=[1, 5]):
    df = df.sort_values(["unit", "cycle"]).copy()
    sensors = [c for c in df.columns if c.startswith("s")]

    for w in windows:
        mean_ = df.groupby("unit")[sensors].rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
        std_  = df.groupby("unit")[sensors].rolling(w, min_periods=1).std().fillna(0).reset_index(level=0, drop=True)
        mean_.columns = [f"{c}_mean{w}" for c in sensors]
        std_.columns  = [f"{c}_std{w}" for c in sensors]
        df = pd.concat([df, mean_, std_], axis=1)

    for lag in lags:
        shifted = df.groupby("unit")[sensors].shift(lag)
        shifted.columns = [f"{c}_lag{lag}" for c in sensors]
        # Compute delta safely row-wise
        delta = (df[sensors].values - shifted.values)
        delta = pd.DataFrame(delta, columns=[f"{c}_d{lag}" for c in sensors])
        df = pd.concat([df, shifted, delta], axis=1)


    df = df.groupby("unit", group_keys=False).apply(lambda g: g.fillna(method="bfill").fillna(method="ffill"))
    return df.reset_index(drop=True)

# ----- Step 5: Full pipeline -----
def load_prepare_fd(fd="FD001", data_dir="data/raw", rul_cap=125, persist_dir="data/processed"):
    train, test, rul_vec = _read_cmapss_split(fd, data_dir)
    train, test = add_rul_labels(train, test, rul_vec)
    train = cap_rul(train, rul_cap)
    test = cap_rul(test, rul_cap)
    train_f = engineer_features(train)
    test_f  = engineer_features(test)

    os.makedirs(persist_dir, exist_ok=True)
    train_f.to_csv(os.path.join(persist_dir, f"{fd}_train_features.csv"), index=False)
    test_f.to_csv(os.path.join(persist_dir, f"{fd}_test_features.csv"), index=False)
    print(f"[INFO] Saved processed {fd} data in '{persist_dir}'")
    return train_f, test_f
