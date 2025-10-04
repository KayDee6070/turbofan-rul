# src/train_baseline.py
import os
import argparse
import pandas as pd
from joblib import dump
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from data_prep import load_prepare_fd
from math import sqrt

def _split_last_cycle(df):
    """Return (X_all, y_all, X_last, y_last)."""
    drop_cols = {"RUL", "unit", "cycle"}
    feat_cols = [c for c in df.columns if c not in drop_cols]

    X_all, y_all = df[feat_cols].values, df["RUL"].values
    last = df.sort_values(["unit", "cycle"]).groupby("unit").tail(1)
    X_last, y_last = last[feat_cols].values, last["RUL"].values
    return feat_cols, X_all, y_all, X_last, y_last

def _report(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))  # manual RMSE
    return {"model": name, "MAE": mae, "RMSE": rmse}

def main(fd, rul_cap, out_dir):
    print(f"[INFO] Loading {fd} ...")
    train_df, test_df = load_prepare_fd(fd=fd, rul_cap=rul_cap)

    print("[INFO] Building splits ...")
    feat_cols, Xtr_all, ytr_all, Xtr_last, ytr_last = _split_last_cycle(train_df)
    _, Xte_all, yte_all, Xte_last, yte_last         = _split_last_cycle(test_df)

    results = []

    # ----- Random Forest -----
    print("[INFO] Training Random Forest ...")
    rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=42)
    rf.fit(Xtr_all, ytr_all)
    pred_all  = rf.predict(Xte_all)
    pred_last = rf.predict(Xte_last)
    results.append(_report("RandomForest (all cycles)", yte_all, pred_all))
    results.append(_report("RandomForest (last cycle)", yte_last, pred_last))
    dump(rf, os.path.join(out_dir, f"{fd}_rf.joblib"))

    # ----- XGBoost -----
    print("[INFO] Training XGBoost ...")
    xgb = XGBRegressor(
        n_estimators=600, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, tree_method="hist"
    )
    xgb.fit(Xtr_all, ytr_all, eval_set=[(Xte_all, yte_all)], verbose=False)
    pred_all  = xgb.predict(Xte_all)
    pred_last = xgb.predict(Xte_last)
    results.append(_report("XGBoost (all cycles)", yte_all, pred_all))
    results.append(_report("XGBoost (last cycle)", yte_last, pred_last))
    dump(xgb, os.path.join(out_dir, f"{fd}_xgb.joblib"))

    table = pd.DataFrame(results)
    print("\n=== Baseline Results ({}) ===".format(fd))
    print(table.to_string(index=False))

    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, f"{fd}_baseline_metrics.csv")
    table.to_csv(metrics_path, index=False)
    print(f"[INFO] Saved models + metrics to '{out_dir}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fd", type=str, default="FD001")
    parser.add_argument("--rul_cap", type=int, default=125)
    parser.add_argument("--out_dir", type=str, default="models")
    args = parser.parse_args()
    main(args.fd, args.rul_cap, args.out_dir)
