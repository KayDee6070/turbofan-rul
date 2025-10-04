# src/app_rul_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from joblib import load
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# ---------------- CONFIG ----------------
st.set_page_config(page_title="NASA Turbofan RUL Dashboard", layout="wide")
st.title("NASA Turbofan Engine Remaining Useful Life (RUL) Prediction")

# ---------------- LOAD MODELS ----------------
rf_model = load("models/FD001_rf.joblib")
xgb_model = load("models/FD001_xgb.joblib")
lstm_model = load_model("models/FD001_lstm.h5", compile=False)

TRAIN_META_PATH = "data/processed/FD001_train_features.csv"
_train_head = pd.read_csv(TRAIN_META_PATH, nrows=1)
TREE_FEATURE_COLS = [c for c in _train_head.columns if c not in {"unit", "cycle", "RUL"}]

LSTM_FEATS = ["s4_mean5", "s9_mean5", "s11_mean5", "s9_mean20", "s14_mean20"]
SEQ_LEN = 30

# ---------------- FUNCTIONS ----------------
def predict_tree(df, model):
    X = df[TREE_FEATURE_COLS].values
    preds = model.predict(X)
    return preds

def predict_lstm(df):
    df_scaled = df.copy()
    scaler = StandardScaler()
    df_scaled[LSTM_FEATS] = scaler.fit_transform(df_scaled[LSTM_FEATS])
    arr = df_scaled[LSTM_FEATS].values
    X = [arr[i:i + SEQ_LEN, :] for i in range(len(arr) - SEQ_LEN)]
    X = np.array(X)
    preds = lstm_model.predict(X).flatten()
    return np.concatenate([np.full(SEQ_LEN, np.nan), preds])

def plot_rul_trend(df, unit_id, pred_col="Pred_RF", ax=None):
    df_unit = df[df["unit"] == unit_id]
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(df_unit["cycle"], df_unit[pred_col], label="Predicted RUL", color="steelblue")
    if "RUL" in df_unit.columns:
        ax.plot(df_unit["cycle"], df_unit["RUL"], label="Actual RUL", color="orange")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Remaining Useful Life")
    ax.set_title(f"Engine {unit_id} RUL Trend")
    ax.legend()
    return ax

def plot_feature_importance(model, title):
    importances = model.feature_importances_
    feat_df = pd.DataFrame({"Feature": TREE_FEATURE_COLS, "Importance": importances})
    feat_df = feat_df.sort_values("Importance", ascending=False).head(15)
    fig, ax = plt.subplots()
    sns.barplot(x="Importance", y="Feature", data=feat_df, ax=ax)
    ax.set_title(title)
    st.pyplot(fig)

def shap_explain(model, X_sample):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    st.pyplot(bbox_inches="tight")

# ---------------- SIDEBAR ----------------
uploaded_file = st.sidebar.file_uploader("Upload processed test data", type=["csv"])
compare_models = st.sidebar.checkbox("Compare All Models", value=True)
show_shap = st.sidebar.checkbox("Show SHAP Feature Importance", value=False)

# ---------------- MAIN ----------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of uploaded data")
    st.dataframe(df.head())

    engine_ids = sorted(df["unit"].unique()) if "unit" in df.columns else []
    selected_engine = st.sidebar.selectbox("Select Engine Unit ID", engine_ids)

    if st.button("Run Prediction"):
        try:
            preds_rf = predict_tree(df, rf_model)
            preds_xgb = predict_tree(df, xgb_model)
            preds_lstm = predict_lstm(df)

            df["Pred_RF"] = preds_rf
            df["Pred_XGB"] = preds_xgb
            df["Pred_LSTM"] = preds_lstm

            st.success("Prediction complete.")

            # --- Model comparison summary ---
            if compare_models:
                last = df.groupby("unit")[["Pred_RF", "Pred_XGB", "Pred_LSTM"]].tail(1)
                st.subheader("Model Comparison (Last Cycle per Engine)")
                st.dataframe(last.round(2))

                plot_df = last.copy()
                if len(plot_df) > 25:
                    plot_df = plot_df.sample(25, random_state=42).sort_index()

                fig, ax = plt.subplots(figsize=(10, 5))
                bar_width = 0.25
                x = np.arange(len(plot_df))

                ax.bar(x - bar_width, plot_df["Pred_RF"], width=bar_width, label="Random Forest")
                ax.bar(x, plot_df["Pred_XGB"], width=bar_width, label="XGBoost")
                ax.bar(x + bar_width, plot_df["Pred_LSTM"], width=bar_width, label="LSTM")

                ax.set_xticks(x)
                ax.set_xticklabels(plot_df.index, rotation=45, ha="right")
                ax.set_xlabel("Engine Unit")
                ax.set_ylabel("Predicted RUL (cycles)")
                ax.set_title("Predicted RUL Comparison per Engine (Last Cycle, Sampled Engines)")
                ax.legend()
                st.pyplot(fig)

            # --- Selected engine trend ---
            st.subheader(f"Engine {selected_engine} RUL Trend")
            fig2, ax2 = plt.subplots()
            plot_rul_trend(df, selected_engine, pred_col="Pred_RF", ax=ax2)
            st.pyplot(fig2)

            # --- Feature importance ---
            st.subheader("Feature Importance (Random Forest)")
            plot_feature_importance(rf_model, "Random Forest - Top 15 Features")

            # --- SHAP explainability ---
            if show_shap:
                st.subheader("SHAP Feature Explainability (Random Forest)")
                X_sample = df[TREE_FEATURE_COLS].sample(500, random_state=42)
                shap_explain(rf_model, X_sample)

        except Exception as e:
            st.error(str(e))
else:
    st.info("Upload the processed FD001_test_features.csv file to start analysis.")
