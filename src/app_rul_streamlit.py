# src/app_rul_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import logging
from joblib import load
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# ---------------- CONFIG ----------------
st.set_page_config(page_title="NASA Turbofan RUL Dashboard", layout="wide")
st.title("NASA Turbofan Engine Remaining Useful Life (RUL) Prediction")

# ---------------- LOGGING ----------------
logging.basicConfig(
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    level=logging.INFO,
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)
logger = logging.getLogger()
logger.info("App started.")

# ---------------- CONSTANTS ----------------
TRAIN_META_PATH = "data/processed/FD001_train_features.csv"
_train_head = pd.read_csv(TRAIN_META_PATH, nrows=1)
TREE_FEATURE_COLS = [c for c in _train_head.columns if c not in {"unit", "cycle", "RUL"}]
LSTM_FEATS = ["s4_mean5", "s9_mean5", "s11_mean5", "s9_mean20", "s14_mean20"]
SEQ_LEN = 30

# ---------------- MODEL LOADING ----------------
@st.cache_resource
def load_models():
    rf = load("models/FD001_rf.joblib")
    xgb = load("models/FD001_xgb.joblib")
    lstm = load_model("models/FD001_lstm.h5", compile=False)
    logger.info("Models loaded successfully (RF, XGB, LSTM).")
    return rf, xgb, lstm

rf_model, xgb_model, lstm_model = load_models()

# ---------------- FUNCTIONS ----------------
def predict_tree(df, model):
    X = df[TREE_FEATURE_COLS].values
    preds = model.predict(X)
    logger.info(f"Predictions completed for {len(preds)} samples (tree model).")
    return preds

def predict_lstm(df):
    df_scaled = df.copy()
    scaler = StandardScaler()
    df_scaled[LSTM_FEATS] = scaler.fit_transform(df_scaled[LSTM_FEATS])
    arr = df_scaled[LSTM_FEATS].values
    X = [arr[i:i + SEQ_LEN, :] for i in range(len(arr) - SEQ_LEN)]
    X = np.array(X)
    preds = lstm_model.predict(X, verbose=0).flatten()
    preds = np.concatenate([np.full(SEQ_LEN, np.nan), preds])
    logger.info(f"LSTM predictions completed for {len(preds)} samples.")
    return preds

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
    plt.close(fig)

# ---------------- SHAP & PERMUTATION EXPLAINABILITY ----------------
from sklearn.inspection import permutation_importance

def explain_model(model, X_sample, method="shap"):
    """Explain model predictions using SHAP (accurate) or permutation importance (fast fallback)."""
    with st.spinner(f"⚙️ Computing {method.upper()} feature importance... please wait"):
        try:
            if method == "shap":
                import shap
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)

                st.write(f"✅ SHAP computed successfully on {len(X_sample)} samples.")
                fig, ax = plt.subplots()
                shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
                st.pyplot(fig, bbox_inches="tight")
                plt.close(fig)
                logger.info("SHAP explanation plotted successfully.")

            else:
                # --- FAST PERMUTATION IMPORTANCE (top 20 features only) ---
                st.write("Using top 20 most important features for faster computation...")

                # Select top 20 important features from the RandomForest model
                top_feats = pd.Series(model.feature_importances_, index=X_sample.columns)
                top_feats = top_feats.sort_values(ascending=False).head(20).index

                # Compute model predictions as pseudo-targets
                y_pred = model.predict(X_sample)

                # Run permutation importance on top 20 features only
                result = permutation_importance(
                    model,
                    X_sample[top_feats],
                    y_pred,
                    n_repeats=3,        # fewer repeats = faster
                    random_state=42,
                    n_jobs=-1
                )

                # Build DataFrame of feature importances
                importances = pd.DataFrame({
                    "Feature": X_sample[top_feats].columns,
                    "Importance": result.importances_mean
                }).sort_values("Importance", ascending=False)

                # Plot bar chart
                fig, ax = plt.subplots()
                sns.barplot(x="Importance", y="Feature", data=importances, ax=ax)
                ax.set_title("Permutation-Based Feature Importance (Fast Mode)")
                st.pyplot(fig)
                plt.close(fig)

                st.success("✅ Permutation feature importance computed successfully (fast mode).")
                logger.info("Permutation importance plotted successfully (fast mode).")

        except Exception as e:
            st.error(f"{method.upper()} computation failed: {str(e)}")
            logger.exception(f"{method.upper()} computation failed:")


# ---------------- SIDEBAR ----------------
uploaded_file = st.sidebar.file_uploader("Upload processed test data", type=["csv"])
compare_models = st.sidebar.checkbox("Compare All Models", value=True)
explain_method = st.sidebar.radio(
    "Select Explainability Method",
    ("SHAP (accurate, slower)", "Permutation (fast, fallback)")
)


# ---------------- MAIN ----------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of uploaded data")
    st.dataframe(df.head())

    if "unit" in df.columns:
        engine_ids = sorted(df["unit"].unique())
        selected_engine = st.sidebar.selectbox("Select Engine Unit ID", engine_ids)
    else:
        st.error("No 'unit' column found in uploaded data.")
        st.stop()

    if st.button("Run Prediction"):
        try:
            logger.info("Starting predictions for uploaded file.")
            preds_rf = predict_tree(df, rf_model)
            preds_xgb = predict_tree(df, xgb_model)
            preds_lstm = predict_lstm(df)

            df["Pred_RF"] = preds_rf
            df["Pred_XGB"] = preds_xgb
            df["Pred_LSTM"] = preds_lstm

            st.success("Prediction complete.")
            logger.info("Prediction complete and DataFrame updated with results.")

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
                plt.close(fig)

            # --- Selected engine trend ---
            st.subheader(f"Engine {selected_engine} RUL Trend")
            fig2, ax2 = plt.subplots()
            plot_rul_trend(df, selected_engine, pred_col="Pred_RF", ax=ax2)
            st.pyplot(fig2)
            plt.close(fig2)

            # --- Feature importance ---
            st.subheader("Feature Importance (Random Forest)")
            plot_feature_importance(rf_model, "Random Forest - Top 15 Features")

            # --- SHAP explainability ---
            if explain_method:
                st.subheader("Model Explainability (Random Forest)")
                X_sample = df[TREE_FEATURE_COLS].sample(n=min(200, len(df)), random_state=42)
                if "SHAP" in explain_method:
                    explain_model(rf_model, X_sample, method="shap")
                else:
                    explain_model(rf_model, X_sample, method="permutation")


            # --- Download predictions ---
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name="rul_predictions.csv",
                mime="text/csv"
            )

        except Exception as e:
            logger.exception("Error during prediction:")
            st.error(f"An error occurred: {str(e)}")

else:
    st.info("Upload the processed FD001_test_features.csv file to start analysis.")
