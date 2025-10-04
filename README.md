<p align="center">
  <img src="banner.png" alt="NASA Turbofan RUL Project Banner" width="100%">
</p>

<h1 align="center">🚀 NASA Turbofan Engine RUL Prediction (FD001)</h1>
<p align="center">
  End-to-end Machine Learning & Deep Learning pipeline for Predictive Maintenance.<br>
  <strong>By Kuntal (KayDee6070)</strong>
</p>


# NASA Turbofan Engine Remaining Useful Life (RUL) Prediction

This project focuses on **predicting the Remaining Useful Life (RUL)** of jet engines using the **NASA CMAPSS dataset**.  
It combines **classical machine learning (Random Forest, XGBoost)** and **deep learning (LSTM)** to develop predictive maintenance models for turbofan engines.

---

## 📘 Project Overview

The goal of this project is to estimate how many operating cycles remain before an engine fails.  
We use the **FD001 dataset** from NASA's CMAPSS (Commercial Modular Aero-Propulsion System Simulation) data.

The workflow follows a standard machine learning pipeline:

1. **Data Acquisition** → NASA CMAPSS datasets  
2. **Data Preprocessing** → Sensor normalization, feature engineering, and RUL capping  
3. **Model Training** → RandomForest, XGBoost, and LSTM  
4. **Evaluation** → MAE and RMSE comparison  
5. **Visualization & Reporting** → Streamlit dashboard and model report  
6. **Deployment** → Local or online interactive app (Streamlit)

---

## ⚙️ Model Training Summary

| Model                        | MAE   | RMSE  |
|------------------------------|-------|-------|
| RandomForest (all cycles)    | 11.97 | 16.64 |
| RandomForest (last cycle)    | 13.06 | 18.40 |
| XGBoost (all cycles)         | 11.63 | 16.01 |
| XGBoost (last cycle)         | 12.63 | 17.56 |
| LSTM (deep learning)         | 29.68 | 36.63 |

> **Observation:**  
> XGBoost (all cycles) achieved the lowest RMSE, indicating it provides the most stable and accurate predictions for this dataset.

---

## 📊 Model Performance Visualization

Below is a high-resolution comparison of RMSE across all models (lower = better):

<p align="center">
  <img src="models/FD001_comparison.png" alt="Model RMSE Comparison" width="600"/>
</p>

The highlighted blue bar indicates the **best performing model** (XGBoost on all cycles).

---

## 🧠 Streamlit Application

An interactive web app is included to explore and visualize predictions.

### Launch locally

```bash
streamlit run src/app_rul_streamlit.py
```

### App Features
- Upload preprocessed test data (`FD001_test.csv`)
- Choose model: RandomForest, XGBoost, or LSTM
- Predict Remaining Useful Life (RUL)
- Visualize results dynamically

---

## 📁 Project Structure

```
turbofan-rul/
│
├── archive/                   # NASA CMAPSS datasets
├── data/processed/            # Preprocessed CSVs
├── models/                    # Trained models + report
│   ├── FD001_model_report.pdf
│   ├── FD001_comparison.png
│   └── *.joblib / *.h5
│
├── src/
│   ├── data_prep.py           # Data preprocessing
│   ├── train_baseline.py      # ML model training
│   ├── train_lstm.py          # LSTM model training
│   ├── model_report.py        # Generates report and plots
│   └── app_rul_streamlit.py   # Streamlit dashboard
│
└── README.md
```

---

## 🧩 Requirements

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Key packages:
```
pandas
numpy
scikit-learn
xgboost
tensorflow
keras
matplotlib
seaborn
streamlit
joblib
```

---

## 📄 Model Report Example

A detailed model comparison PDF is automatically generated:

- Location: `models/FD001_model_report.pdf`  
- Includes metrics table + RMSE visualization

---

## 🔍 Future Enhancements

- Add **SHAP** interpretability for model explainability  
- Extend to FD002–FD004 datasets  
- Integrate real-time RUL prediction API  
- Add uncertainty quantification (UQ) and calibration metrics  

---

## 🧑‍💻 Author

**Kuntal Dive (KayDee6070)**  
MSc Digital Engineering, Bauhaus-Universität Weimar  
Project Mentor: Self-led applied ML study on Predictive Maintenance  
Repository: [github.com/KayDee6070/turbofan-rul](https://github.com/KayDee6070/turbofan-rul)

---

## 📬 Citation

If you use this project or its structure in your own work, please cite:

> Kuntal Dive, *"NASA Turbofan Engine RUL Prediction using ML and LSTM"*, 2025.

---

