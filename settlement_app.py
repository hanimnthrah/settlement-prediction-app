import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================================================
# 1. LOAD TRAINED MODELS AND SCALERS
# ============================================================
ann = joblib.load("settlement_ann.pkl")
svr = joblib.load("settlement_svr.pkl")
gpr = joblib.load("settlement_gpr.pkl")

scaler_ann = joblib.load("scaler_ann.pkl")
scaler_svr = joblib.load("scaler_svr.pkl")
scaler_gpr = joblib.load("scaler_gpr.pkl")


# ============================================================
# 2. INPUT FEATURES
# ============================================================
features = [
    '𝛾emb (kN/m3)',
    'Hemb (m)',
    'B1 (m)',
    '𝛾soil (kN/m3)',
    '𝛾sat (kN/m3)',
    'GWL (m)',
    'Dpeat (m)',
    'Cc',
    'eo'
]


# ============================================================
# 3. PREDICTION FUNCTIONS
# ============================================================
def make_input_data(input_values):
    return pd.DataFrame([input_values], columns=features)


def predict_ann(input_values):
    input_data = make_input_data(input_values)
    input_scaled = scaler_ann.transform(input_data)
    prediction = ann.predict(input_scaled)[0]
    return prediction


def predict_svr(input_values):
    input_data = make_input_data(input_values)
    input_scaled = scaler_svr.transform(input_data)
    prediction = svr.predict(input_scaled)[0]
    return prediction


def predict_gpr(input_values):
    input_data = make_input_data(input_values)
    input_scaled = scaler_gpr.transform(input_data)
    prediction, uncertainty = gpr.predict(input_scaled, return_std=True)
    return prediction[0], uncertainty[0]


def predict_all_models(input_values):
    ann_pred = predict_ann(input_values)
    svr_pred = predict_svr(input_values)
    gpr_pred, gpr_uncertainty = predict_gpr(input_values)

    average_pred = np.mean([ann_pred, svr_pred, gpr_pred])

    return ann_pred, svr_pred, gpr_pred, gpr_uncertainty, average_pred


# ============================================================
# 4. SAFE HEIGHT RECOMMENDATION FUNCTION
# ============================================================
def recommend_safe_height(
    gamma_emb, B1,
    gamma_soil, gamma_sat,
    GWL, Dpeat, Cc, eo,
    allowable_settlement_mm
):
    safe_height = None
    safe_ann = None
    safe_svr = None
    safe_gpr = None
    safe_average = None
    safe_uncertainty = None

    height_values = np.arange(0.5, 10.1, 0.1)

    for Hemb in height_values:
        input_values = [
            gamma_emb, Hemb, B1,
            gamma_soil, gamma_sat,
            GWL, Dpeat, Cc, eo
        ]

        ann_pred, svr_pred, gpr_pred, gpr_uncertainty, average_pred = predict_all_models(input_values)

        average_mm = average_pred * 1000

        if average_mm <= allowable_settlement_mm:
            safe_height = Hemb
            safe_ann = ann_pred * 1000
            safe_svr = svr_pred * 1000
            safe_gpr = gpr_pred * 1000
            safe_average = average_mm
            safe_uncertainty = gpr_uncertainty * 1000

    return safe_height, safe_ann, safe_svr, safe_gpr, safe_average, safe_uncertainty


# ============================================================
# 5. APP LAYOUT
# ===============================================
