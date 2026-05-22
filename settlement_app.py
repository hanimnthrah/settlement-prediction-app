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
# ============================================================
st.set_page_config(
    page_title="Settlement Prediction App",
    layout="wide"
)

st.markdown(
    """
    <style>
    .stApp {
        background: #f4f7f6;
        color: #17202A;
    }

    .block-container {
        padding-top: 2rem;
        max-width: 1180px;
    }

    .main-title {
        background: linear-gradient(90deg, #12372A, #1B6B50);
        padding: 28px 34px;
        border-radius: 10px;
        color: #dceee7 !important;
        margin-bottom: 28px;
    }

    .main-title h1 {
        color: #dceee7 !important;
        margin-bottom: 4px;
        font-size: 42px;
        letter-spacing: 0;
    }

    .main-title p {
        color: #dceee7 !important;
        font-size: 18px;
        margin: 0;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #12372A !important;
    }

    p, label, span {
        color: #17202A !important;
    }

    [data-testid="stMarkdownContainer"] {
        color: #17202A !important;
    }

    [data-testid="stWidgetLabel"] {
        color: #17202A !important;
        font-weight: 600;
    }

    [data-testid="stNumberInput"] {
        background: white;
        padding: 8px 10px 12px 10px;
        border-radius: 8px;
        border: 1px solid #d7e1dc;
    }

    [data-testid="stNumberInput"] label,
    [data-testid="stNumberInput"] label p {
        color: #17202A !important;
        font-weight: 600;
    }

    [data-testid="stNumberInput"] input {
        background-color: #ffffff !important;
        color: #17202A !important;
        border: 1px solid #cfd8d3 !important;
    }

    [data-testid="stNumberInput"] button {
        background-color: #eef3f1 !important;
        color: #17202A !important;
        border: 1px solid #cfd8d3 !important;
    }

    .stButton > button {
        background-color: #1B6B50;
        color: white !important;
        border-radius: 8px;
        border: none;
        padding: 0.7rem 1.4rem;
        font-weight: 700;
        width: 100%;
    }

    .stButton > button:hover {
        background-color: #144D3A;
        color: white !important;
    }

    [data-testid="stMetric"] {
        background-color: white;
        padding: 18px;
        border-radius: 8px;
        border: 1px solid #d9e2dc;
    }

    [data-testid="stMetric"] * {
        color: #17202A !important;
    }

    [data-testid="stDataFrame"] {
        background: white;
        border-radius: 8px;
    }

    .stAlert * {
        color: inherit !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="main-title">
        <h1>Settlement Prediction App</h1>
        <p>Prediction of 7-Year Embankment Settlement using ANN, SVR, and GPR</p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 6. USER INPUT SECTION
# ============================================================
st.header("Input Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    gamma_emb = st.number_input("γemb (kN/m³)", min_value=0.0, value=17.0, step=0.1)
    Hemb = st.number_input("Hemb (m)", min_value=0.0, value=4.0, step=0.1)
    B1 = st.number_input("B1 (m)", min_value=0.0, value=3.5, step=0.1)

with col2:
    gamma_soil = st.number_input("γsoil (kN/m³)", min_value=0.0, value=12.0, step=0.1)
    gamma_sat = st.number_input("γsat (kN/m³)", min_value=0.0, value=15.0, step=0.1)
    GWL = st.number_input("GWL (m)", min_value=0.0, value=0.5, step=0.1)

with col3:
    Dpeat = st.number_input("Dpeat (m)", min_value=0.0, value=3.0, step=0.1)
    Cc = st.number_input("Cc", min_value=0.0, value=3.0, step=0.1)
    eo = st.number_input("eo", min_value=0.0, value=8.0, step=0.1)

allowable_settlement_mm = st.number_input(
    "Allowable Settlement Limit (mm)",
    min_value=0.0,
    value=50.0,
    step=1.0
)


# ============================================================
# 7. PREDICTION BUTTON
# ============================================================
if st.button("Calculate Settlement"):

    input_values = [
        gamma_emb, Hemb, B1,
        gamma_soil, gamma_sat,
        GWL, Dpeat, Cc, eo
    ]

    ann_pred, svr_pred, gpr_pred, gpr_uncertainty, average_pred = predict_all_models(input_values)

    ann_mm = ann_pred * 1000
    svr_mm = svr_pred * 1000
    gpr_mm = gpr_pred * 1000
    average_mm = average_pred * 1000
    gpr_uncertainty_mm = gpr_uncertainty * 1000

    st.header("Settlement Prediction Results")

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        st.metric("ANN", f"{ann_mm:.2f} mm")
    with metric_col2:
        st.metric("SVR", f"{svr_mm:.2f} mm")
    with metric_col3:
        st.metric("GPR", f"{gpr_mm:.2f} mm")
    with metric_col4:
        st.metric("Average ML", f"{average_mm:.2f} mm")

    result_table = pd.DataFrame({
        "Model": ["ANN", "SVR", "GPR", "Average ML Total Settlement"],
        "Predicted Settlement (m)": [
            ann_pred,
            svr_pred,
            gpr_pred,
            average_pred
        ],
        "Predicted Settlement (mm)": [
            ann_mm,
            svr_mm,
            gpr_mm,
            average_mm
        ]
    })

    st.dataframe(result_table, use_container_width=True)

    st.write(f"GPR prediction uncertainty: **±{gpr_uncertainty_mm:.2f} mm**")

    st.subheader("Settlement Prediction Graph")

    graph_data = pd.DataFrame({
        "Model": ["ANN", "SVR", "GPR", "Average"],
        "Settlement (mm)": [ann_mm, svr_mm, gpr_mm, average_mm]
    })

    st.bar_chart(graph_data, x="Model", y="Settlement (mm)")

    if average_mm <= allowable_settlement_mm:
        st.success("Status: SAFE based on average ML settlement and allowable settlement limit")
    else:
        st.error("Status: NOT SAFE based on average ML settlement and allowable settlement limit")

    st.header("Recommended Safe Embankment Height")

    (
        safe_height,
        safe_ann,
        safe_svr,
        safe_gpr,
        safe_average,
        safe_uncertainty
    ) = recommend_safe_height(
        gamma_emb=gamma_emb,
        B1=B1,
        gamma_soil=gamma_soil,
        gamma_sat=gamma_sat,
        GWL=GWL,
        Dpeat=Dpeat,
        Cc=Cc,
        eo=eo,
        allowable_settlement_mm=allowable_settlement_mm
    )

    if safe_height is None:
        st.warning("No safe embankment height was found between 0.5 m and 10.0 m.")
    else:
        st.success(f"Recommended maximum safe Hemb: {safe_height:.2f} m")

        safe_table = pd.DataFrame({
            "Model": ["ANN", "SVR", "GPR", "Average ML Total Settlement"],
            "Settlement at Safe Hemb (mm)": [
                safe_ann,
                safe_svr,
                safe_gpr,
                safe_average
            ]
        })

        st.dataframe(safe_table, use_container_width=True)

        st.write(f"GPR uncertainty at safe Hemb: **±{safe_uncertainty:.2f} mm**")

        st.subheader("Settlement at Recommended Safe Height")

        safe_graph_data = pd.DataFrame({
            "Model": ["ANN", "SVR", "GPR", "Average"],
            "Settlement (mm)": [
                safe_ann,
                safe_svr,
                safe_gpr,
                safe_average
            ]
        })

        st.bar_chart(safe_graph_data, x="Model", y="Settlement (mm)")


# ============================================================
# 8. DISCLAIMER AND COPYRIGHT
# ============================================================
st.info(
    "Note: The app predicts 7-year settlement using ANN, SVR, and GPR models. "
    "The Average ML Total Settlement is the average of the three model predictions. "
    "Final embankment design should be verified by a qualified geotechnical engineer."
)

st.markdown("---")
st.caption("© 2026 N.H.N 🍀. All rights reserved.")
