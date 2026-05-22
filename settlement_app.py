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
    layout="centered"
)

# Background and style
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #eef7f2 0%, #dceefb 50%, #f7f4ea 100%);
    }

    h1, h2, h3 {
        color: #12372A;
    }

    .stButton > button {
        background-color: #1B6B50;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
    }

    .stButton > button:hover {
        background-color: #144D3A;
        color: white;
    }

    [data-testid="stMetric"] {
        background-color: white;
        padding: 16px;
        border-radius: 8px;
        border: 1px solid #d9e2dc;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Settlement Prediction App")
st.subheader("Prediction of 7-Year Embankment Settlement")


# ============================================================
# 6. USER INPUT SECTION
# ============================================================
st.header("Input Parameters")

gamma_emb = st.number_input("γemb (kN/m³)", min_value=0.0, value=17.0, step=0.1)
Hemb = st.number_input("Hemb (m)", min_value=0.0, value=4.0, step=0.1)
B1 = st.number_input("B1 (m)", min_value=0.0, value=3.5, step=0.1)

gamma_soil = st.number_input("γsoil (kN/m³)", min_value=0.0, value=12.0, step=0.1)
gamma_sat = st.number_input("γsat (kN/m³)", min_value=0.0, value=15.0, step=0.1)

GWL = st.number_input("GWL (m)", min_value=0.0, value=0.5, step=0.1)
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

    st.metric(
        "Average ML Total Settlement",
        f"{average_mm:.2f} mm"
    )

    st.write(f"GPR prediction uncertainty: **±{gpr_uncertainty_mm:.2f} mm**")

    st.subheader("Settlement Prediction Graph")

    graph_data = pd.DataFrame({
        "Model": ["ANN", "SVR", "GPR", "Average"],
        "Settlement (mm)": [ann_mm, svr_mm, gpr_mm, average_mm]
    })

    st.bar_chart(
        graph_data,
        x="Model",
        y="Settlement (mm)"
    )

    if average_mm <= allowable_settlement_mm:
        st.success("Status: SAFE based on average ML settlement and allowable settlement limit")
    else:
        st.error("Status: NOT SAFE based on average ML settlement and allowable settlement limit")


    # ========================================================
    # RECOMMENDED SAFE EMBANKMENT HEIGHT
    # ========================================================
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

        st.bar_chart(
            safe_graph_data,
            x="Model",
            y="Settlement (mm)"
        )


# ============================================================
# 8. DISCLAIMER AND COPYRIGHT
# ============================================================
st.info(
    "Note: The app predicts 7-year settlement using ANN, SVR, and GPR models. "
    "The Average ML Total Settlement is the average of the three model predictions. "
    "Final embankment design should be verified by a qualified geotechnical engineer."
)

st.markdown("---")
st.caption("© 2026 Hanim 🍀. All rights reserved.")
