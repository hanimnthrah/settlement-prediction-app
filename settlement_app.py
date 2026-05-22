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
    }

    .main-title {
        background: linear-gradient(90deg, #12372A, #1B6B50);
        padding: 28px 34px;
        border-radius: 10px;
        color: white;
        margin-bottom: 28px;
    }

    .main-title h1 {
        color: white;
        margin-bottom: 4px;
        font-size: 42px;
    }

    .main-title p {
        color: #dceee7;
        font-size: 18px;
        margin: 0;
    }

    h2, h3 {
        color: #12372A;
    }

    [data-testid="stNumberInput"] {
        background: white;
        padding: 8px 10px 12px 10px;
        border-radius: 8px;
        border: 1px solid #e1e8e4;
    }

    .stButton > button {
        background-color: #1B6B50;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.7rem 1.4rem;
        font-weight: 700;
        width: 100%;
    }

    .stButton > button:hover {
        background-color: #144D3A;
        color: white;
    }

    [data-testid="stMetric"] {
        background-color: white;
        padding: 18px;
        border-radius: 8px;
        border: 1px solid #d9e2dc;
    }

    [data-testid="stDataFrame"] {
        background: white;
        border-radius: 8px;
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
