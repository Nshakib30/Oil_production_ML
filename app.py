import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

st.set_page_config(
    page_title="Volve Oil Recovery Predictor",
    layout="wide",
)

st.markdown(
    """
    <style>
    .header-title { font-size: 2.1rem; font-weight: 700; color: #9DC4F0; margin-bottom: 0.1rem; }
    .header-sub { font-size: 1rem; color: #B7C3D1; margin-top: 0; margin-bottom: 1.2rem; }
    [data-testid="stMetricValue"] { color: #14315C; font-weight: 700; }
    [data-testid="stMetric"] {
        background-color: white; border: 1px solid #E3E8EE; border-radius: 10px;
        padding: 0.8rem 1rem;
    }
    .stButton>button {
        background-color: #5B9BD5; color: #0F1620; border-radius: 6px;
        font-weight: 600; border: none; padding: 0.6rem 1rem;
    }
    .stButton>button:hover { background-color: #7FB3E8; color: #0F1620; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("feature_columns.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    return model, scaler, feature_cols


@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)


model, scaler, FEATURE_COLS = load_artifacts()
explainer = get_explainer(model)

WELLS = [
    "NO 15/9-F-1 C", "NO 15/9-F-11 H", "NO 15/9-F-12 H",
    "NO 15/9-F-14 H", "NO 15/9-F-15 D", "NO 15/9-F-5 AH",
]

st.markdown('<p class="header-title">Volve Oil Recovery Predictor</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="header-sub">XGBoost regression with SHAP explainability, '
    'trained on the Equinor Volve North Sea field dataset (open access)</p>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Well Operating Conditions")

    well = st.selectbox("Well", WELLS)

    st.subheader("Pressure")
    downhole_p = st.number_input("Downhole Pressure (psi)", 0.0, 400.0, 250.0, step=5.0)
    whp = st.number_input("Wellhead Pressure, WHP (psi)", 0.0, 140.0, 40.0, step=1.0)
    annulus = st.number_input("Annulus Pressure (psi)", 0.0, 30.0, 15.0, step=0.5)
    gauge_valid = st.checkbox("Downhole gauge reading valid", value=True)

    st.subheader("Flow")
    on_stream = st.slider("On-Stream Hours", 0.0, 24.0, 24.0)
    choke = st.slider("Choke Size (%)", 0.0, 100.0, 50.0)
    wht = st.number_input("Wellhead Temperature, WHT (°C)", 0.0, 100.0, 70.0, step=1.0)
    wat_vol = st.number_input("Water Volume (bbl)", 0.0, 1000.0, 50.0, step=10.0)

    predict_clicked = st.button("Predict Oil Recovery", use_container_width=True)

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
    st.session_state.last_shap = None

if predict_clicked:
    row = {c: 0 for c in FEATURE_COLS}
    row["ON_STREAM_HRS"] = on_stream
    row["AVG_ANNULUS_PRESS"] = annulus
    row["AVG_CHOKE_SIZE_P"] = choke
    row["AVG_WHP_P"] = whp
    row["AVG_WHT_P"] = wht
    row["BORE_WAT_VOL"] = wat_vol
    row["PRESSURE_DRAWDOWN"] = downhole_p - whp
    row["DOWNHOLE_GAUGE_VALID"] = int(gauge_valid)
    row[f"WELL_{well}"] = 1

    X_input = pd.DataFrame([row])[FEATURE_COLS]
    X_scaled = scaler.transform(X_input)
    pred = model.predict(X_scaled)[0]
    shap_row = explainer.shap_values(X_scaled)[0]

    st.session_state.last_prediction = pred
    st.session_state.last_shap = pd.Series(shap_row, index=FEATURE_COLS)

col1, col2 = st.columns([1, 1.4])

with col1:
    st.subheader("Predicted Oil Volume")
    pred = st.session_state.last_prediction

    if pred is not None:
        st.metric("Daily oil rate", f"{pred:.1f} bbl/day")
    else:
        st.info("Set parameters in the sidebar and click Predict.")
        pred = 0

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pred,
        number={"suffix": " bbl/day"},
        gauge={
            "axis": {"range": [0, 1000]},
            "bar": {"color": "#14315C"},
            "steps": [
                {"range": [0, 250], "color": "#FBE3E0"},
                {"range": [250, 600], "color": "#FDF1CF"},
                {"range": [600, 1000], "color": "#DFF0E0"},
            ],
        },
    ))
    gauge.update_layout(height=280, margin=dict(t=20, b=10, l=20, r=20))
    st.plotly_chart(gauge, use_container_width=True)

with col2:
    st.subheader("Why this prediction (SHAP)")
    if st.session_state.last_shap is not None:
        contrib = st.session_state.last_shap.sort_values()
        colors = ["#BF616A" if v < 0 else "#2E5984" for v in contrib.values]
        shap_fig = go.Figure(go.Bar(
            x=contrib.values, y=contrib.index, orientation="h",
            marker_color=colors,
        ))
        shap_fig.update_layout(
            height=420,
            margin=dict(t=20, b=10, l=10, r=10),
            xaxis_title="Impact on predicted oil volume (bbl/day)",
        )
        st.plotly_chart(shap_fig, use_container_width=True)
        st.caption(
            "Blue bars push the prediction up, red bars push it down, "
            "relative to the model's average prediction."
        )
    else:
        st.info("Run a prediction to see which inputs drove the result.")

st.markdown("---")
st.subheader("Model Performance (Held-Out Test Set)")
m1, m2, m3 = st.columns(3)
m1.metric("R²", "0.945")
m2.metric("RMSE", "58.78 bbl")
m3.metric("MAE", "30.94 bbl")

st.caption(
    "Tuned XGBoost regression model. SHAP (TreeExplainer) used for feature attribution. "
    "This is a research demo accompanying an academic paper on ML-based oil recovery "
    "prediction and is not intended as a standalone production forecasting tool. "
    "Pressure drawdown, the model's most influential feature, was found to be partially "
    "confounded with well identity in two wells with documented downhole gauge issues; "
    "see the accompanying paper for details."
)
