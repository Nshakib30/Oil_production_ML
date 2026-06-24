import io
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    .section-heading { font-size: 1.2rem; font-weight: 600; color: #EDF1F5; margin-bottom: 0.4rem; }
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
    '<p class="header-sub">XGBoost regression with SHAP explainability — '
    'Equinor Volve North Sea field dataset (open access)</p>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# Sidebar — inputs
# ---------------------------------------------------------------
with st.sidebar:
    st.header("Well Operating Conditions")
    well = st.selectbox("Well", WELLS)

    st.subheader("Pressure & Flow")
    downhole_p  = st.number_input("Downhole Pressure (psi)",       0.0, 400.0, 250.0, step=5.0)
    whp         = st.number_input("Wellhead Pressure, WHP (psi)",  0.0, 140.0,  40.0, step=1.0)
    wht         = st.number_input("Wellhead Temperature, WHT (°C)",0.0, 100.0,  70.0, step=1.0)
    annulus     = st.number_input("Annulus Pressure (psi)",        0.0,  30.0,  15.0, step=0.5)
    choke       = st.number_input("Choke Size (%)",                0.0, 100.0,  50.0, step=1.0)

    st.subheader("Production")
    on_stream   = st.slider("On-Stream Hours", 0.0, 24.0, 24.0)
    gauge_valid = st.checkbox("Downhole gauge reading valid", value=True)

    predict_clicked = st.button("Predict Oil Recovery", use_container_width=True)

# ---------------------------------------------------------------
# Session state
# ---------------------------------------------------------------
for key in ("last_prediction", "last_shap", "last_X_scaled"):
    if key not in st.session_state:
        st.session_state[key] = None

if predict_clicked:
    row = {c: 0 for c in FEATURE_COLS}
    row["ON_STREAM_HRS"]        = on_stream
    row["AVG_ANNULUS_PRESS"]    = annulus
    row["AVG_CHOKE_SIZE_P"]     = choke
    row["AVG_WHP_P"]            = whp
    row["AVG_WHT_P"]            = wht
    row["PRESSURE_DRAWDOWN"]    = downhole_p - whp
    row["DOWNHOLE_GAUGE_VALID"] = int(gauge_valid)
    row[f"WELL_{well}"]         = 1

    X_input  = pd.DataFrame([row])[FEATURE_COLS]
    X_scaled = scaler.transform(X_input)
    pred     = model.predict(X_scaled)[0]
    shap_row = explainer.shap_values(X_scaled)[0]

    st.session_state.last_prediction = pred
    st.session_state.last_shap       = pd.Series(shap_row, index=FEATURE_COLS)
    st.session_state.last_X_scaled   = pd.DataFrame(X_scaled, columns=FEATURE_COLS)

# ---------------------------------------------------------------
# Oil prediction gauge — full width
# ---------------------------------------------------------------
st.markdown('<p class="section-heading">Predicted Oil Volume</p>', unsafe_allow_html=True)
pred = st.session_state.last_prediction or 0

if st.session_state.last_prediction is None:
    st.info("Set parameters in the sidebar and click Predict.")

gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=pred,
    number={"suffix": " bbl/day", "font": {"size": 36}},
    gauge={
        "axis": {"range": [0, 1000]},
        "bar": {"color": "#5B9BD5"},
        "steps": [
            {"range": [0,   250], "color": "#3A1A1A"},
            {"range": [250, 600], "color": "#2A2A1A"},
            {"range": [600,1000], "color": "#1A2E1A"},
        ],
    },
))
gauge.update_layout(
    height=300,
    paper_bgcolor="rgba(0,0,0,0)",
    font_color="#EDF1F5",
    margin=dict(t=20, b=10, l=20, r=20),
)
st.plotly_chart(gauge, use_container_width=True)

# ---------------------------------------------------------------
# SHAP bar + Waterfall — side by side, below gauge
# ---------------------------------------------------------------
if st.session_state.last_shap is not None:
    st.markdown("---")
    st.markdown('<p class="section-heading">SHAP Analysis</p>', unsafe_allow_html=True)

    bar_col, wf_col = st.columns([1.4, 1])

    with bar_col:
        st.caption("Feature Impact — how each variable pushed this prediction")
        contrib = st.session_state.last_shap.sort_values()
        colors  = ["#BF616A" if v < 0 else "#5B9BD5" for v in contrib.values]
        shap_fig = go.Figure(go.Bar(
            x=contrib.values, y=contrib.index,
            orientation="h", marker_color=colors,
        ))
        shap_fig.update_layout(
            height=380,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#EDF1F5",
            margin=dict(t=10, b=10, l=10, r=10),
            xaxis_title="Impact on predicted oil volume (bbl/day)",
            xaxis=dict(gridcolor="#2A3A4A"),
        )
        st.plotly_chart(shap_fig, use_container_width=True)
        st.caption("Blue = pushes prediction up. Red = pushes prediction down.")

    with wf_col:
        st.caption("Waterfall plot — cumulative feature contributions from baseline")
        try:
            exp    = explainer(st.session_state.last_X_scaled)
            fig_wf = plt.figure(figsize=(6, 5))
            fig_wf.patch.set_facecolor("white")
            shap.plots.waterfall(exp[0], max_display=10, show=False)
            for ax in fig_wf.get_axes():
                ax.set_facecolor("white")
                ax.tick_params(colors="black")
                ax.xaxis.label.set_color("black")
                ax.yaxis.label.set_color("black")
                for text in ax.texts:
                    text.set_color("black")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#CCCCCC")
            buf_wf = io.BytesIO()
            plt.savefig(buf_wf, format="png", dpi=150,
                        bbox_inches="tight", facecolor="white")
            plt.close()
            buf_wf.seek(0)
            st.image(buf_wf, use_container_width=True)
        except Exception as e:
            st.warning(f"Waterfall unavailable: {e}")

# ---------------------------------------------------------------
# Model performance
# ---------------------------------------------------------------
st.markdown("---")
st.markdown('<p class="section-heading">Model Performance (Held-Out Test Set)</p>',
            unsafe_allow_html=True)
m1, m2, m3 = st.columns(3)
m1.metric("R²", "0.935", help="Proportion of variance explained. Closer to 1 is better.")
m2.metric("RMSE", "63.99 bbl/day", help="Root Mean Squared Error — average prediction error.")
m3.metric("MAE",  "34.20 bbl/day", help="Mean Absolute Error — typical day-to-day error.")

st.caption(
    "Tuned XGBoost model trained on the Equinor Volve field dataset. "
    "Research demo only — not intended as a production forecasting tool."
)
