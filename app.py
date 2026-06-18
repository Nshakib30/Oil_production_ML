# Install Streamlit if it's not already installed
import streamlit as st
import pandas as pd
import pickle

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('feature_columns.pkl', 'rb') as f:
    FEATURE_COLS = pickle.load(f)

WELLS = ['NO 15/9-F-1 C','NO 15/9-F-11 H','NO 15/9-F-12 H',
         'NO 15/9-F-14 H','NO 15/9-F-15 D','NO 15/9-F-5 AH']

st.title("Volve Oil Recovery Predictor")
st.caption("XGBoost model — Equinor Volve North Sea field dataset")

well = st.selectbox("Well", WELLS)
on_stream = st.slider("On-Stream Hours", 0.0, 24.0, 24.0)
downhole_p = st.number_input("Downhole Pressure", 0.0, 400.0, 250.0)
whp = st.number_input("Wellhead Pressure (WHP)", 0.0, 140.0, 40.0)
wht = st.number_input("Wellhead Temperature (WHT)", 0.0, 100.0, 70.0)
annulus = st.number_input("Annulus Pressure", 0.0, 30.0, 15.0)
choke = st.slider("Choke Size (%)", 0.0, 100.0, 50.0)
wat_vol = st.number_input("Water Volume (bbl)", 0.0, 1000.0, 50.0)
gauge_valid = st.checkbox("Downhole Gauge Valid", value=True)

if st.button("Predict"):
    row = {c: 0 for c in FEATURE_COLS}
    row['ON_STREAM_HRS'] = on_stream
    row['AVG_ANNULUS_PRESS'] = annulus
    row['AVG_CHOKE_SIZE_P'] = choke
    row['AVG_WHP_P'] = whp
    row['AVG_WHT_P'] = wht
    row['BORE_WAT_VOL'] = wat_vol
    row['PRESSURE_DRAWDOWN'] = downhole_p - whp
    row['DOWNHOLE_GAUGE_VALID'] = int(gauge_valid)
    row[f'WELL_{well}'] = 1

    X_input = pd.DataFrame([row])[FEATURE_COLS]  # reindex guarantees exact training order
    X_scaled = scaler.transform(X_input)
    pred = model.predict(X_scaled)[0]
    st.success(f"Predicted Oil Volume: {pred:.2f} bbl/day")
