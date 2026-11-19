import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Machine Failure Predictor", page_icon="⚙️")

# Load the trained model and the column order it expects
saved = joblib.load("model.pkl")
model, columns = saved["model"], saved["columns"]

st.title("⚙️ Machine Failure Predictor")
st.write("Enter today's sensor readings to check the risk of failure.")

machine_type = st.selectbox("Machine type", ["L", "M", "H"],
                            format_func=lambda t: {"L": "L – Low grade", "M": "M – Medium grade", "H": "H – High grade"}[t])
col1, col2 = st.columns(2)
with col1:
    air_temp = st.slider("Air temperature (°C)", 15.0, 40.0, 27.0, 0.1)
    process_temp = st.slider("Process temperature (°C)", 25.0, 50.0, 37.0, 0.1)
    rpm = st.slider("Rotational speed (RPM)", 700, 2300, 1500, 10)
    torque = st.slider("Torque (Nm)", 0.0, 80.0, 40.0, 0.5)
with col2:
    tool_wear = st.slider("Tool wear (minutes)", 0, 250, 100)
    vibration = st.slider("Vibration (mm/s)", 0.0, 15.0, 5.0, 0.1)
    load = st.slider("Load (%)", 20.0, 100.0, 70.0, 0.5)
    hours = st.slider("Hours since last service", 0, 2500, 800, 10)

if st.button("Check failure risk", type="primary"):
    row = pd.DataFrame([{
        "air_temp_c": air_temp, "process_temp_c": process_temp,
        "rotational_speed_rpm": rpm, "torque_nm": torque,
        "tool_wear_min": tool_wear, "vibration_mm_s": vibration,
        "load_percent": load, "hours_since_service": hours,
        "machine_type": machine_type,
    }])
    row = pd.get_dummies(row, dtype=int).reindex(columns=columns, fill_value=0)
    risk = model.predict_proba(row)[0][1]

    st.metric("Failure risk", f"{risk:.0%}")
    if risk < 0.30:
        st.success("🟢 Low risk – machine is healthy.")
    elif risk < 0.60:
        st.warning("🟠 Medium risk – plan a check this week.")
    else:
        st.error("🔴 High risk – schedule maintenance now.")
