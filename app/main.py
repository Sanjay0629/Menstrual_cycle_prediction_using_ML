# app/main.py

import streamlit as st
import joblib
import datetime
import numpy as np
import pandas as pd
import os

# Load XGBoost model
model = joblib.load("models/menstrual_model_xgb.pkl")

# CSV file to store history
HISTORY_FILE = "prediction_history.csv"

# UI Title
st.title("🌸 Menstrual Cycle Predictor")
st.write("Enter your recent cycle details and symptoms to predict your next period date.")

# Form for user input
with st.form("cycle_form"):
    last_period_date = st.date_input("🩸 Last Period Start Date")

    mean_cycle_length = st.slider("📆 Average Cycle Length (days)", 20, 40, 28)
    luteal_phase = st.slider("🧪 Luteal Phase Length", 10, 16, 12)
    period_length = st.slider("🩸 Period Length (days)", 2, 8, 5)
    peak_days = st.slider("🌟 Number of Peak Days", 0, 5, 2)
    menses_score = st.slider("🩸 Menses Intensity Score", 1, 100, 50)
    age = st.number_input("🎂 Age", min_value=10, max_value=60, value=25)
    bmi = st.number_input("⚖️ BMI", min_value=10.0, max_value=40.0, value=21.0)

    mood = st.selectbox("🧠 Mood", ["Happy", "Normal", "Moody", "Sad"])
    cramps = st.selectbox("⚡ Cramps Level", ["None", "Mild", "Severe"])

    submitted = st.form_submit_button("🔮 Predict Next Cycle")

# Predict and display
if submitted:
    # Encode mood & cramps
    mood_map = {"Happy": 0, "Normal": 1, "Moody": 2, "Sad": 3}
    cramps_map = {"None": 0, "Mild": 1, "Severe": 2}
    mood_encoded = mood_map[mood]
    cramps_encoded = cramps_map[cramps]

    # Engineered features
    cycle_variability = abs(mean_cycle_length - period_length)
    symptom_score = mood_encoded + cramps_encoded

    # Final input vector with 11 features
    input_data = np.array([[mean_cycle_length, luteal_phase, period_length,
                            peak_days, menses_score, age, bmi,
                            mood_encoded, cramps_encoded,
                            cycle_variability, symptom_score]])

    predicted_cycle_length = model.predict(input_data)[0]
    next_period_date = last_period_date + datetime.timedelta(days=round(predicted_cycle_length))

    # Show result
    st.success(f"✅ Predicted Cycle Length: {round(predicted_cycle_length)} days")
    st.info(f"📅 Estimated Next Period Start Date: **{next_period_date.strftime('%B %d, %Y')}**")

    # 🔄 Save to CSV history
    entry = {
        "Prediction Date": datetime.date.today(),
        "Last Period Date": last_period_date,
        "Avg Cycle": mean_cycle_length,
        "Period Length": period_length,
        "Mood": mood,
        "Cramps": cramps,
        "Predicted Cycle Length": round(predicted_cycle_length),
        "Next Period Date": next_period_date
    }

    if os.path.exists(HISTORY_FILE):
        history_df = pd.read_csv(HISTORY_FILE)
    else:
        history_df = pd.DataFrame()

    history_df = pd.concat([history_df, pd.DataFrame([entry])], ignore_index=True)
    history_df.to_csv(HISTORY_FILE, index=False)

    # 📅 Show upcoming predicted cycles
    st.subheader("📅 Upcoming Predicted Periods")
    num_cycles = 4
    upcoming_periods = []
    current_date = next_period_date

    for i in range(num_cycles):
        upcoming_periods.append({
            "Cycle ": i + 1,
            "Start Date": current_date.strftime('%d-%m-%y')
        })
        current_date += datetime.timedelta(days=round(predicted_cycle_length))

    st.table(upcoming_periods)

# 📊 Display Prediction History
if os.path.exists(HISTORY_FILE):
    st.subheader("🕓 Prediction History")
    history = pd.read_csv(HISTORY_FILE)
    st.dataframe(history[::-1], use_container_width=True)

# 🧹 Optional: Button to clear prediction history
if os.path.exists(HISTORY_FILE):
    st.subheader("🗑️ Clear Prediction History")
    if st.button("Clear History"):
        os.remove(HISTORY_FILE)
        st.success("Prediction history cleared!")
        st.rerun()

# 📈 Trend Chart: Predicted Cycle Length Over Time
if os.path.exists(HISTORY_FILE):
    st.subheader("📈 Cycle Length Trend Over Time")
    history = pd.read_csv(HISTORY_FILE)

    # Convert date column
    history["Prediction Date"] = pd.to_datetime(history["Prediction Date"], errors="coerce")

    # Sort by date
    history = history.sort_values("Prediction Date")

    # Keep only relevant columns
    trend_data = history[["Prediction Date", "Predicted Cycle Length"]].dropna()

    # Rename for chart
    trend_data = trend_data.set_index("Prediction Date")

    # Plot
    st.line_chart(trend_data)

