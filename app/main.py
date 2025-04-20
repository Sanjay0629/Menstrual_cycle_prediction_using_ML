import streamlit as st
import joblib
import datetime
import numpy as np
import pandas as pd
import os

# Load model
model = joblib.load("models/menstrual_model_xgb.pkl")
HISTORY_FILE = "prediction_history.csv"

# Title
st.title("🌸 Menstrual Cycle Predictor")
st.write("Enter your cycle details and symptoms to predict your next period date.")

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

# Predict
if submitted:
    mood_map = {"Happy": 0, "Normal": 1, "Moody": 2, "Sad": 3}
    cramps_map = {"None": 0, "Mild": 1, "Severe": 2}
    mood_encoded = mood_map[mood]
    cramps_encoded = cramps_map[cramps]

    cycle_variability = abs(mean_cycle_length - period_length)
    symptom_score = mood_encoded + cramps_encoded

    input_data = np.array([[mean_cycle_length, luteal_phase, period_length,
                            peak_days, menses_score, age, bmi,
                            mood_encoded, cramps_encoded,
                            cycle_variability, symptom_score]])

    predicted_cycle_length = model.predict(input_data)[0]
    next_period_date = last_period_date + datetime.timedelta(days=round(predicted_cycle_length))

    st.success(f"✅ Predicted Cycle Length: {round(predicted_cycle_length)} days")
    st.info(f"📅 Estimated Next Period Start Date: **{next_period_date.strftime('%B %d, %Y')}**")

    # Save to CSV
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

    # Show next 4 predicted periods
    st.subheader("📅 Upcoming Predicted Periods")
    upcoming_periods = []
    current_date = next_period_date
    for i in range(4):
        upcoming_periods.append({"Cycle #": i + 1, "Start Date": current_date.strftime('%Y-%m-%d')})
        current_date += datetime.timedelta(days=round(predicted_cycle_length))
    st.table(upcoming_periods)

# Show prediction history
if os.path.exists(HISTORY_FILE):
    st.subheader("🕓 Prediction History")
    history = pd.read_csv(HISTORY_FILE)
    st.dataframe(history[::-1], use_container_width=True)

# Clear history button
if os.path.exists(HISTORY_FILE):
    st.subheader("🗑️ Clear Prediction History")
    if st.button("Clear History"):
        os.remove(HISTORY_FILE)
        st.success("Prediction history cleared!")
        st.rerun()

# Trend chart
if os.path.exists(HISTORY_FILE):
    st.subheader("📈 Cycle Length Trend Over Time")
    history = pd.read_csv(HISTORY_FILE)

    if "Prediction Date" in history.columns and "Predicted Cycle Length" in history.columns:
        try:
            history["Prediction Date"] = pd.to_datetime(history["Prediction Date"], errors="coerce")
            trend_data = history[["Prediction Date", "Predicted Cycle Length"]].dropna()
            trend_data = trend_data.sort_values("Prediction Date")

            if not trend_data.empty:
                trend_data = trend_data.set_index("Prediction Date")
                st.line_chart(trend_data)
            else:
                st.warning("No valid prediction data to plot.")
        except Exception as e:
            st.error(f"Error processing trend chart: {e}")
    else:
        st.warning("Missing columns for trend chart.")

# 📈 Combined Mood & Cramps Trend Chart
if os.path.exists(HISTORY_FILE):
    st.subheader("📊 Mood & Cramp Trends (Combined)")
    history = pd.read_csv(HISTORY_FILE)

    if "Prediction Date" in history.columns and "Mood" in history.columns and "Cramps" in history.columns:
        try:
            history["Prediction Date"] = pd.to_datetime(history["Prediction Date"], errors="coerce")
            history = history.sort_values("Prediction Date")

            # Convert to scores
            mood_map = {"Happy": 0, "Normal": 1, "Moody": 2, "Sad": 3}
            cramp_map = {"None": 0, "Mild": 1, "Severe": 2}

            history["Mood Score"] = history["Mood"].map(mood_map)
            history["Cramp Score"] = history["Cramps"].map(cramp_map)

            # Combine for one chart
            combined = history[["Prediction Date", "Mood Score", "Cramp Score"]].dropna()
            combined.set_index("Prediction Date", inplace=True)

            st.line_chart(combined)

        except Exception as e:
            st.error(f"Error creating combined trend chart: {e}")

