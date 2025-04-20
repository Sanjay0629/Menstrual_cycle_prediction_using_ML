import streamlit as st
import joblib
import datetime
import numpy as np
import pandas as pd
import os
import sqlite3
import hashlib

# ========== DATABASE SETUP ==========
DB_PATH = "users.db"
HISTORY_FILE = "prediction_history.csv"

def create_users_table():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            email TEXT UNIQUE,
            password TEXT
        )
    ''')
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def add_user(username, email, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
              (username, email, hash_password(password)))
    conn.commit()
    conn.close()

def verify_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username=? AND password=?',
              (username, hash_password(password)))
    result = c.fetchone()
    conn.close()
    return result

# ========== STREAMLIT APP START ==========

st.set_page_config(page_title="Cycle Predictor", page_icon="🌸")
create_users_table()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ========== LOGIN / SIGNUP ==========
if not st.session_state.logged_in:
    st.title("🔐 Login / Signup")

    action = st.radio("Choose Action", ["Login", "Signup"])

    if action == "Signup":
        st.subheader("📨 Create New Account")
        new_user = st.text_input("Username")
        new_email = st.text_input("Email")
        new_password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        if st.button("Create Account"):
            if new_password != confirm_password:
                st.error("Passwords do not match.")
            elif not new_user or not new_email:
                st.warning("All fields are required.")
            else:
                try:
                    add_user(new_user, new_email, new_password)
                    st.success("Account created successfully. Please login.")
                except sqlite3.IntegrityError:
                    st.error("Username or email already exists.")

    else:  # Login
        st.subheader("🔑 Login to Continue")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if verify_user(username, password):
                st.session_state.logged_in = True
                st.session_state.user = username
                st.success(f"Welcome, {username}!")
                st.rerun()
            else:
                st.error("Invalid credentials.")
    st.stop()

# ========== LOGGED-IN VIEW ==========
st.title("🌸 Menstrual Cycle Predictor")
st.caption(f"Logged in as **{st.session_state.user}**")

if st.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.rerun()

# ========== PREDICTION FORM ==========
model = joblib.load("models/menstrual_model_xgb.pkl")

with st.form("cycle_form"):
    last_period_date = st.date_input("🩸 Last Period Start Date")
    mean_cycle_length = st.slider("📆 Average Cycle Length", 20, 40, 28)
    luteal_phase = st.slider("🧪 Luteal Phase Length", 10, 16, 12)
    period_length = st.slider("🩸 Period Length", 2, 8, 5)
    peak_days = st.slider("🌟 Number of Peak Days", 0, 5, 2)
    menses_score = st.slider("🩸 Menses Intensity Score", 1, 100, 50)
    age = st.number_input("🎂 Age", min_value=10, max_value=60, value=25)
    bmi = st.number_input("⚖️ BMI", min_value=10.0, max_value=40.0, value=21.0)
    mood = st.selectbox("🧠 Mood", ["Happy", "Normal", "Moody", "Sad"])
    cramps = st.selectbox("⚡ Cramps Level", ["None", "Mild", "Severe"])
    submitted = st.form_submit_button("🔮 Predict Next Cycle")

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

    entry = {
        "Username": st.session_state.user,
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

    # Upcoming periods
    st.subheader("📅 Upcoming Predicted Periods")
    future_periods = []
    for i in range(4):
        start_date = next_period_date + datetime.timedelta(days=round(predicted_cycle_length) * i)
        future_periods.append({"Cycle #": i+1, "Start Date": start_date.strftime('%Y-%m-%d')})
    st.table(future_periods)

# ========== HISTORY TABLE ==========
if os.path.exists(HISTORY_FILE):
    st.subheader("🕓 Prediction History")
    df = pd.read_csv(HISTORY_FILE)
    user_history = df[df["Username"] == st.session_state.user]
    st.dataframe(user_history[::-1], use_container_width=True)

# ========== CLEAR HISTORY ==========
if os.path.exists(HISTORY_FILE):
    st.subheader("🗑️ Clear Your Prediction History")
    if st.button("Clear History"):
        df = pd.read_csv(HISTORY_FILE)
        df = df[df["Username"] != st.session_state.user]
        df.to_csv(HISTORY_FILE, index=False)
        st.success("Your history has been cleared!")
        st.rerun()

# ========== TREND CHARTS ==========
if os.path.exists(HISTORY_FILE):
    st.subheader("📈 Cycle, Mood & Cramp Trends")
    df = pd.read_csv(HISTORY_FILE)
    df = df[df["Username"] == st.session_state.user]

    if not df.empty:
        df["Prediction Date"] = pd.to_datetime(df["Prediction Date"], errors="coerce")
        df = df.sort_values("Prediction Date")

        # Map scores
        df["Mood Score"] = df["Mood"].map({"Happy": 0, "Normal": 1, "Moody": 2, "Sad": 3})
        df["Cramp Score"] = df["Cramps"].map({"None": 0, "Mild": 1, "Severe": 2})

        # ===== 📉 Cycle Length Chart =====
        cycle_chart = df[["Prediction Date", "Predicted Cycle Length"]].dropna()
        cycle_chart = cycle_chart.set_index("Prediction Date")
        st.write("📊 Predicted Cycle Length Over Time")
        st.line_chart(cycle_chart)

        # ===== 🧠 Mood + ⚡ Cramp Chart =====
        mood_cramp_chart = df[["Prediction Date", "Mood Score", "Cramp Score"]].dropna()
        mood_cramp_chart = mood_cramp_chart.set_index("Prediction Date")
        st.write("🧠 Mood & ⚡ Cramps Trend Over Time")
        st.line_chart(mood_cramp_chart)

    else:
        st.warning("No prediction history found to plot.")

