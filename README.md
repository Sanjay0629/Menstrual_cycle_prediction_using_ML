
# 🌸 Menstrual Cycle Predictor using Machine Learning

An intelligent and user-personalized menstrual cycle prediction app built with **Streamlit**, **XGBoost & Random Forest**, and **SQLite**. It allows users to track their cycles, mood, and cramps, and provides accurate predictions based on user symptoms and cycle patterns.

---

## 🚀 Features

- 🔐 **Login & Signup System** using SQLite (secure password hashing with SHA-256)
- 🔮 **Cycle Length Prediction** using an XGBoost regression model
- 🩸 **Symptom Tracking** (Mood + Cramps) via Streamlit UI
- 📈 **Trend Visualizations**:
  - Predicted Cycle Length over time
  - Mood and Cramps scores combined in a single chart
- 📅 **Upcoming Period Forecast** – displays next 4 predicted periods
- 🧠 **User-Specific History** – users can view only their past predictions
- 🧹 **Clear History** per user

---

## 🧠 Machine Learning

- Model: `XGBoostRegressor & Random Forest`
- Trained on a symptom-enhanced menstrual dataset
- Features used:
  - Mean Cycle Length
  - Luteal Phase
  - Period Length
  - Peak Days
  - Menses Score
  - Age
  - BMI
  - Mood Score
  - Cramp Score
  - Cycle Variability
  - Symptom Score

---

## 📁 Project Structure

```
MenstrualCyclePredictionApp/
├── app/
│   └── main.py                       # Streamlit app
├── data/
│   └── prediction_history.csv        # User-specific predictions                      
├── models/
│   └── menstrual_model_xgb.pkl       # Trained XGBoost model
├── src/
│   ├── preprocessing.py              # Data preprocessing
│   ├── model_training.py             # RandomForest training (archived)
│   └── xgboost_training.py           # XGBoost training
├── enhanced_dataset/
│   └── menstrual_data_with_symptoms.csv
├── requirements.txt
└── README.md
└── users.db                           # SQLite DB for user authentication
```

---

## ⚙️ Installation

1. **Clone the repo**  
```bash
git clone https://github.com/your-username/MenstrualCyclePredictionApp.git
cd MenstrualCyclePredictionApp
```

2. **Install dependencies**  
```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app**  
```bash
streamlit run app/main.py
```

---

## ✅ Requirements

```text
streamlit
pandas
numpy
scikit-learn
xgboost
joblib
matplotlib
```

> These are listed in `requirements.txt`

---

## 📄 License

This project is for academic and educational use only.
