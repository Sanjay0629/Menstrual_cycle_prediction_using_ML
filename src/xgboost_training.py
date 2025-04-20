# src/xgboost_training.py

from preprocessing import load_and_preprocess_data
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt

def train_xgboost_model(data_path: str, model_path: str = "models/menstrual_model_xgb.pkl"):
    # Step 1: Load preprocessed data
    X, y = load_and_preprocess_data(data_path)

    # Step 2: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 3: Train XGBoost Regressor
    model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    # Step 4: Predict and evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("✅ XGBoost model trained successfully.")
    print(f"📊 Mean Absolute Error: {mae:.2f}")
    print(f"📈 R² Score: {r2:.2f}")

    # Step 5: Save the model
    joblib.dump(model, model_path)
    print(f"💾 XGBoost model saved to: {model_path}")

    # Step 6: Plot Feature Importance
    importances = model.feature_importances_
    feature_names = X.columns
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, importances)
    plt.xlabel("Importance")
    plt.title("XGBoost Feature Importance")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Run directly
if __name__ == "__main__":
    train_xgboost_model("data/menstrual_data_with_symptoms.csv")
