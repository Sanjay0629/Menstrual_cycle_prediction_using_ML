# src/model_training.py

from preprocessing import load_and_preprocess_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt

def train_model(data_path: str, model_path: str = "models/menstrual_model.pkl"):
    # Step 1: Load preprocessed data
    X, y = load_and_preprocess_data(data_path)

    # Step 2: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 3: Use best version of RandomForest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Step 4: Predict and evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("✅ Model trained with default parameters.")
    print(f"📊 Mean Absolute Error: {mae:.2f}")
    print(f"📈 R² Score: {r2:.2f}")

    # Step 5: Save model
    joblib.dump(model, model_path)
    print(f"💾 Model saved to: {model_path}")

    # Step 6: Plot Feature Importance
    importances = model.feature_importances_
    feature_names = X.columns
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, importances)
    plt.xlabel("Importance")
    plt.title("Feature Importance")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Run directly
if __name__ == "__main__":
    train_model("data/menstrual_data_with_symptoms.csv")
