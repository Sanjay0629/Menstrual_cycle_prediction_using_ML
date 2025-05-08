# src/xgboost_training.py

import os
from preprocessing import load_and_preprocess_data
from xgboost import XGBRegressor, XGBClassifier, __version__ as xgb_version
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, roc_curve, auc, accuracy_score, log_loss
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Set the working directory to the project root
PROJECT_ROOT = "C:/Users/Sanja/OneDrive/Desktop/Menstrual_cycle_prediction_using_ML"
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "menstrual_data_with_symptoms.csv")
MODEL_PATH_REG = os.path.join(PROJECT_ROOT, "models", "menstrual_model_xgb_reg.pkl")
MODEL_PATH_CLASS = os.path.join(PROJECT_ROOT, "models", "menstrual_model_xgb_class.pkl")

# Extract XGBoost version
xgb_version_major, xgb_version_minor = map(int, xgb_version.split('.')[:2])

def train_xgboost_model(data_path: str):
    # Load preprocessed data
    print("📦 Loading and preprocessing data...")
    X, y, y_class = load_and_preprocess_data(data_path)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42)

    # ===============================
    # TRAINING XGBoost Regressor
    # ===============================
    print("🚀 Training XGBoost Regressor...")
    reg_model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    reg_model.fit(X_train, y_train)
    joblib.dump(reg_model, MODEL_PATH_REG)

    # Evaluate Regressor
    y_pred = reg_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"📊 Regression MAE: {mae:.2f}, R²: {r2:.2f}")

    # ===============================
    # TRAINING XGBoost Classifier
    # ===============================
    print("🚀 Training XGBoost Classifier...")
    class_model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)

    eval_set = [(X_train_class, y_train_class), (X_test_class, y_test_class)]

    # Check if eval_metric is supported
    use_eval_metric = xgb_version_major >= 1 and xgb_version_minor >= 3

    if use_eval_metric:
        print("✅ XGBoost version supports eval_metric. Proceeding with eval_metric.")
        class_model.fit(X_train_class, y_train_class, eval_set=eval_set, eval_metric=["logloss", "error"], verbose=False)
        results = class_model.evals_result()
    else:
        print("⚠️ Older XGBoost version detected. Proceeding without eval_metric.")
        class_model.fit(X_train_class, y_train_class)
        # Manually calculate accuracy and log loss
        results = {
            "validation_0": {"error": [], "logloss": []},
            "validation_1": {"error": [], "logloss": []}
        }
        for epoch in range(1, 101):
            y_train_pred = class_model.predict(X_train_class)
            y_test_pred = class_model.predict(X_test_class)
            train_error = 1 - accuracy_score(y_train_class, y_train_pred)
            test_error = 1 - accuracy_score(y_test_class, y_test_pred)
            train_loss = log_loss(y_train_class, class_model.predict_proba(X_train_class))
            test_loss = log_loss(y_test_class, class_model.predict_proba(X_test_class))
            results["validation_0"]["error"].append(train_error)
            results["validation_1"]["error"].append(test_error)
            results["validation_0"]["logloss"].append(train_loss)
            results["validation_1"]["logloss"].append(test_loss)

    joblib.dump(class_model, MODEL_PATH_CLASS)

    # ===============================
    # PLOT ROC CURVE
    # ===============================
    print("📊 Plotting ROC Curve...")
    y_prob = class_model.predict_proba(X_test_class)[:, 1]
    fpr, tpr, _ = roc_curve(y_test_class, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color='blue')
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # ===============================
    # PLOT LOSS GRAPH
    # ===============================
    print("📊 Plotting Loss Graph...")
    epochs = range(len(results["validation_0"]["logloss"]))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, results["validation_0"]["logloss"], label="Train Log Loss", color='red')
    plt.plot(epochs, results["validation_1"]["logloss"], label="Test Log Loss", color='orange')
    plt.xlabel("Epochs")
    plt.ylabel("Log Loss")
    plt.title("Training vs Testing Log Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # ===============================
    # PLOT ACCURACY GRAPH
    # ===============================
    print("📊 Plotting Accuracy Graph...")
    train_acc = [1 - e for e in results["validation_0"]["error"]]
    test_acc = [1 - e for e in results["validation_1"]["error"]]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_acc, label="Train Accuracy", color='green')
    plt.plot(epochs, test_acc, label="Test Accuracy", color='blue')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training vs Testing Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Show all graphs at once
    plt.show(block=False)

    # Keep the plots open
    input("Press Enter to close all graphs and exit...")

# Run the script
if __name__ == "__main__":
    train_xgboost_model(DATA_PATH)
