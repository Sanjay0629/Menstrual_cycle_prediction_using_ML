# src/preprocessing.py

import pandas as pd

def load_and_preprocess_data(filepath: str):
    """
    Load and preprocess the menstrual cycle data including symptoms and engineered features.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        X (DataFrame): Features.
        y (Series): Target for regression.
        y_class (Series): Target for binary classification.
    """
    # Load data
    df = pd.read_csv(filepath)

    # Drop rows with missing values in essential columns
    required_columns = [
        "MeanCycleLength", "LengthofLutealPhase", "LengthofMenses",
        "TotalNumberofPeakDays", "TotalMensesScore", "Age", "BMI",
        "LengthofCycle", "Mood", "Cramps"
    ]
    df = df.dropna(subset=required_columns)

    # Convert numeric columns
    numeric_columns = [
        "MeanCycleLength", "LengthofLutealPhase", "LengthofMenses",
        "TotalNumberofPeakDays", "TotalMensesScore", "Age", "BMI", "LengthofCycle"
    ]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Encode Mood and Cramps
    mood_map = {"Happy": 0, "Normal": 1, "Moody": 2, "Sad": 3}
    cramps_map = {"None": 0, "Mild": 1, "Severe": 2}
    df["Mood"] = df["Mood"].map(mood_map)
    df["Cramps"] = df["Cramps"].map(cramps_map)

    # Feature Engineering
    df["CycleVariability"] = abs(df["LengthofCycle"] - df["MeanCycleLength"])
    df["SymptomScore"] = df["Mood"] + df["Cramps"]

    # Drop rows with any remaining NaNs
    df = df.dropna()

    # Define Binary Classification Target
    # Classify as Short (0) if LengthofCycle < 28, Long (1) if LengthofCycle >= 28
    df["CycleType"] = (df["LengthofCycle"] >= 28).astype(int)

    # Feature matrix (X) and targets (y for regression, y_class for classification)
    X = df[[
        "MeanCycleLength", "LengthofLutealPhase", "LengthofMenses",
        "TotalNumberofPeakDays", "TotalMensesScore", "Age", "BMI",
        "Mood", "Cramps", "CycleVariability", "SymptomScore"
    ]]
    y = df["LengthofCycle"]         # Regression Target
    y_class = df["CycleType"]       # Binary Classification Target

    return X, y, y_class
