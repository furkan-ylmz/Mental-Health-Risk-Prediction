"""
Data Preprocessing Pipeline for Mental Health Risk Prediction
Loads raw data, performs outlier analysis, eliminates leakage features,
applies label encoding, and saves the cleaned dataset.
"""

import os
import pandas as pd
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder


CATEGORICAL_COLUMNS = [
    "gender",
    "employment_status",
    "work_environment",
    "mental_health_history",
    "seeks_treatment",
    "mental_health_risk",
]


def load_raw_data(filepath: str = "data/raw/mental_health_dataset.csv") -> pd.DataFrame:
    """Load raw dataset from CSV file."""
    if not os.path.exists(filepath):
        if os.path.exists("mental_health_dataset.csv"):
            filepath = "mental_health_dataset.csv"
        else:
            raise FileNotFoundError(f"Raw dataset not found at: {filepath}")
    return pd.read_csv(filepath)


def detect_outliers(df: pd.DataFrame, threshold: float = 3.0) -> int:
    """Detect number of rows containing numeric outliers based on Z-score."""
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    z_scores = df[numeric_cols].apply(zscore)
    outlier_mask = (z_scores.abs() > threshold).any(axis=1)
    outlier_count = int(outlier_mask.sum())
    print(f"Aykiri deger kontrolu: |Z-skor| > {threshold} olan {outlier_count} satir tespit edildi.")
    return outlier_count


def preprocess_data(
    raw_path: str = "data/raw/mental_health_dataset.csv",
    output_path: str = "data/processed/mental_health_data_processed.csv",
) -> pd.DataFrame:
    """
    Main preprocessing routine:
    1. Reads raw data.
    2. Drops depression_score (prevents target leakage).
    3. Encodes categorical columns via LabelEncoder.
    4. Saves single cleaned dataset to data/processed/.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = load_raw_data(raw_path)
    print(f"Ham veri seti yuklendi: {df.shape[0]} satir, {df.shape[1]} sutun.")

    detect_outliers(df)

    # Drop target leakage column
    if "depression_score" in df.columns:
        df = df.drop(columns=["depression_score"])
        print("Bilgi sizintisini onlemek icin 'depression_score' sutunu cikarildi.")

    # Encode categorical columns
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    df = df.dropna()

    df.to_csv(output_path, index=False)
    print(f"On islenmis veri basariyla kaydedildi -> {output_path} ({df.shape[0]} satir, {df.shape[1]} sutun)")
    return df


if __name__ == "__main__":
    preprocess_data()
