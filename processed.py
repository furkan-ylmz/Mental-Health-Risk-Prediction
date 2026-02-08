import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from scipy.stats import zscore

df = pd.read_csv("mental_health_dataset.csv")

corr_matrix = df.corr(numeric_only=True)

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Korelasyon Matrisi Isı Haritası")
plt.tight_layout()
plt.show()

numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
z_scores = df[numeric_cols].apply(zscore)
outlier_mask = (z_scores.abs() > 3).any(axis=1)
outlier_count = outlier_mask.sum()
print(f"Aykırı değer içeren satır sayısı: {outlier_count}")

drop_cols = [
    "depression_score"
]

df = df.drop(columns=drop_cols)

cat_cols = [
    "gender",
    "employment_status",
    "work_environment",
    "mental_health_history",
    "seeks_treatment",
    "mental_health_risk"
]

for col in cat_cols:
    if col in df.columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

df = df.dropna()

df.to_csv("mental_health_data_processed.csv", index=False)
print("Ön işlenmiş veri 'mental_health_data_processed.csv' olarak kaydedildi.")