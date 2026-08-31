# -*- coding: utf-8 -*-
"""
Training and Evaluation Pipeline for Mental Health Risk Prediction
Trains MLP, KNN, Decision Tree, and Random Forest across Chi-Square, ANOVA, and RFE feature selection,
evaluates performance, saves the best model, and generates high-resolution analytical charts.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, chi2, f_classif, RFE

from preprocessing import preprocess_data


# Reset styling to ensure clean layout
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.sans-serif": ["Arial", "DejaVu Sans", "Segoe UI"],
    "font.family": "sans-serif",
    "figure.autolayout": False,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})


def generate_correlation_matrix(df_raw: pd.DataFrame, output_path: str = "results/correlation_matrix.png") -> None:
    """Generate correlation heatmap for numeric features."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    numeric_df = df_raw.select_dtypes(include=["float64", "int64"])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(9.5, 7.8), dpi=300)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        linewidths=0.75,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap / Korelasyon Isı Haritası", fontsize=13, pad=18, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.1, dpi=300)
    plt.close(fig)
    print(f"[Grafik] Korelasyon matrisi kaydedildi -> {output_path}")


def generate_metrics_table_image(performance_df: pd.DataFrame, output_path: str = "results/model_evaluation_metrics.png") -> None:
    """
    Generate clean, publication-ready metric summary table with balanced margins.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    table_df = performance_df[["Model", "Feature Selection", "Accuracy", "Precision", "Recall", "F1 Score"]].copy()

    for col in ["Accuracy", "Precision", "Recall", "F1 Score"]:
        table_df[col] = table_df[col].apply(lambda x: f"{x:.4f}")

    fig, ax = plt.subplots(figsize=(11.5, 4.2), dpi=300)
    ax.axis("off")

    col_labels = ["Model", "Feature Selection", "Accuracy", "Precision", "Recall", "F1 Score"]
    col_widths = [0.18, 0.20, 0.155, 0.155, 0.155, 0.155]

    table = ax.table(
        cellText=table_df.values,
        colLabels=col_labels,
        colWidths=col_widths,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.35)

    # Style Header and Alternating Rows
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        if row_idx == 0:
            cell.set_facecolor("#1e3a8a")
            cell.set_text_props(color="white", fontweight="bold")
        elif row_idx % 2 == 1:
            cell.set_facecolor("#f8fafc")
        else:
            cell.set_facecolor("#ffffff")

    ax.set_title("Comparative Model Performance Benchmark / Karşılaştırmalı Model Performans Tablosu", fontsize=13, fontweight="bold", pad=18)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.1, dpi=300)
    plt.close(fig)
    print(f"[Grafik] Model metrik tablosu kaydedildi -> {output_path}")


def generate_performance_comparison(benchmark_results: dict, output_path: str = "results/performance_comparison.png") -> None:
    """Generate 2x2 multi-panel bar charts comparing all models with balanced spacing."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    models = list(benchmark_results.keys())
    methods = ["Chi2", "ANOVA", "RFE"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score"]
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), dpi=300)
    axes = axes.flatten()

    for idx, model_name in enumerate(models):
        ax = axes[idx]
        data = []
        for method in methods:
            m = benchmark_results[model_name][method]["metrics"]
            data.append([m["accuracy"], m["precision"], m["recall"], m["f1"]])

        df_plot = pd.DataFrame(data, index=methods, columns=metric_labels)
        df_plot.plot(kind="bar", ax=ax, color=colors, rot=0, width=0.75, edgecolor="black", linewidth=0.5)
        ax.set_title(f"{model_name} - Performance Across Feature Selection", fontweight="bold", fontsize=11, pad=5)
        ax.set_ylabel("Score", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower right", frameon=True, fontsize=8.5)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

    fig.suptitle("Model Evaluation & Feature Selection Dynamics / Model Değerlendirme ve Özellik Seçimi Dinamikleri", fontsize=13.5, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0.01, 1, 0.95], h_pad=2.2, w_pad=1.8)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.08, dpi=300)
    plt.close(fig)
    print(f"[Grafik] Performans karsilastirma grafigi kaydedildi -> {output_path}")


def generate_confusion_matrices(benchmark_results: dict, output_path: str = "results/confusion_matrices.png") -> None:
    """
    Generate 4x3 grid of confusion matrices with ample spacing preventing any title-axis overlaps.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    models = list(benchmark_results.keys())
    methods = ["Chi2", "ANOVA", "RFE"]
    class_names = ["High", "Low", "Medium"]

    fig, axes = plt.subplots(len(models), len(methods), figsize=(12, 13.5), dpi=300)

    for i, model_name in enumerate(models):
        for j, method in enumerate(methods):
            ax = axes[i, j]
            cm = benchmark_results[model_name][method]["metrics"]["confusion_matrix"]
            acc = benchmark_results[model_name][method]["metrics"]["accuracy"]

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names,
                cbar=False,
                ax=ax,
            )
            ax.set_title(f"{model_name} ({method}) | Acc: {acc:.4f}", fontsize=10.5, fontweight="bold", pad=6)
            if i == len(models) - 1:
                ax.set_xlabel("Predicted Class", fontsize=9.5, labelpad=5)
            else:
                ax.set_xlabel("")

            if j == 0:
                ax.set_ylabel("True Class", fontsize=9.5, labelpad=5)
            else:
                ax.set_ylabel("")

    fig.suptitle("Test Confusion Matrices / Test Konfüzyon Matrisleri", fontsize=14, fontweight="bold", y=0.985)
    fig.tight_layout(rect=[0, 0.01, 1, 0.96], h_pad=2.6, w_pad=2.0)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.1, dpi=300)
    plt.close(fig)
    print(f"[Grafik] Konfuzyon matrisleri kaydedildi -> {output_path}")


def run_training() -> None:
    """Main training, evaluation, and artifact generation routine."""
    print("=" * 70)
    print("[Egitim Hatti] Mental Saglik Riski Tahmini - Egitim ve Degerlendirme")
    print("=" * 70)

    # 1. Load / Prepare data
    raw_path = "data/raw/mental_health_dataset.csv"
    processed_path = "data/processed/mental_health_data_processed.csv"

    if not os.path.exists(raw_path) and os.path.exists("mental_health_dataset.csv"):
        raw_path = "mental_health_dataset.csv"

    if not os.path.exists(processed_path):
        print("On islenmis veri bulunamadi, on isleme calistiriliyor...")
        data = preprocess_data(raw_path=raw_path, output_path=processed_path)
    else:
        data = pd.read_csv(processed_path)

    # Save Correlation Heatmap from raw data if present
    if os.path.exists(raw_path):
        raw_df = pd.read_csv(raw_path)
        generate_correlation_matrix(raw_df)

    target_col = "mental_health_risk"
    X = data.drop(columns=[target_col])
    y = data[target_col]

    # Scaling numeric features
    numeric_cols = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
    scaler = MinMaxScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # Stratified Train-Test Split (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Veri Bolunmesi -> Egitim: {len(X_train)} ornek, Test: {len(X_test)} ornek.")

    # 2. Feature Selection
    print("\nOznitelik secimi yontemleri calistiriliyor (k=4)...")
    feature_sets = {}

    # Chi2
    chi2_sel = SelectKBest(score_func=chi2, k=4)
    X_tr_chi2 = chi2_sel.fit_transform(X_train, y_train)
    X_te_chi2 = chi2_sel.transform(X_test)
    chi2_features = list(X.columns[chi2_sel.get_support()])
    feature_sets["Chi2"] = {"X_train": X_tr_chi2, "X_test": X_te_chi2, "features": chi2_features}
    print(f"-> Chi2 ile secilen ozellikler: {chi2_features}")

    # ANOVA
    anova_sel = SelectKBest(score_func=f_classif, k=4)
    X_tr_anova = anova_sel.fit_transform(X_train, y_train)
    X_te_anova = anova_sel.transform(X_test)
    anova_features = list(X.columns[anova_sel.get_support()])
    feature_sets["ANOVA"] = {"X_train": X_tr_anova, "X_test": X_te_anova, "features": anova_features}
    print(f"-> ANOVA ile secilen ozellikler: {anova_features}")

    # RFE
    rfe_sel = RFE(estimator=DecisionTreeClassifier(random_state=42), n_features_to_select=4)
    X_tr_rfe = rfe_sel.fit_transform(X_train, y_train)
    X_te_rfe = rfe_sel.transform(X_test)
    rfe_features = list(X.columns[rfe_sel.support_])
    feature_sets["RFE"] = {"X_train": X_tr_rfe, "X_test": X_te_rfe, "features": rfe_features}
    print(f"-> RFE ile secilen ozellikler: {rfe_features}")

    # 3. Model Definitions
    def get_fresh_models():
        return {
            "MLP": MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=500, random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=15),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
        }

    benchmark_results = {"MLP": {}, "KNN": {}, "Decision Tree": {}, "Random Forest": {}}
    records = []

    print("\nModeller 12 farkli kombinasyon uzerinde egitiliyor...")
    for model_name in benchmark_results.keys():
        for method, f_data in feature_sets.items():
            models = get_fresh_models()
            model = models[model_name]
            model.fit(f_data["X_train"], y_train)

            y_pred = model.predict(f_data["X_test"])
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
            rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
            cm = confusion_matrix(y_test, y_pred)

            metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "confusion_matrix": cm}
            benchmark_results[model_name][method] = {
                "model": model,
                "metrics": metrics,
                "features": f_data["features"],
            }

            records.append({
                "Model": model_name,
                "Feature Selection": method,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1 Score": f1,
                "Features": ", ".join(f_data["features"]),
            })

    performance_df = pd.DataFrame(records)
    print("\n[Sonuc Tablosu]")
    print(performance_df[["Model", "Feature Selection", "Accuracy", "Precision", "Recall", "F1 Score"]].to_string(index=False))

    # Best Model Selection
    best_row = performance_df.sort_values(by=["Accuracy", "F1 Score"], ascending=[False, False]).iloc[0]
    best_model_name = best_row["Model"]
    best_method = best_row["Feature Selection"]
    best_entry = benchmark_results[best_model_name][best_method]

    # Save Self-Contained Best Model
    os.makedirs("models", exist_ok=True)
    best_model_artifact = {
        "model": best_entry["model"],
        "model_name": f"{best_model_name} ({best_method})",
        "base_model": best_model_name,
        "feature_selection": best_method,
        "accuracy": float(best_row["Accuracy"]),
        "precision": float(best_row["Precision"]),
        "recall": float(best_row["Recall"]),
        "f1": float(best_row["F1 Score"]),
        "feature_names": best_entry["features"],
        "scaler": scaler,
        "all_feature_columns": list(X.columns),
    }

    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(best_model_artifact, f)
    print(f"\n[En Iyi Model] '{best_model_artifact['model_name']}' models/best_model.pkl dosyasina kaydedildi.")
    print(f"Dogruluk (Accuracy): {best_model_artifact['accuracy']:.4f} | F1: {best_model_artifact['f1']:.4f}")
    print(f"Kullanilan Ozellikler: {best_model_artifact['feature_names']}")

    # 4. Generate Visualizations
    print("\nGrafikler uretiliyor...")
    generate_metrics_table_image(performance_df)
    generate_performance_comparison(benchmark_results)
    generate_confusion_matrices(benchmark_results)

    print("\n[Tamamlandi] Tum egitim, degerlendirme ve gorsellestirme adimlari basariyla sonuclandi!")


if __name__ == "__main__":
    run_training()
