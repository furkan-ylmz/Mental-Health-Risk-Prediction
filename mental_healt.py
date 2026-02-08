import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from sklearn.feature_selection import RFE, SelectKBest, f_classif, chi2

data = pd.read_csv("mental_health_data_processed.csv")

target = "mental_health_risk"
X = data.drop(columns=[target])
y = data[target]

numeric_cols = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
scaler = MinMaxScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# CHI2
chi2_selector = SelectKBest(score_func=chi2, k=4)
X_train_chi2 = chi2_selector.fit_transform(X_train, y_train)
X_test_chi2 = chi2_selector.transform(X_test)
chi2_selected_features = X.columns[chi2_selector.get_support()]
print("Chi-Square ile seçilen özellikler:\n", chi2_selected_features)

# RFE
rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=4)
X_train_rfe = rfe.fit_transform(X_train, y_train)
X_test_rfe = rfe.transform(X_test)
selected_features = X.columns[rfe.support_]
print("RFE ile seçilen özellikler:\n", selected_features)

# ANOVA
anova_selector = SelectKBest(score_func=f_classif, k=4)
X_train_anova = anova_selector.fit_transform(X_train, y_train)
X_test_anova = anova_selector.transform(X_test)
anova_selected_features = X.columns[anova_selector.get_support()]
print("ANOVA ile seçilen özellikler:\n", anova_selected_features)

# Decision Tree
tree_model_chi2 = DecisionTreeClassifier()
tree_model_chi2.fit(X_train_chi2, y_train)
y_pred_chi2_tree = tree_model_chi2.predict(X_test_chi2)
accuracy_chi2_tree = accuracy_score(y_test, y_pred_chi2_tree)

tree_model_anova = DecisionTreeClassifier()
tree_model_anova.fit(X_train_anova, y_train)
y_pred_anova_tree = tree_model_anova.predict(X_test_anova)
accuracy_anova_tree = accuracy_score(y_test, y_pred_anova_tree)

tree_model_rfe = DecisionTreeClassifier()
tree_model_rfe.fit(X_train_rfe, y_train)
y_pred_rfe_tree = tree_model_rfe.predict(X_test_rfe)
accuracy_rfe_tree = accuracy_score(y_test, y_pred_rfe_tree)

print(f"Decision Tree (Chi2) Accuracy: {accuracy_chi2_tree}")
print(f"Decision Tree (ANOVA) Accuracy: {accuracy_anova_tree}")
print(f"Decision Tree (RFE) Accuracy: {accuracy_rfe_tree}")

# ANN
ann_model_chi2 = MLPClassifier(hidden_layer_sizes=(50,50,), max_iter=500)
ann_model_chi2.fit(X_train_chi2, y_train)
y_pred_chi2_ann = ann_model_chi2.predict(X_test_chi2)
accuracy_chi2_ann = accuracy_score(y_test, y_pred_chi2_ann)

ann_model_anova = MLPClassifier(hidden_layer_sizes=(50,50,), max_iter=500)
ann_model_anova.fit(X_train_anova, y_train)
y_pred_anova_ann = ann_model_anova.predict(X_test_anova)
accuracy_anova_ann = accuracy_score(y_test, y_pred_anova_ann)

ann_model_rfe = MLPClassifier(hidden_layer_sizes=(50,50,), max_iter=500)
ann_model_rfe.fit(X_train_rfe, y_train)
y_pred_rfe_ann = ann_model_rfe.predict(X_test_rfe)
accuracy_rfe_ann = accuracy_score(y_test, y_pred_rfe_ann)

print(f"MLP (Chi2) Accuracy: {accuracy_chi2_ann}")
print(f"MLP (ANOVA) Accuracy: {accuracy_anova_ann}")
print(f"MLP (RFE) Accuracy: {accuracy_rfe_ann}")

# KNN
knn_model_chi2 = KNeighborsClassifier(n_neighbors=15)
knn_model_chi2.fit(X_train_chi2, y_train)
y_pred_chi2_knn = knn_model_chi2.predict(X_test_chi2)
accuracy_chi2_knn = accuracy_score(y_test, y_pred_chi2_knn)

knn_model_anova = KNeighborsClassifier(n_neighbors=15)
knn_model_anova.fit(X_train_anova, y_train)
y_pred_anova_knn = knn_model_anova.predict(X_test_anova)
accuracy_anova_knn = accuracy_score(y_test, y_pred_anova_knn)

knn_model_rfe = KNeighborsClassifier(n_neighbors=15)
knn_model_rfe.fit(X_train_rfe, y_train)
y_pred_rfe_knn = knn_model_rfe.predict(X_test_rfe)
accuracy_rfe_knn = accuracy_score(y_test, y_pred_rfe_knn)

print(f"KNN (Chi2) Accuracy: {accuracy_chi2_knn}")
print(f"KNN (ANOVA) Accuracy: {accuracy_anova_knn}")
print(f"KNN (RFE) Accuracy: {accuracy_rfe_knn}")

# Random Forest
rf_model_chi2 = RandomForestClassifier(n_estimators=300)
rf_model_chi2.fit(X_train_chi2, y_train)
y_pred_chi2_rf = rf_model_chi2.predict(X_test_chi2)
accuracy_chi2_rf = accuracy_score(y_test, y_pred_chi2_rf)

rf_model_anova = RandomForestClassifier(n_estimators=300)
rf_model_anova.fit(X_train_anova, y_train)
y_pred_anova_rf = rf_model_anova.predict(X_test_anova)
accuracy_anova_rf = accuracy_score(y_test, y_pred_anova_rf)

rf_model_rfe = RandomForestClassifier(n_estimators=300)
rf_model_rfe.fit(X_train_rfe, y_train)
y_pred_rfe_rf = rf_model_rfe.predict(X_test_rfe)
accuracy_rfe_rf = accuracy_score(y_test, y_pred_rfe_rf)

print(f"Random Forest (Chi2) Accuracy: {accuracy_chi2_rf}")
print(f"Random Forest (ANOVA) Accuracy: {accuracy_anova_rf}")
print(f"Random Forest (RFE) Accuracy: {accuracy_rfe_rf}")

# Model performance function
def model_performance(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    return accuracy, precision, recall, f1

accuracy_chi2_tree, precision_chi2_tree, recall_chi2_tree, f1_chi2_tree = model_performance(tree_model_chi2, X_test_chi2, y_test)
accuracy_anova_tree, precision_anova_tree, recall_anova_tree, f1_anova_tree = model_performance(tree_model_anova, X_test_anova, y_test)
accuracy_rfe_tree, precision_rfe_tree, recall_rfe_tree, f1_rfe_tree = model_performance(tree_model_rfe, X_test_rfe, y_test)

accuracy_chi2_ann, precision_chi2_ann, recall_chi2_ann, f1_chi2_ann = model_performance(ann_model_chi2, X_test_chi2, y_test)
accuracy_anova_ann, precision_anova_ann, recall_anova_ann, f1_anova_ann = model_performance(ann_model_anova, X_test_anova, y_test)
accuracy_rfe_ann, precision_rfe_ann, recall_rfe_ann, f1_rfe_ann = model_performance(ann_model_rfe, X_test_rfe, y_test)

accuracy_chi2_knn, precision_chi2_knn, recall_chi2_knn, f1_chi2_knn = model_performance(knn_model_chi2, X_test_chi2, y_test)
accuracy_anova_knn, precision_anova_knn, recall_anova_knn, f1_anova_knn = model_performance(knn_model_anova, X_test_anova, y_test)
accuracy_rfe_knn, precision_rfe_knn, recall_rfe_knn, f1_rfe_knn = model_performance(knn_model_rfe, X_test_rfe, y_test)

accuracy_chi2_rf, precision_chi2_rf, recall_chi2_rf, f1_chi2_rf = model_performance(rf_model_chi2, X_test_chi2, y_test)
accuracy_anova_rf, precision_anova_rf, recall_anova_rf, f1_anova_rf = model_performance(rf_model_anova, X_test_anova, y_test)
accuracy_rfe_rf, precision_rfe_rf, recall_rfe_rf, f1_rfe_rf = model_performance(rf_model_rfe, X_test_rfe, y_test)

# Create confusion matrices
cm_chi2_tree = confusion_matrix(y_test, y_pred_chi2_tree)
cm_anova_tree = confusion_matrix(y_test, y_pred_anova_tree)
cm_rfe_tree = confusion_matrix(y_test, y_pred_rfe_tree)

cm_chi2_ann = confusion_matrix(y_test, y_pred_chi2_ann)
cm_anova_ann = confusion_matrix(y_test, y_pred_anova_ann)
cm_rfe_ann = confusion_matrix(y_test, y_pred_rfe_ann)

cm_chi2_knn = confusion_matrix(y_test, y_pred_chi2_knn)
cm_anova_knn = confusion_matrix(y_test, y_pred_anova_knn)
cm_rfe_knn = confusion_matrix(y_test, y_pred_rfe_knn)

cm_chi2_rf = confusion_matrix(y_test, y_pred_chi2_rf)
cm_anova_rf = confusion_matrix(y_test, y_pred_anova_rf)
cm_rfe_rf = confusion_matrix(y_test, y_pred_rfe_rf)

def get_metrics_for_all_methods():
    metrics = {
        "accuracy": {
            "Chi2": [accuracy_chi2_ann, accuracy_chi2_knn, accuracy_chi2_tree, accuracy_chi2_rf],
            "ANOVA": [accuracy_anova_ann, accuracy_anova_knn, accuracy_anova_tree, accuracy_anova_rf],
            "RFE": [accuracy_rfe_ann, accuracy_rfe_knn, accuracy_rfe_tree, accuracy_rfe_rf],
        },
        "precision": {
            "Chi2": [precision_chi2_ann, precision_chi2_knn, precision_chi2_tree, precision_chi2_rf],
            "ANOVA": [precision_anova_ann, precision_anova_knn, precision_anova_tree, precision_anova_rf],
            "RFE": [precision_rfe_ann, precision_rfe_knn, precision_rfe_tree, precision_rfe_rf],
        },
        "recall": {
            "Chi2": [recall_chi2_ann, recall_chi2_knn, recall_chi2_tree, recall_chi2_rf],
            "ANOVA": [recall_anova_ann, recall_anova_knn, recall_anova_tree, recall_anova_rf],
            "RFE": [recall_rfe_ann, recall_rfe_knn, recall_rfe_tree, recall_rfe_rf],
        },
        "f1": {
            "Chi2": [f1_chi2_ann, f1_chi2_knn, f1_chi2_tree, f1_chi2_rf],
            "ANOVA": [f1_anova_ann, f1_anova_knn, f1_anova_tree, f1_anova_rf],
            "RFE": [f1_rfe_ann, f1_rfe_knn, f1_rfe_tree, f1_rfe_rf],
        }
    }
    return metrics

metrics = get_metrics_for_all_methods()

model_names = ['MLP', 'KNN', 'Decision Tree', 'Random Forest']

# Then print the summary table
performance_df = pd.DataFrame({
    'Model': model_names,
    'Chi2 Accuracy': metrics["accuracy"]["Chi2"],
    'ANOVA Accuracy': metrics["accuracy"]["ANOVA"],
    'RFE Accuracy': metrics["accuracy"]["RFE"],
    'Chi2 Precision': metrics["precision"]["Chi2"],
    'ANOVA Precision': metrics["precision"]["ANOVA"],
    'RFE Precision': metrics["precision"]["RFE"],
    'Chi2 Recall': metrics["recall"]["Chi2"],
    'ANOVA Recall': metrics["recall"]["ANOVA"],
    'RFE Recall': metrics["recall"]["RFE"],
    'Chi2 F1 Score': metrics["f1"]["Chi2"],
    'ANOVA F1 Score': metrics["f1"]["ANOVA"],
    'RFE F1 Score': metrics["f1"]["RFE"]
})

print("\nModel Performans Değerlendirmesi (Tablo):\n")
print(performance_df)

model_variable_mapping = {
    "MLP": "ann_model",
    "KNN": "knn_model",
    "Decision Tree": "tree_model",
    "Random Forest": "rf_model"
}

performance_df['Best Accuracy'] = performance_df[['Chi2 Accuracy', 'ANOVA Accuracy', 'RFE Accuracy']].max(axis=1)
best_model_index = performance_df['Best Accuracy'].idxmax()
best_model_name = performance_df.loc[best_model_index, 'Model']
best_accuracy = performance_df.loc[best_model_index, 'Best Accuracy']

if best_accuracy == performance_df.loc[best_model_index, 'Chi2 Accuracy']:
    transformation = "Chi2"
elif best_accuracy == performance_df.loc[best_model_index, 'ANOVA Accuracy']:
    transformation = "ANOVA"
else:
    transformation = "RFE"

model_variable_name = f"{model_variable_mapping[best_model_name]}_{transformation.lower()}"
best_model = eval(model_variable_name)

# Determine used feature names
if transformation == "Chi2":
    feature_names = list(chi2_selected_features)
elif transformation == "ANOVA":
    feature_names = list(anova_selected_features)
elif transformation == "RFE":
    feature_names = list(selected_features)
else:
    feature_names = list(X.columns)

if not os.path.exists("models"):
    os.makedirs("models")

# Include feature names when saving the model
with open("models/best_model.pkl", "wb") as file:
    pickle.dump({
        "model": best_model,
        "model_name": f"{best_model_name} ({transformation})",
        "accuracy": best_accuracy,
        "feature_names": feature_names
    }, file)

print(f"En iyi model '{best_model_name} ({transformation})' başarıyla kaydedildi (Doğruluk: {best_accuracy:.2f}).")

metrics_labels = ["Accuracy", "Precision", "Recall", "F1 Score"]

# Decision Tree
tree_metrics = [
    [accuracy_chi2_tree, accuracy_anova_tree, accuracy_rfe_tree],
    [precision_chi2_tree, precision_anova_tree, precision_rfe_tree],
    [recall_chi2_tree, recall_anova_tree, recall_rfe_tree],
    [f1_chi2_tree, f1_anova_tree, f1_rfe_tree]
]
plt.figure(figsize=(8, 5))
df_tree = pd.DataFrame(tree_metrics, index=metrics_labels, columns=["Chi2", "ANOVA", "RFE"])
df_tree.T.plot(kind="bar", rot=0)
plt.title("Decision Tree - Performans Sonuçları")
plt.ylabel("Skor")
plt.ylim(0, 1)
plt.legend(title="Metrik")
plt.tight_layout()
plt.show()

# ANN
ann_metrics = [
    [accuracy_chi2_ann, accuracy_anova_ann, accuracy_rfe_ann],
    [precision_chi2_ann, precision_anova_ann, precision_rfe_ann],
    [recall_chi2_ann, recall_anova_ann, recall_rfe_ann],
    [f1_chi2_ann, f1_anova_ann, f1_rfe_ann]
]
plt.figure(figsize=(8, 5))
df_ann = pd.DataFrame(ann_metrics, index=metrics_labels, columns=["Chi2", "ANOVA", "RFE"])
df_ann.T.plot(kind="bar", rot=0)
plt.title("MLP - Performans Sonuçları")
plt.ylabel("Skor")
plt.ylim(0, 1)
plt.legend(title="Metrik")
plt.tight_layout()
plt.show()

# KNN
knn_metrics = [
    [accuracy_chi2_knn, accuracy_anova_knn, accuracy_rfe_knn],
    [precision_chi2_knn, precision_anova_knn, precision_rfe_knn],
    [recall_chi2_knn, recall_anova_knn, recall_rfe_knn],
    [f1_chi2_knn, f1_anova_knn, f1_rfe_knn]
]
plt.figure(figsize=(8, 5))
df_knn = pd.DataFrame(knn_metrics, index=metrics_labels, columns=["Chi2", "ANOVA", "RFE"])
df_knn.T.plot(kind="bar", rot=0)
plt.title("KNN - Performans Sonuçları")
plt.ylabel("Skor")
plt.ylim(0, 1)
plt.legend(title="Metrik")
plt.tight_layout()
plt.show()

# Random Forest
rf_metrics = [
    [accuracy_chi2_rf, accuracy_anova_rf, accuracy_rfe_rf],
    [precision_chi2_rf, precision_anova_rf, precision_rfe_rf],
    [recall_chi2_rf, recall_anova_rf, recall_rfe_rf],
    [f1_chi2_rf, f1_anova_rf, f1_rfe_rf]
]
plt.figure(figsize=(8, 5))
df_rf = pd.DataFrame(rf_metrics, index=metrics_labels, columns=["Chi2", "ANOVA", "RFE"])
df_rf.T.plot(kind="bar", rot=0)
plt.title("Random Forest - Performans Sonuçları")
plt.ylabel("Skor")
plt.ylim(0, 1)
plt.legend(title="Metrik")
plt.tight_layout()
plt.show()

# Show 3 confusion matrices for Decision Tree in a single row
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, cm, title in zip(
    axes,
    [cm_chi2_tree, cm_anova_tree, cm_rfe_tree],
    ["Chi2", "ANOVA", "RFE"]
):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"Decision Tree ({title})")
plt.tight_layout()
plt.show()

# Show 3 confusion matrices for MLP in a single row
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, cm, title in zip(
    axes,
    [cm_chi2_ann, cm_anova_ann, cm_rfe_ann],
    ["Chi2", "ANOVA", "RFE"]
):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"MLP ({title})")
plt.tight_layout()
plt.show()

# Show 3 confusion matrices for KNN in a single row
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, cm, title in zip(
    axes,
    [cm_chi2_knn, cm_anova_knn, cm_rfe_knn],
    ["Chi2", "ANOVA", "RFE"]
):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"KNN ({title})")
plt.tight_layout()
plt.show()

# Show 3 confusion matrices for Random Forest in a single row
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, cm, title in zip(
    axes,
    [cm_chi2_rf, cm_anova_rf, cm_rfe_rf],
    ["Chi2", "ANOVA", "RFE"]
):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"Random Forest ({title})")
plt.tight_layout()
plt.show()