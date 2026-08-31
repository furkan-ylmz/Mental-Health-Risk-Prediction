"""
Interactive Streamlit Web Dashboard for Mental Health Risk Prediction
Provides real-time risk assessment, data exploration, and model performance visualization.
"""

import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler


st.set_page_config(
    page_title="Mental Health Risk Prediction | Mental Sağlık Riski Tahmini",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .risk-card-high {
        background-color: #fee2e2;
        border-radius: 12px;
        padding: 20px;
        border: 2px solid #ef4444;
        color: #991b1b;
    }
    .risk-card-medium {
        background-color: #fef3c7;
        border-radius: 12px;
        padding: 20px;
        border: 2px solid #f59e0b;
        color: #92400e;
    }
    .risk-card-low {
        background-color: #dcfce7;
        border-radius: 12px;
        padding: 20px;
        border: 2px solid #22c55e;
        color: #166534;
    }
</style>
""", unsafe_allow_html=True)


CATEGORY_MAPPINGS = {
    "gender": {"Female": 0, "Male": 1, "Non-binary": 2, "Prefer not to say": 3},
    "employment_status": {"Employed": 0, "Self-employed": 1, "Student": 2, "Unemployed": 3},
    "work_environment": {"Hybrid": 0, "On-site": 1, "Remote": 2},
    "mental_health_history": {"No": 0, "Yes": 1},
    "seeks_treatment": {"No": 0, "Yes": 1},
}

RISK_MAP = {
    0: {"en": "High", "tr": "Yüksek", "class": "risk-card-high", "title": "🔴 Yüksek Risk / High Risk",
        "desc": "Model, girilen parametreler doğrultusunda yüksek mental sağlık riski öngörmüştür.",
        "advice": "Profesyonel bir psikolojik danışman veya terapist desteği alınması kuvvetle tavsiye edilir."},
    1: {"en": "Low", "tr": "Düşük", "class": "risk-card-low", "title": "🟢 Düşük Risk / Low Risk",
        "desc": "Model, düşük seviyede mental sağlık riski öngörmüştür.",
        "advice": "Mevcut dengeli yaşam alışkanlıklarının ve sosyal destek mekanizmalarının sürdürülmesi önerilir."},
    2: {"en": "Medium", "tr": "Orta", "class": "risk-card-medium", "title": "🟡 Orta Risk / Medium Risk",
        "desc": "Model, orta seviyede mental sağlık riski tespit etmiştir.",
        "advice": "Stres yönetimi teknikleri, düzenli uyku ve fiziksel aktivite seviyesinin artırılması faydalı olacaktır."},
}


@st.cache_resource
def load_best_model():
    """Load serialized model artifact."""
    model_path = "models/best_model.pkl"
    if not os.path.exists(model_path):
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_processed_data():
    """Load processed dataset."""
    paths = ["data/processed/mental_health_data_processed.csv", "mental_health_data_processed.csv"]
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    return None


@st.cache_data
def load_raw_data():
    """Load raw survey data."""
    paths = ["data/raw/mental_health_dataset.csv", "mental_health_dataset.csv"]
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    return None


def predict_risk(model_artifact, input_dict, processed_df):
    """
    Transforms user input dictionary and executes prediction using the best model.
    """
    model = model_artifact["model"]
    feature_names = model_artifact["feature_names"]

    # Encode categorical fields
    encoded_dict = input_dict.copy()
    for cat_col, mapping in CATEGORY_MAPPINGS.items():
        if cat_col in encoded_dict:
            val = encoded_dict[cat_col]
            encoded_dict[cat_col] = mapping.get(str(val), val)

    input_df = pd.DataFrame([encoded_dict])

    # Fit scaler on processed numerical features
    if processed_df is not None:
        target_col = "mental_health_risk"
        X_ref = processed_df.drop(columns=[target_col]) if target_col in processed_df.columns else processed_df.copy()
        numeric_cols = X_ref.select_dtypes(include=["float64", "int64"]).columns.tolist()

        scaler = MinMaxScaler()
        scaler.fit(X_ref[numeric_cols])

        # Scale input
        for col in numeric_cols:
            if col not in input_df.columns:
                input_df[col] = X_ref[col].mean()

        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    # Select only required features
    X_input = input_df[feature_names].values
    pred_code = int(model.predict(X_input)[0])

    probs = None
    if hasattr(model, "predict_proba"):
        try:
            raw_proba = model.predict_proba(X_input)[0]
            probs = {
                RISK_MAP[idx]["en"]: float(raw_proba[idx])
                for idx in range(len(raw_proba))
            }
        except Exception:
            probs = None

    return pred_code, probs


def main():
    st.markdown('<div class="main-title">🧠 Mental Health Risk Prediction Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Yapay Zeka Destekli Mental Sağlık Risk Seviyesi Değerlendirme ve Analiz Sistemi</div>', unsafe_allow_html=True)

    model_artifact = load_best_model()
    processed_df = load_processed_data()

    if model_artifact is None:
        st.warning("Eğitilmiş model bulunamadı. Lütfen önce terminalden `python core/train.py` komutunu çalıştırın.")
        return

    # Sidebar Info
    st.sidebar.header("📌 Model Bilgileri / Model Info")
    st.sidebar.info(f"**Aktif Model:** {model_artifact['model_name']}\n\n**Test Doğruluğu (Accuracy):** %{model_artifact['accuracy'] * 100:.2f}\n\n**F1 Skoru:** {model_artifact['f1']:.4f}")
    st.sidebar.markdown(f"**Kullanılan Özellikler ({len(model_artifact['feature_names'])}):**\n" + "\n".join([f"- `{f}`" for f in model_artifact['feature_names']]))

    # Tabs
    tab1, tab2, tab3 = st.tabs(["🔮 Canlı Risk Tahmini / Prediction", "📊 Model Başarım Analizi / Evaluation", "📁 Veri Seti İnceleme / Dataset Explorer"])

    with tab1:
        st.subheader("Bireysel Parametre Girişi / Individual Input Parameters")
        st.write("Aşağıdaki alanları doldurarak bireyin tahmini mental sağlık riski seviyesini hesaplayabilirsiniz.")

        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Yaş / Age", min_value=18, max_value=80, value=32, step=1)
            gender = st.selectbox("Cinsiyet / Gender", ["Male", "Female", "Non-binary", "Prefer not to say"], index=0)
            employment_status = st.selectbox("Çalışma Durumu / Employment Status", ["Employed", "Self-employed", "Student", "Unemployed"], index=0)
            work_environment = st.selectbox("Çalışma Ortamı / Work Environment", ["On-site", "Hybrid", "Remote"], index=0)

        with col2:
            mental_health_history = st.selectbox("Geçmişte Mental Sağlık Tanısı / Mental Health History", ["No", "Yes"], index=0)
            seeks_treatment = st.selectbox("Tedavi / Destek Arayışı / Seeks Treatment", ["No", "Yes"], index=0)
            stress_level = st.slider("Stres Seviyesi / Stress Level (1-10)", min_value=1, max_value=10, value=6)
            sleep_hours = st.slider("Günlük Ortalama Uyku / Sleep Hours", min_value=3.0, max_value=12.0, value=7.0, step=0.5)

        with col3:
            physical_activity_days = st.slider("Haftalık Fiziksel Aktivite Günü / Activity Days", min_value=0, max_value=7, value=3)
            anxiety_score = st.slider("Anksiyete Skoru / Anxiety Score (0-30)", min_value=0, max_value=30, value=12)
            social_support_score = st.slider("Sosyal Destek Skoru / Social Support (0-100)", min_value=0, max_value=100, value=60)
            productivity_score = st.slider("Üretkenlik Skoru / Productivity Score (0-100)", min_value=0, max_value=100, value=65)

        input_data = {
            "age": age,
            "gender": gender,
            "employment_status": employment_status,
            "work_environment": work_environment,
            "mental_health_history": mental_health_history,
            "seeks_treatment": seeks_treatment,
            "stress_level": stress_level,
            "sleep_hours": sleep_hours,
            "physical_activity_days": physical_activity_days,
            "anxiety_score": anxiety_score,
            "social_support_score": social_support_score,
            "productivity_score": productivity_score,
        }

        st.markdown("---")
        if st.button("🚀 Risk Seviyesini Tahmin Et / Predict Risk", type="primary", use_container_width=True):
            pred_code, probs = predict_risk(model_artifact, input_data, processed_df)
            risk_info = RISK_MAP.get(pred_code, RISK_MAP[0])

            st.subheader("Tahmin Sonucu / Prediction Result")
            res_col1, res_col2 = st.columns([1, 1])

            with res_col1:
                st.markdown(f"""
                <div class="{risk_info['class']}">
                    <h2>{risk_info['title']}</h2>
                    <p><strong>Değerlendirme:</strong> {risk_info['desc']}</p>
                    <p><strong>Öneri:</strong> {risk_info['advice']}</p>
                </div>
                """, unsafe_allow_html=True)

            with res_col2:
                st.write("**Sınıf Olasılık Dağılımı / Class Probabilities:**")
                if probs:
                    prob_df = pd.DataFrame(
                        list(probs.items()),
                        columns=["Risk Seviyesi", "Olasılık"]
                    )
                    st.bar_chart(prob_df.set_index("Risk Seviyesi"))
                else:
                    st.info(f"Tahmin Edilen Sınıf: **{risk_info['tr']}** ({risk_info['en']})")

    with tab2:
        st.subheader("Model Performans Göstergeleri ve Grafikler / Evaluation Artifacts")
        st.write("Eğitim aşamasında 4 farklı makine öğrenmesi modeli ve 3 farklı özellik seçimi yöntemi (Chi2, ANOVA, RFE) benchmark edilmiştir.")

        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            if os.path.exists("results/model_evaluation_metrics.png"):
                st.image("results/model_evaluation_metrics.png", caption="Model Karşılaştırma Tablosu / Metric Summary Table")
            if os.path.exists("results/correlation_matrix.png"):
                st.image("results/correlation_matrix.png", caption="Özellik Korelasyon Matrisi / Correlation Heatmap")

        with metric_col2:
            if os.path.exists("results/performance_comparison.png"):
                st.image("results/performance_comparison.png", caption="Özellik Seçimi ve Model Başarım Grafikleri / Performance Comparison")
            if os.path.exists("results/confusion_matrices.png"):
                st.image("results/confusion_matrices.png", caption="Test Konfüzyon Matrisleri / Confusion Matrices")

    with tab3:
        st.subheader("Veri Seti Önizleme / Dataset Preview")
        raw_df = load_raw_data()
        if raw_df is not None:
            st.write(f"Toplam Satır Sayısı: **{len(raw_df)}** | Sütun Sayısı: **{len(raw_df.columns)}**")
            st.dataframe(raw_df.head(100), use_container_width=True)

            st.subheader("Hedef Değişken Dağılımı / Target Class Distribution")
            st.bar_chart(raw_df["mental_health_risk"].value_counts())


if __name__ == "__main__":
    main()
