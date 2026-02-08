import streamlit as st
import pandas as pd
import pickle

# Özelliklerin Türkçe karşılıkları
feature_tr = {
    "age": "Yaş",
    "gender": "Cinsiyet",
    "employment_status": "Çalışma Durumu",
    "work_environment": "Çalışma Ortamı",
    "mental_health_history": "Mental Sağlık Geçmişi",
    "seeks_treatment": "Tedavi Arayışı",
    "stress_level": "Stres Seviyesi",
    "sleep_hours": "Uyku Saatleri",
    "physical_activity_days": "Fiziksel Aktivite Günleri",
    "social_support_score": "Sosyal Destek Skoru",
    "proactive_score": "Proaktif Destek Skoru",
    "anxiety_score": "Anksiyete Skoru",
    "productivity_score": "Üretkenlik Skoru",
}

# Tahmin edilen riskin string karşılıkları
risk_map = {0: "Yüksek", 1: "Düşük", 2: "Orta"}

@st.cache_resource
def load_best_model():
    try:
        with open("models/best_model.pkl", "rb") as file:
            data = pickle.load(file)
        return data["model"], data["model_name"], data["accuracy"], data.get("feature_names", None)
    except FileNotFoundError:
        st.error("En iyi model dosyası bulunamadı! Lütfen modeli kaydedin.")
        return None, None, None, None

@st.cache_data
def load_final_data():
    return pd.read_csv("mental_health_data_processed.csv")

def initialize_input_data(data, columns):
    if "input_data" not in st.session_state:
        st.session_state.input_data = {col: data[col].mode()[0] if data[col].dtype == 'O' else int(data[col].mean()) for col in columns}

def get_user_inputs(columns, data):
    for col in columns:
        label = feature_tr.get(col, col)
        if data[col].dtype == 'O':
            options = sorted(data[col].unique())
            val = st.session_state.input_data.get(col, options[0])
            st.session_state.input_data[col] = st.sidebar.selectbox(label, options=options, index=options.index(val))
        else:
            min_val = int(data[col].min())
            max_val = int(data[col].max())
            val = int(st.session_state.input_data.get(col, min_val))
            step = 1
            fmt = "%d"
            st.session_state.input_data[col] = st.sidebar.number_input(
                label,
                min_value=min_val,
                max_value=max_val,
                value=val,
                step=step,
                format=fmt
            )

def make_prediction(model, input_data, norm_data, used_features):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(norm_data[used_features])
    input_df = pd.DataFrame([input_data])
    input_norm = scaler.transform(input_df)
    try:
        prediction = model.predict(input_norm)
        return prediction[0]
    except Exception as e:
        st.error(f"Tahmin sırasında bir hata oluştu: {e}")
        return None

def main():
    st.title("Mental Sağlık Riski Tahmini")

    model, model_name, model_accuracy, feature_names = load_best_model()
    if model:
        st.write(f"Bu uygulama, '{model_name}' modeli ile tahmin yapar.")
        st.info(f"Model doğruluk oranı: **{model_accuracy:.2f}**")
    else:
        st.stop()

    data = load_final_data()

    st.write("Veri setinden örnekler:")
    st.dataframe(data.head(50))

    # Sadece modelin beklediği özellikleri kullan
    if feature_names is not None:
        used_features = [col for col in feature_names if col in data.columns]
    else:
        used_features = data.select_dtypes(include=["float64", "int64"]).columns.tolist()

    X = data[used_features]
    initialize_input_data(data, X.columns)
    st.sidebar.header("Tahmin için Girdi Değerleri")
    st.sidebar.write("Lütfen tahmin için gerekli değerleri girin:")
    get_user_inputs(X.columns, data)

    if st.sidebar.button("Tahmin Yap"):
        st.write("Kullanıcı girdileri:")
        # Türkçe başlıklarla göster
        user_input_df = pd.DataFrame([st.session_state.input_data])
        user_input_df.columns = [feature_tr.get(col, col) for col in user_input_df.columns]
        st.write(user_input_df)
        prediction = make_prediction(model, {k: st.session_state.input_data[k] for k in X.columns}, data, X.columns)
        if prediction is not None:
            risk_str = risk_map.get(prediction, str(prediction))
            st.success(f"Tahmin edilen mental sağlık riski: **{risk_str}**")

if __name__ == "__main__":
    main()