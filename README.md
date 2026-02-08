# Mental Sağlık Riski Tahmini - Mental Health Risk Prediction

Bu proje, bireylerin çeşitli kişisel ve işle ilgili faktörlerine dayanarak mental sağlık riski seviyelerini (Düşük, Orta, Yüksek) tahmin etmeyi amaçlayan bir makine öğrenmesi uygulamasıdır. Veri ön işleme, özellik seçimi, model eğitimi ve kullanıcı dostu bir Streamlit arayüzü içerir.

## Özellikler

- **Veri Ön İşleme:** Ham veri setinin temizlenmesi, aykırı değerlerin tespiti ve kategorik verilerin sayısal hale getirilmesi (`Label Encoding`).
- **Özellik Seçimi:** Model performansını artırmak için Chi-Square, RFE (Recursive Feature Elimination) ve ANOVA gibi yöntemlerle en önemli özelliklerin belirlenmesi.
- **Çoklu Model Karşılaştırması:** Decision Tree, MLP (Yapay Sinir Ağları), KNN ve Random Forest algoritmalarının eğitilmesi ve karşılaştırılması.
- **İnteraktif Arayüz:** Streamlit ile geliştirilmiş, kullanıcıların kendi verilerini girerek anlık risk tahmini alabileceği web arayüzü.

## Dosya Yapısı

- `processed.py`: Ham veri setini (`mental_health_dataset.csv`) okur, temizler, şifreler ve işlenmiş veriyi (`mental_health_data_processed.csv`) kaydeder.
- `mental_risk.py`: İşlenmiş veriyi kullanarak makine öğrenmesi modellerini eğitir, performanslarını karşılaştırır ve en iyi modeli `models/` klasörüne kaydeder.
- `streamlit.py`: Kullanıcı arayüzünü oluşturur. Eğitilmiş modeli yükler ve kullanıcıdan alınan girdilere göre tahmin yapar.
- `models/`: Eğitilen modellerin `pickle` formatında saklandığı klasör.
- `mental_health_dataset.csv`: Ham veri seti.

## Kurulum

Projeyi yerel ortamınızda çalıştırmak için aşağıdaki adımları izleyin:

1. **Repoyu klonlayın:**
   ```bash
   git clone https://github.com/furkan-ylmz/Mental-Health-Risk-Prediction.git
   cd Mental-Health-Risk-Prediction
   ```

2. **Gerekli kütüphaneleri yükleyin:**
   Python 3.8+ önerilir. Bağımlılıkları yüklemek için:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn streamlit scipy
   ```

## Kullanım

Uygulamayı sıfırdan çalıştırmak için şu adımları takip edebilirsiniz:

1. **Veri Ön İşleme:**
   Veriyi temizlemek ve hazırlamak için çalıştırın:
   ```bash
   python processed.py
   ```

2. **Model Eğitimi:**
   Modelleri eğitmek ve en iyi modeli kaydetmek için çalıştırın:
   ```bash
   python mental_risk.py
   ```

3. **Uygulamayı Başlatma:**
   Streamlit arayüzünü başlatmak için:
   ```bash
   streamlit run streamlit.py
   ```

Tarayıcınızda otomatik olarak açılan sayfada (genellikle `http://localhost:8501`) değerleri girerek risk tahmini yapabilirsiniz.

## Kullanılan Teknolojiler

- **Python**
- **Streamlit** (Web Arayüzü)
- **Scikit-learn** (Makine Öğrenmesi)
- **Pandas & NumPy** (Veri İşleme)
- **Matplotlib & Seaborn** (Veri Görselleştirme)