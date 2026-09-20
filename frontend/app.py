import streamlit as st
import requests
import pandas as pd
import os

# FastAPI sunucunun çalıştığı temel adres
API_BASE_URL = "http://localhost:8000"

def fetch_datasets():
    """GET isteği ile backend'den mevcut veri setlerini çeker."""
    try:
        response = requests.get(f"{API_BASE_URL}/datasets")
        response.raise_for_status() # 200 OK dışındaki yanıtlarda hata fırlatır
        data = response.json()
        return data.get("datasets", [])
    except requests.exceptions.ConnectionError:
        st.error("🚨 Backend sunucusuna bağlanılamadı. Lütfen terminalde 'uvicorn main:app' komutunun çalıştığından emin ol.")
        return []
    except Exception as e:
        st.error(f"Veri setleri alınırken bir hata oluştu: {str(e)}")
        return []

def fetch_models():
    """GET isteği ile backend'den mevcut algoritmaları çeker."""
    try:
        response = requests.get(f"{API_BASE_URL}/analyze/models")
        response.raise_for_status()
        data = response.json()
        return data.get("models", [])
    except Exception as e:
        st.error(f"Modeller alınırken bir hata oluştu: {str(e)}")
        return []

def run_analysis(dataset_name, auto_route=True, algorithms=None):
    """POST isteği ile seçilen parametreleri backend'e gönderir ve sonuçları alır."""
    payload = {
        "dataset_name": dataset_name,
        "auto_route": auto_route,
        "algorithms": algorithms if algorithms else []
    }
    
    try:
        # Analiz uzun sürebileceği için kullanıcıya yükleniyor animasyonu gösteriyoruz
        with st.spinner('Motor çalışıyor, anomaliler tespit ediliyor... Lütfen bekleyin.'):
            response = requests.post(f"{API_BASE_URL}/analyze", json=payload)
            response.raise_for_status()
            return response.json()
    except requests.exceptions.HTTPError as http_err:
        st.error(f"API Hatası (HTTP {response.status_code}): {response.text}")
        return None
    except Exception as e:
        st.error(f"Analiz isteği sırasında beklenmeyen hata: {str(e)}")
        return None


# --- ARAYÜZ (UI) YAPILANDIRMASI ---
# Sayfanın geniş ve profesyonel görünmesi için temel ayarlar
st.set_page_config(page_title="Anomaly Detection Engine", page_icon="🔍", layout="wide")

st.title("🔍 Anomaly Detection Engine")
st.markdown("Farklı veri setleri üzerinde çoklu anomali algoritmalarını çalıştırın ve ortak (kesişim) anomalileri keşfedin.")

# --- KONTROL PANELİ (SIDEBAR) ---
st.sidebar.header("⚙️ Analiz Ayarları")

# 1. Veri Seti Seçimi
available_datasets = fetch_datasets()
if not available_datasets:
    st.sidebar.warning("Lokal 'data' klasöründe veri seti bulunamadı veya backend kapalı.")
    selected_dataset = None
else:
    selected_dataset = st.sidebar.selectbox("📂 Veri Seti Seçin", available_datasets)

st.sidebar.divider() # Görsel bir ayraç çizgisi çeker

# 2. Model Yönlendirme (Auto-route vs Manuel)
auto_route = st.sidebar.toggle("🤖 Otomatik Model Seçimi", value=True)

selected_algs = []
if not auto_route:
    st.sidebar.markdown("Kendi algoritmalarınızı seçin:")
    
    available_models = fetch_models()
    if available_models:
        selected_algs = st.sidebar.multiselect(
            "Algoritmalar", 
            options=available_models, 
            default=[]
        )
    else:
        st.sidebar.warning("API'den model listesi alınamadı!")
        selected_algs = []

st.sidebar.divider()

# 3. Tetikleyici Buton
start_button = st.sidebar.button("🚀 Analizi Başlat", use_container_width=True)

# --- ANA EKRAN (ANALİZ VE SONUÇLAR) ---
if start_button and selected_dataset:
    # API'ye istek atıp sonuçları alıyoruz
    results = run_analysis(selected_dataset, auto_route, algorithms=selected_algs)
    
    if results:
        # Sonuçları Streamlit'in geçici hafızasına (session_state) kaydediyoruz
        # Böylece arayüzde başka bir yere tıklanınca sonuçlar ekrandan silinmez
        st.session_state['results'] = results
        st.success("Analiz başarıyla tamamlandı!")

# Geçici hafızada sonuç varsa ekrana bas
if 'results' in st.session_state:
    results = st.session_state['results']
    
    st.divider() # Ekranı böl
    st.header("📊 Analiz Sonuçları")
    
    # --- 1. SKOR KARTLARI (METRİKLER) ---
    # Ekranı 3 sütuna bölüyoruz
    col1, col2, col3 = st.columns(3)
    
    col1.metric(label="Toplam İncelenen Satır", value=results.get("total_rows", 0))
    col2.metric(label="Kesin Anomali Sayısı (Kesişim)", value=len(results.get("common_anomalies", [])))
    col3.metric(label="Çalışma Süresi", value=f"{results.get('execution_time_sec', 0)} sn")
    
    # --- 2. MODEL BAZLI DETAYLAR ---
    st.subheader("🤖 Algoritma Performansları")
    
    model_results = results.get("model_results", [])
    if model_results:
        # Modelleri ekrana yan yana sığdırmak için kolonlara ayır (eğer çok model varsa row/col mantığı da kurulabilir, şimdilik basit tutuyoruz)
        cols = st.columns(len(model_results))
        for idx, model in enumerate(model_results):
            model_name = model.get("name", f"Model {idx+1}")
            anomaly_count = len(model.get("anomalies", []))
            cols[idx].info(f"**{model_name}** toplam **{anomaly_count}** satıra anomali dedi.")
    else:
        st.info("Herhangi bir model sonucu dönmedi.")

    # --- 3. SEKMELİ VERİ TABLOLARI (TABS) ---
    st.subheader("🔍 Detaylı Veri İncelemesi")
    
    common_indices = results.get("common_anomalies", [])
    model_results = results.get("model_results", [])
    
    dataset_name = results.get("dataset_name")
    file_path = os.path.join("data", dataset_name)
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path) # Tüm orijinal veriyi okuyoruz
        
        # Sekme başlıklarını dinamik olarak oluştur
        tab_names = ["🚨 Kesişim (Consensus)"]
        for model in model_results:
            tab_names.append(f"🔍 {model.get('name', 'Model')} Anomalileri")
        tab_names.append("📂 Tüm Veri Seti")
        
        # Streamlit Tabs (Sekmeler) oluşturuyoruz
        tabs = st.tabs(tab_names)
        
        # 1. Sekme: Kesişim (Ortak Anomaliler)
        with tabs[0]:
            if len(common_indices) > 0:
                st.error("Aşağıdaki tablo, çalışan algoritmaların ortaklaşa 'Anomali' olarak tespit ettiği riskli satırları göstermektedir.")
                st.dataframe(df.iloc[common_indices].copy(), use_container_width=True, key="df_consensus")
            else:
                st.success("Harika! Tüm modellerin ortaklaşa anomali dediği hiçbir satır bulunamadı.")
                
        # Ara Sekmeler: Her bir modelin buldukları
        for i, model in enumerate(model_results):
            with tabs[i + 1]:
                m_name = model.get('name', f'Model {i+1}')
                m_indices = model.get('anomalies', [])
                if len(m_indices) > 0:
                    st.info(f"{m_name} algoritmasının tespit ettiği tüm anomaliler:")
                    st.dataframe(df.iloc[m_indices].copy(), use_container_width=True, key=f"df_model_{i}")
                else:
                    st.info(f"{m_name} herhangi bir anomali tespit etmedi.")
                
        # Son Sekme: Orijinal Veri Seti
        with tabs[-1]:
            st.write("Veri setinin orijinal hali:")
            st.dataframe(df, use_container_width=True, key="df_all_data")
            
    else:
        st.warning("Orijinal veri seti okunamadı, tablolar gösterilemiyor.")