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
    consensus_levels = results.get("consensus_levels", {})
    
    if consensus_levels:
        # Toplam Satır, [Dinamik Kesişim Seviyeleri...], Süre
        num_metrics = 2 + len(consensus_levels)
        cols = st.columns(num_metrics)
        
        cols[0].metric(label="Toplam İncelenen Satır", value=results.get("total_rows", 0))
        
        col_idx = 1
        for level_name, indices in consensus_levels.items():
            cols[col_idx].metric(label=level_name, value=len(indices))
            col_idx += 1
            
        cols[-1].metric(label="Çalışma Süresi", value=f"{results.get('execution_time_sec', 0)} sn")
    else:
        # Geriye dönük uyumluluk (Eski versiyon)
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
        # CSV ve JSON dosyalarını backend ile aynı şekilde oku
        if file_path.endswith(".json"):
            try:
                from src.engine.graph_feature_engineering import GraphFeatureEngineer
                graph_engineer = GraphFeatureEngineer(file_path)
                df = graph_engineer.transform()
            except Exception:
                import json
                with open(file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                df = pd.json_normalize(json_data if isinstance(json_data, list) else json_data)
        else:
            df = pd.read_csv(file_path)
            
        # Streamlit (Glide Data Grid) iç içe JSON (liste/sözlük) yapılarını "[object Object]" olarak
        # gösterdiği için bunları okunabilir string formatına dönüştürüyoruz.
        for col in df.columns:
            if df[col].apply(lambda x: isinstance(x, (dict, list))).any():
                df[col] = df[col].astype(str)
        
        consensus_levels = results.get("consensus_levels", {})
        
        # Sekme başlıklarını dinamik olarak oluştur
        tab_names = []
        if consensus_levels:
            for level in consensus_levels.keys():
                tab_names.append(f"🚨 {level}")
        else:
            tab_names.append("🚨 Kesişim (Consensus)")
            
        for model in model_results:
            tab_names.append(f"🔍 {model.get('name', 'Model')} Anomalileri")
        tab_names.append("📂 Tüm Veri Seti")
        
        # Streamlit Tabs (Sekmeler) oluşturuyoruz
        tabs = st.tabs(tab_names)
        
        tab_idx = 0
        
        # 1. Consensus Sekmeleri
        if consensus_levels:
            for level_name, indices in consensus_levels.items():
                with tabs[tab_idx]:
                    if len(indices) > 0:
                        st.error(f"Aşağıdaki tablo, algoritmaların '{level_name}' koşulunu sağlayarak 'Anomali' olarak tespit ettiği riskli satırları göstermektedir.")
                        st.dataframe(df.iloc[indices].copy(), use_container_width=True, key=f"df_consensus_{tab_idx}")
                    else:
                        st.success("Bu seviyede ortak anomali bulunamadı.")
                tab_idx += 1
        else:
            with tabs[0]:
                if len(common_indices) > 0:
                    st.error("Aşağıdaki tablo, çalışan algoritmaların ortaklaşa 'Anomali' olarak tespit ettiği riskli satırları göstermektedir.")
                    st.dataframe(df.iloc[common_indices].copy(), use_container_width=True, key="df_consensus_legacy")
                else:
                    st.success("Harika! Tüm modellerin ortaklaşa anomali dediği hiçbir satır bulunamadı.")
            tab_idx += 1
                
        # Ara Sekmeler: Her bir modelin buldukları
        for i, model in enumerate(model_results):
            with tabs[tab_idx]:
                m_name = model.get('name', f'Model {i+1}')
                m_indices = model.get('anomalies', [])
                if len(m_indices) > 0:
                    st.info(f"{m_name} algoritmasının tespit ettiği tüm anomaliler:")
                    st.dataframe(df.iloc[m_indices].copy(), use_container_width=True, key=f"df_model_{tab_idx}")
                else:
                    st.info(f"{m_name} herhangi bir anomali tespit etmedi.")
            tab_idx += 1
                
        # Son Sekme: Orijinal Veri Seti
        with tabs[-1]:
            st.write("Veri setinin orijinal hali:")
            st.dataframe(df, use_container_width=True, key="df_all_data")
            
    else:
        st.warning("Orijinal veri seti okunamadı, tablolar gösterilemiyor.")