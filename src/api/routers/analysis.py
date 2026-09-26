import os
import time
from fastapi import APIRouter, HTTPException
from src.api.schemas import AnalysisRequest, AnalysisResponse

# Senin yazdığın sınıfları import ediyoruz
from src.profiler.profiler import DataProfiler
from src.profiler.router import ModelRouter
from src.engine.model_factory import ModelFactory

router = APIRouter(
    prefix="/analyze",
    tags=["Analysis"]
)

@router.get("/models")
def get_available_models():
    """Backend'de kayıtlı tüm modelleri liste olarak döner."""
    return {"models": ModelFactory.list_available_models()}


DATA_FOLDER = "data"

@router.post("", response_model=AnalysisResponse)
@router.post("/", response_model=AnalysisResponse, include_in_schema=False)
def run_analysis(request: AnalysisRequest):
    """
    Kullanıcının seçtiği veri setini ve algoritmaları alır.
    A ve B algoritmalarını çalıştırır, buldukları anomalileri ve 
    ortak kesişimlerini (common anomalies) hesaplayıp JSON olarak döner.
    """
    import pandas as pd
    import numpy as np

    start_time = time.time()
    file_path = os.path.join(DATA_FOLDER, request.dataset_name)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Veri seti bulunamadı.")

    # Veriyi Oku
    try:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith(".json"):
            # Graf JSON dosyaları özel işlem gerektirir (iç içe list/dict yapıları var)
            from src.engine.graph_feature_engineering import GraphFeatureEngineer
            try:
                graph_engineer = GraphFeatureEngineer(file_path)
                df = graph_engineer.transform()
            except Exception:
                # Graf formatı değilse düz json_normalize dene
                import json
                with open(file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                if isinstance(json_data, list):
                    df = pd.json_normalize(json_data)
                else:
                    df = pd.json_normalize(json_data)
        else:
            raise ValueError("Sadece .csv ve .json formatları desteklenmektedir.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Veri seti okunamadı: {str(e)}")

    # --- 1. Veri Profilleme ve Model Seçimi ---
    profiler = DataProfiler(df, data_dir=os.path.dirname(file_path))
    meta_vector = profiler.profile()
    
    model_router = ModelRouter()
    config = model_router.route(meta_vector)
    
    alg_list = request.algorithms

    # Eğer kullanıcı "auto_route" seçtiyse veya model seçmeyi unuttuysa:
    if request.auto_route or not alg_list:
        alg_list = config.get("recommended_models", [])

    config["recommended_models"] = alg_list

    # --- 2. Label Sütununu Çıkar (Supervised modeller için) ---
    y = None
    label_cols = config.get("metadata", {}).get("label_columns", [])
    if label_cols:
        label_col = label_cols[0]
        if label_col in df.columns:
            y_series = df[label_col]
            # Sayısal değilse encode et (string label → 0/1)
            if not pd.api.types.is_numeric_dtype(y_series):
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y = le.fit_transform(y_series.fillna("unknown"))
            else:
                y = y_series.fillna(0).values.astype(int)
            print(f"  [BİLGİ] Label sütunu '{label_col}' tespit edildi. Sınıf dağılımı: {np.bincount(y)}")

    # --- 3. Anomaly Engine'in Çalışması ---
    # `main.py`'deki gibi AnomalyEngine sınıfını kullanıyoruz
    from src.engine.executor import AnomalyEngine
    engine = AnomalyEngine()
    
    try:
        # Motoru çalıştır
        engine_result = engine.run(df, config, y=y)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analiz sırasında motor bir hata verdi: {str(e)}")

    execution_time = round(time.time() - start_time, 2)

    # --- 3. Sonuçların Formatlanıp Dönülmesi ---
    model_results_list = []
    
    from collections import Counter
    vote_counter = Counter()
    num_models = len(engine_result.model_results)

    for model_name, model_data in engine_result.model_results.items():
        labels = model_data.get("labels", [])
        
        # Etiketi 1 (anomali) olan satır indeksleri
        anomaly_indices = np.where(labels == 1)[0].tolist()
        
        model_results_list.append({
            "name": model_name,
            "anomalies": anomaly_indices,
            "auprc": model_data.get("auprc")
        })
        
        # Oyları say
        vote_counter.update(anomaly_indices)

    consensus_levels = {}
    common_anomalies = []

    if num_models > 0:
        tam_kesisim = [idx for idx, count in vote_counter.items() if count == num_models]
        n_eksi_1 = [idx for idx, count in vote_counter.items() if count >= num_models - 1]
        n_eksi_2 = [idx for idx, count in vote_counter.items() if count >= num_models - 2]
        
        common_anomalies = tam_kesisim

        if num_models <= 3:
            consensus_levels[f"Tam Kesişim ({num_models}/{num_models})"] = tam_kesisim
        elif 3 < num_models <= 6:
            consensus_levels[f"Tam Kesişim ({num_models}/{num_models})"] = tam_kesisim
            consensus_levels[f"Çoğunluk Kesişimi (Min {num_models-1}/{num_models})"] = n_eksi_1
        else:
            consensus_levels[f"Tam Kesişim ({num_models}/{num_models})"] = tam_kesisim
            consensus_levels[f"Güçlü Kesişim (Min {num_models-1}/{num_models})"] = n_eksi_1
            consensus_levels[f"Çoğunluk Kesişimi (Min {num_models-2}/{num_models})"] = n_eksi_2

    return AnalysisResponse(
        dataset_name=request.dataset_name,
        data_type=meta_vector.dataset_type,
        total_rows=meta_vector.total_rows,
        model_results=model_results_list,
        common_anomalies=common_anomalies,
        consensus_levels=consensus_levels,
        label_columns=label_cols,
        execution_time_sec=execution_time
    )