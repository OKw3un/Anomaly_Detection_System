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

@router.post("/", response_model=AnalysisResponse)
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
        df = pd.read_csv(file_path)
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

    # --- 2. Anomaly Engine'in Çalışması ---
    # `main.py`'deki gibi AnomalyEngine sınıfını kullanıyoruz
    from src.engine.executor import AnomalyEngine
    engine = AnomalyEngine()
    
    try:
        # Motoru çalıştır
        engine_result = engine.run(df, config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analiz sırasında motor bir hata verdi: {str(e)}")

    execution_time = round(time.time() - start_time, 2)

    # --- 3. Sonuçların Formatlanıp Dönülmesi ---
    model_results_list = []
    
    # Ortak kesişimi (tüm modellerin '1' dediği) bulmak için
    common_anomalies_set = None

    for model_name, model_data in engine_result.model_results.items():
        labels = model_data.get("labels", [])
        
        # Etiketi 1 (anomali) olan satır indeksleri
        anomaly_indices = np.where(labels == 1)[0].tolist()
        
        model_results_list.append({
            "name": model_name,
            "anomalies": anomaly_indices
        })
        
        # Kesişim hesabı
        if common_anomalies_set is None:
            common_anomalies_set = set(anomaly_indices)
        else:
            common_anomalies_set = common_anomalies_set.intersection(set(anomaly_indices))

    common_anomalies = list(common_anomalies_set) if common_anomalies_set else []

    return AnalysisResponse(
        dataset_name=request.dataset_name,
        data_type=meta_vector.dataset_type,
        total_rows=meta_vector.total_rows,
        model_results=model_results_list,
        common_anomalies=common_anomalies,
        execution_time_sec=execution_time
    )