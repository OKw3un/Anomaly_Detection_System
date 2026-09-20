import os
from fastapi import APIRouter, HTTPException
from src.api.schemas import DatasetListResponse, RecommendationResponse

# Senin geliştirdiğin motorları (engine/profiler) buraya dahil ediyoruz
# Kendi dosya yapına göre import yollarını düzeltmen gerekebilir
from src.profiler.profiler import DataProfiler 
from src.profiler.router import ModelRouter 

router = APIRouter(
    prefix="/datasets",
    tags=["Datasets"]
)

DATA_FOLDER = "data" # Veri setlerinin bulunduğu klasör

@router.get("/", response_model=DatasetListResponse)
def list_datasets():
    """Lokal 'data' klasöründeki ve alt klasörlerindeki mevcut veri setlerini listeler."""
    if not os.path.exists(DATA_FOLDER):
        return DatasetListResponse(datasets=[])
        
    dataset_files = []
    
    # os.listdir yerine os.walk kullanıyoruz (Alt klasörlere de girer)
    for root, dirs, files in os.walk(DATA_FOLDER):
        for file in files:
            if file.endswith((".csv", ".json")):
                # Dosyanın ana klasöre göre konumunu buluyoruz
                rel_dir = os.path.relpath(root, DATA_FOLDER)
                
                if rel_dir == ".":
                    # Dosya doğrudan 'data' klasörünün içindeyse
                    dataset_files.append(file)
                else:
                    # Dosya bir alt klasördeyse (örn: alt_klasor/veri.csv)
                    # Windows (ters slash \) ve Mac/Linux (düz slash /) uyumu için replace kullanıyoruz
                    full_path = os.path.join(rel_dir, file).replace("\\", "/")
                    dataset_files.append(full_path)
                    
    return DatasetListResponse(datasets=dataset_files)

@router.get("/{dataset_name:path}/recommend", response_model=RecommendationResponse)
def get_model_recommendations(dataset_name: str):
    """
    Seçilen veri setini DataProfiler ile analiz edip 
    ModelRouter üzerinden uygun iki modeli önerir.
    """
    file_path = os.path.join(DATA_FOLDER, dataset_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Veri seti bulunamadı.")

    import pandas as pd
    import json
    
    # 1. Pandas ile veriyi oku
    try:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
            
        elif file_path.endswith(".json"):
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # İç içe geçmiş (nested) JSON yapısını düz bir Pandas tablosuna çevir
            df = pd.json_normalize(json_data)
            
        else:
            raise ValueError("Sadece .csv ve .json formatları desteklenmektedir.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Veri seti okunamadı: {str(e)}")

    # 2. DataProfiler'a DataFrame'i vererek incele
    profiler = DataProfiler(df, dataset_name=dataset_name, data_dir=os.path.dirname(file_path))
    meta_vector = profiler.profile()
    
    # 3. ModelRouter ile uygun modelleri seç
    router = ModelRouter()
    config = router.route(meta_vector)
    recommended_models = config.get("recommended_models", [])

    return RecommendationResponse(
        dataset_name=dataset_name,
        data_type=meta_vector.dataset_type,
        recommended_algorithms=recommended_models,
        message="Sistem tarafından veri yapısına göre önerilen modellerdir."
    )