from pydantic import BaseModel, Field
from typing import List, Optional

# --- GİRDİ (REQUEST) ŞEMALARI ---

class AnalysisRequest(BaseModel):
    dataset_name: str = Field(..., description="Analiz edilecek veri setinin dosya adı (örn: dataset1.csv)")
    
    # Kullanıcı otomatik yönlendirme mi istiyor, yoksa kendi mi seçecek?
    auto_route: bool = Field(default=True, description="True ise DataProfiler/ModelRouter otomatik model seçer")
    
    # Eğer auto_route False ise, kullanıcının seçtiği modeller burada gelir
    algorithms: Optional[List[str]] = Field(default=None, description="Manuel seçim için algoritmalar listesi (örn: ['IForest', 'LOF'])")

# --- ÇIKTI (RESPONSE) ŞEMALARI ---

class DatasetListResponse(BaseModel):
    datasets: List[str] = Field(..., description="Lokal klasördeki mevcut veri setlerinin listesi")

class ModelResult(BaseModel):
    name: str
    anomalies: List[int] = Field(..., description="Modelin anomali bulduğu satır indeksleri")
    auprc: Optional[float] = Field(default=None, description="Eğer etiket (y) mevcutsa hesaplanan AUPRC skoru")

class AnalysisResponse(BaseModel):
    dataset_name: str
    data_type: str = Field(..., description="DataProfiler tarafından tespit edilen veri tipi (tabular, time-series, log)")
    total_rows: int
    
    # Tüm modellerin Sonuçları
    model_results: List[ModelResult] = Field(..., description="Seçilen modellerin bulduğu anomali sonuçları")
    
    # Tüm modellerin KESİŞİMİ (Ortak anomaliler)
    common_anomalies: List[int] = Field(..., description="Tüm modellerin de anomali olarak işaretlediği ortak satır indeksleri")
    
    # Dinamik oylama tabanlı kesişimler (N-1, N-2 vb.)
    consensus_levels: Optional[dict] = Field(default_factory=dict, description="Farklı seviyelerdeki (N, N-1, N-2) ortak anomaliler")
    
    # Tespit edilen etiket sütunları (label, class vb.)
    label_columns: Optional[List[str]] = Field(default_factory=list, description="Veri profilinden tespit edilen etiket sütunları")
    
    execution_time_sec: float = Field(..., description="Analizin kaç saniye sürdüğü")
    
class RecommendationResponse(BaseModel):
    dataset_name: str
    data_type: str = Field(..., description="Tabular, time-series veya log")
    recommended_algorithms: List[str] = Field(..., description="Önerilen algoritmalar")
    message: str = Field(default="Bu veri seti için sistemin önerdiği modellerdir.")