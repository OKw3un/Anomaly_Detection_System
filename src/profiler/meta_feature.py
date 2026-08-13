from dataclasses import dataclass, asdict
from typing import List, Dict, Any


@dataclass
class MetaFeatureVector:

    # =========================================================
    # Dataset-level information
    # =========================================================

    total_rows: int
    total_columns: int

    dataset_type: str # Tabular mı time series mi vs.

    missing_data_ratio: float # Gerekirse imputation işlemi için

    # Bunların amacı satır sayısına kıyasla çok fazla sütun(feature) varsa mesafe tabanlı algoritmalar çökeceği için uygun filtreleme görevi görmeleri.
    feature_to_sample_ratio: float
    high_dimensionality: bool

    # =========================================================
    # Column-level information
    # =========================================================

    continuous_cols: List[str]
    categorical_cols: List[str]
    timestamp_cols: List[str]
    text_cols: List[str]
    id_cols: List[str] # Gerekli mi ?

    # Columns that may contain temporal information
    temporal_candidate_cols: List[str]

    # Possible target / anomaly label columns
    label_cols: List[str]

    # =========================================================
    # Ratios
    # =========================================================

    numeric_ratio: float
    categorical_ratio: float
    text_ratio: float
    id_ratio: float

    # =========================================================
    # Cardinality
    # =========================================================

    column_cardinality: Dict[str, int]
    column_cardinality_ratio: Dict[str, float]
    high_cardinality_cols: List[str]

    # =========================================================
    # Temporal information
    # =========================================================

    is_time_series: bool
    temporal_regularity: str

    temporal_min: Dict[str, Any]
    temporal_max: Dict[str, Any]

    # =========================================================
    # Other information
    # =========================================================

    observations_per_entity: float # Eğer bir varlığa ait birden fazla ölçüm varsa geçmiş veri geleceği etkiliyor demektir.

    # =========================================================
    # Anomaly Characteristics (Behavioral)
    # =========================================================

    anomaly_characteristics: List[str] # "point", "collective", "contextual"
    supervision_level: str # "unsupervised", "semi-supervised", "supervised"
    imbalance_ratio: float # 0.0 to 1.0 (ratio of anomaly class if available)
    
    has_trend: bool
    has_seasonality: bool
    is_stationary: bool
    
    trend_cols: List[str]
    seasonality_cols: List[str]
    stationary_cols: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)