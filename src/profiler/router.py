from typing import Dict, Any
from src.profiler.meta_feature import MetaFeatureVector

class ModelRouter:

    def route(
        self,
        meta: MetaFeatureVector
    ) -> Dict[str, Any]:

        recommended_models = []
        strategy = ""

        # =====================================================
        # Strategy & Model Selection
        # =====================================================

        if meta.supervision_level == "supervised":
            strategy = "Supervised Classification Pipeline"
            recommended_models = ["xgboost", "random_forest"]
                
        elif meta.supervision_level == "semi-supervised":
            strategy = "Semi-Supervised Anomaly Detection Pipeline"
            if meta.total_rows >= 5000:
                # Yeterli veri → Deep SVDD hipersfer öğrenebilir
                recommended_models = ["deep_svdd", "isolation_forest"]
            else:
                # Az veri → DL overfitting riski, sadece geleneksel one-class
                recommended_models = ["ocsvm", "xgbod"]
                
        else:
            # Unsupervised
            if meta.dataset_type == "time_series":
                strategy = "Time-Series Anomaly Detection Pipeline"
                if meta.total_rows >= 5000:
                    # Yeterli veri → LSTM temporal bağlamı öğrenebilir
                    recommended_models = ["isolation_forest"] #time serieste lstm autoencoder ve autoencoder cok fazla vakit istiyor.
                else:
                    # Az veri → DL overfitting yapar, hızlı geleneksel yöntemler
                    recommended_models = ["isolation_forest", "ecod", "ocsvm"]

            elif meta.dataset_type == "text":
                strategy = "Text / Log Anomaly Detection Pipeline"
                if meta.high_dimensionality:
                    # Yüksek boyutlu TF-IDF → boyut indirgeme odaklı modeller
                    recommended_models = ["autoencoder", "deep_svdd", "isolation_forest", "pca"]
                else:
                    # Düşük boyut → hızlı geleneksel yoğunluk/histogram modelleri
                    recommended_models = ["isolation_forest", "lof", "copod", "hbos"]

            elif meta.dataset_type == "graph":
                strategy = "Graph Anomaly Detection Pipeline"
                recommended_models = ["graph_dominant"]

            else:
                strategy = "Tabular Anomaly Detection Pipeline"
                if meta.high_dimensionality and meta.total_rows >= 5000:
                    # Yüksek boyut + büyük veri → DL modelleri tam kapasiteyle çalışır
                    recommended_models = ["autoencoder", "vae", "deep_svdd", "isolation_forest", "ecod"]
                elif meta.high_dimensionality:
                    # Yüksek boyut + az veri → DL overfitting yapar, PCA + geleneksel
                    recommended_models = ["pca", "isolation_forest", "ecod", "ocsvm"]
                elif meta.total_rows >= 5000:
                    # Düşük boyut + büyük veri → LOF O(n²) çöker, ölçeklenebilir modeller
                    recommended_models = ["isolation_forest", "ecod", "copod", "autoencoder", "deep_svdd"]
                else:
                    # Düşük boyut + az veri → yoğunluk/sınır modelleri parlar
                    recommended_models = ["lof", "ocsvm", "isolation_forest", "hbos", "copod"]



        # =====================================================
        # Preprocessing
        # =====================================================

        preprocessing = {

            "impute_missing":
                meta.missing_data_ratio > 0,

            "scale_continuous":
                len(meta.continuous_cols) > 0,

            "encode_categorical":
                len(meta.categorical_cols) > 0,

            # IDs should normally not be used directly
            "exclude_id_columns":
                meta.id_cols,

            # Labels must not be used during unsupervised
            # anomaly model training
            "exclude_label_columns":
                meta.label_cols,

            "text_columns":
                meta.text_cols,
                
            "remove_seasonality_columns":
                meta.seasonality_cols,
                
            "detrend_columns":
                meta.trend_cols
        }

        # =====================================================
        # Additional information for Phase 2
        # =====================================================

        return {

            "dataset_type":
                meta.dataset_type,

            "strategy":
                strategy,

            "recommended_models":
                recommended_models,

            "preprocessing_steps":
                preprocessing,

            "metadata": {

                "rows":
                    meta.total_rows,

                "columns":
                    meta.total_columns,

                "feature_to_sample_ratio":
                    meta.feature_to_sample_ratio,

                "high_dimensionality":
                    meta.high_dimensionality,

                "label_columns":
                    meta.label_cols,

                "id_columns":
                    meta.id_cols,

                "temporal_candidates":
                    meta.temporal_candidate_cols,
                    
                "supervision_level": 
                    meta.supervision_level,
                    
                "anomaly_characteristics": 
                    meta.anomaly_characteristics,
                    
                "is_stationary": 
                    meta.is_stationary
            }
        }

