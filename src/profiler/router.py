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
            if "collective" in meta.anomaly_characteristics:
                recommended_models = ["lstm_classifier", "xgboost_with_lags"]
            else:
                recommended_models = ["xgboost", "random_forest"]
                
        elif meta.supervision_level == "semi-supervised":
            strategy = "Semi-Supervised Anomaly Detection Pipeline"
            if "collective" in meta.anomaly_characteristics or meta.dataset_type == "time_series":
                recommended_models = ["deep_svdd", "one_class_svm"]
            else:
                recommended_models = ["deep_svdd", "one_class_svm"]
                
        else:
            # Unsupervised
            if meta.dataset_type == "time_series":
                strategy = "Time-Series Anomaly Detection Pipeline"
                recommended_models = [
                    "lstm_autoencoder"
                ]
            elif meta.dataset_type == "text":
                strategy = "Text / Log Anomaly Detection Pipeline"
                recommended_models = [
                    "isolation_forest",
                    "lof",
                    "pca",
                    "ecod",
                    "copod",
                    "hbos",
                    "ocsvm",
                    "autoencoder",
                    "vae",
                    "deep_svdd"
                ]
            elif meta.dataset_type == "graph":
                strategy = "Graph Anomaly Detection Pipeline"
                recommended_models = [
                    "graph_anomaly_detector"
                ]
            else:
                strategy = "Tabular Anomaly Detection Pipeline"
                if meta.high_dimensionality:
                    recommended_models = ["autoencoder", "vae", "deep_svdd", "isolation_forest", "ecod"]
                else:
                    recommended_models = ["isolation_forest", "ecod", "lof", "autoencoder", "vae", "deep_svdd"]

        # Adjust for collective anomalies in tabular
        if meta.supervision_level == "unsupervised" and meta.dataset_type == "tabular":
            if "collective" in meta.anomaly_characteristics:
                recommended_models.insert(0, "sliding_window_isolation_forest")

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

