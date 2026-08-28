"""
Executor — Faz 2 Orkestratörü (Dashboard Versiyonu).

Ham veri (DataFrame) + Faz 1 Router çıktısını (config dict) alır ve:
  1. Preprocessor ile veriyi ön işlemden geçirir
  2. ModelFactory ile uygun PyOD modellerini oluşturur
  3. ModelWrapper ile eğitir
  4. Her gözlem için anomali skoru ve anomali etiketini (0/1) üretir
  5. Ortak (konsensüs) kararları tek bir DataFrame olarak döndürür
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from src.engine.preprocessor import Preprocessor
from src.engine.model_factory import ModelFactory
from src.engine.wrappers import ModelWrapper

# =================================================================
# Result Data Class
# =================================================================

@dataclass
class EngineResult:
    """Faz 2 motorunun nihai çıktısı."""

    # Her model için skor ve etiket sonuçları
    model_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Birleştirilmiş ortalama skor
    combined_scores: Optional[np.ndarray] = None

    # Orijinal satır indeksleri
    row_ids: Optional[pd.Index] = None

    # Ön işlemden geçmiş veri matrisi (Model doğrulama kısmı için)
    X_transformed: Optional[np.ndarray] = None

    # Her modelin kendi ürettiği embedding matrisleri (text/log modelleri için)
    model_embeddings: Dict[str, np.ndarray] = field(default_factory=dict)

    # Ön işlem bilgileri
    feature_names: List[str] = field(default_factory=list)
    n_features: int = 0

    # Sınıf dengesizliği kontrolü için bayrak
    is_classification_problem: bool = False

    def to_dataframe(self) -> pd.DataFrame:
        """
        Sonuçları Dashboard'a aktarılacak DataFrame formatına dönüştürür.
        """
        result_df = pd.DataFrame()

        result_df["row_id"] = (
            self.row_ids if self.row_ids is not None else range(len(next(iter(self.model_results.values()))["scores"]))
        )

        label_cols = []
        # Her modelin ayrı skoru ve 0/1 etiketi
        for model_name, model_data in self.model_results.items():
            result_df[f"score_{model_name}"] = model_data["scores"]
            result_df[f"label_{model_name}"] = model_data["labels"]
            
            if self.is_classification_problem and model_data.get("is_supervised", False):
                continue
                
            label_cols.append(f"label_{model_name}")

        # Konsensüs (Ortak Anomali) Hesaplaması
        # Satırın kaç model tarafından anomali olarak işaretlendiği
        result_df["anomaly_vote_count"] = result_df[label_cols].sum(axis=1)

        return result_df


# =================================================================
# Execution Engine
# =================================================================

class AnomalyEngine:
    """Faz 2 Ana Motoru — Profil → Ön İşlem → Model → Sonuç."""

    def __init__(self):
        self.preprocessor = Preprocessor()
        self.factory = ModelFactory()
        self.wrappers: List[ModelWrapper] = []

    def run(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any],
        y: Optional[np.ndarray] = None
    ) -> EngineResult:

        print("\n" + "=" * 60)
        print("FAZ 2: Anomali Tespiti Motoru Başlatılıyor (Dashboard Modu)")
        print("=" * 60)

        # ---------------------------------------------------------
        # Strateji: Otomatik Sınıf Dengesizliği Kontrolü
        # ---------------------------------------------------------
        is_classification_problem = False
        if y is not None and len(y) > 0:
            minority_ratio = np.sum(y == 1) / len(y)
            if minority_ratio > 0.05:
                print(f"\n[UYARI] Azınlık sınıfı (Class=1) oranı %{minority_ratio*100:.2f}. Bu veri seti bir anomali tespiti problemi değil, standart bir sınıflandırma (classification) problemidir.")
                print("Gözetimli modeller anomali oylamasına (vote_count) dahil edilmeyecektir.\n")
                is_classification_problem = True

        # ---------------------------------------------------------
        # Adım 0: Time Series Feature Engineering
        # ---------------------------------------------------------
        dataset_type = config.get("dataset_type", "tabular")
        df_raw = df.copy()
        
        if dataset_type == "time_series":
            try:
                from src.engine.ts_feature_engineering import TimeSeriesFeatureEngineer
                ts_engineer = TimeSeriesFeatureEngineer(period=21, lags=list(range(1, 11)))
                df = ts_engineer.transform(df)
            except ImportError as e:
                print(f"  [UYARI] TS Feature Engineering başlatılamadı: {e}")

        # ---------------------------------------------------------
        # Adım 1: Ön İşleme
        # ---------------------------------------------------------
        print("\n[Adım 1] Veri ön işleme (Zenginleştirilmiş Veri)...")
        preprocessing_steps = config.get("preprocessing_steps", {})

        from copy import deepcopy
        prep_ts = deepcopy(self.preprocessor)
        X, row_ids = prep_ts.fit_transform(
            df,
            preprocessing_steps,
            meta=config.get("metadata")
        )

        n_samples, n_features = X.shape
        print(f"  Ön işlem tamamlandı: {n_samples} satır × {n_features} özellik")
        
        if dataset_type == "time_series":
            print("\n[Adım 1.5] Veri ön işleme (Özel Modeller İçin Ham Veri)...")
            prep_raw = deepcopy(self.preprocessor)
            self.X_raw, _ = prep_raw.fit_transform(
                df_raw,
                preprocessing_steps,
                meta=config.get("metadata")
            )
        else:
            self.X_raw = X

        # ---------------------------------------------------------
        # Adım 2: Model Oluşturma
        # ---------------------------------------------------------
        print("\n[Adım 2] Modeller oluşturuluyor...")
        recommended = config.get("recommended_models", [])

        model_tuples = self.factory.create_models(
            recommended_models=recommended,
            n_features=n_features
        )

        if not model_tuples:
            print("  [Fallback] Önerilen modeller bulunamadı, genel amaçlı modellere geçiliyor...")
            fallback_models = ["isolation_forest", "ecod"]
            model_tuples = self.factory.create_models(
                recommended_models=fallback_models,
                n_features=n_features
            )

        if not model_tuples:
            raise RuntimeError(f"Hiçbir model oluşturulamadı. Önerilen modeller: {recommended}")

        # ---------------------------------------------------------
        # Adım 3: Model Eğitimi ve İkili Etiket Üretimi
        # ---------------------------------------------------------
        print("\n[Adım 3] Modeller eğitiliyor...\n")

        result = EngineResult(
            row_ids=row_ids,
            feature_names=self.preprocessor.feature_names,
            n_features=n_features,
            X_transformed=X,
            is_classification_problem=is_classification_problem
        )

        all_scores = []
        
        # ---------------------------------------------------------
        # ---------------------------------------------------------
        # PRE-CLUSTERING & DISTANCE-ENCODED EARLY FUSION
        # ---------------------------------------------------------
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import MinMaxScaler
        from src.engine.model_factory import ModelFactory
        
        n_samples_total = len(X)
        
        # 1. Özellikleri Ayrıştır (Semantic vs Structural)
        feature_names = self.preprocessor.feature_names
        semantic_idx = [i for i, col in enumerate(feature_names) if "_bert_" in col or "_tfidf_" in col]
        structural_idx = [i for i, col in enumerate(feature_names) if "_bert_" not in col and "_tfidf_" not in col]
        
        X_sem = X[:, semantic_idx] if len(semantic_idx) > 0 else X
        X_str = X[:, structural_idx] if len(structural_idx) > 0 else X

        k_clusters = 1
        cluster_labels = np.zeros(n_samples_total, dtype=int)
        
        # 2. Mesafe Kodlaması (Distance Encoding) - DİNAMİK KÜMELEME
        if len(semantic_idx) > 0 and n_samples_total >= 30:
            from sklearn.metrics import silhouette_score
            
            print("  [ÖN-KÜMELEME] En iyi küme sayısı (k) Silhouette Skoru ile aranıyor (3-15)...")
            best_k = 3
            best_score = -1
            
            # Veri setinin boyutuna göre maksimum aranacak küme sayısı
            max_search_k = min(15, max(4, n_samples_total // 50))
            
            for k in range(3, max_search_k + 1):
                temp_km = KMeans(n_clusters=k, random_state=42, n_init=5)
                temp_labels = temp_km.fit_predict(X_sem)
                # Sadece tek küme çıkarsa (hata durumu) skoru hesaplama
                if len(set(temp_labels)) > 1:
                    score = silhouette_score(X_sem, temp_labels)
                    if score > best_score:
                        best_score = score
                        best_k = k
                        
            k_clusters = best_k
            print(f"  [ÖN-KÜMELEME] En iyi küme sayısı {k_clusters} olarak belirlendi (Score: {best_score:.3f}). 384 boyutlu semantik uzay kümeleniyor...")
            
            km = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
            cluster_labels = km.fit_predict(X_sem)
            
            print("  [FÜZYON] Her logun semantik merkeze (centroid) uzaklığı hesaplanıp 9. özellik olarak yapısal uzaya ekleniyor...")
            # X_sem'in tüm merkezlere uzaklıklarını hesapla: Shape (N, k_clusters)
            all_distances = km.transform(X_sem)
            
            # Her bir noktanın sadece KENDİ atandığı kümenin merkezine olan uzaklığını al
            semantic_distance = np.array([all_distances[i, cluster_labels[i]] for i in range(n_samples_total)]).reshape(-1, 1)
            
            # Yapısal özelliklere (Örn: 8 boyut) semantik uzaklığı (1 boyut) ekle -> Yeni Dengeli X (9 Boyut)
            X_combined = np.hstack((X_str, semantic_distance))
        else:
            X_combined = X
            
        for model_name, model, registry_info in model_tuples:
            try:
                print(f"  > {model_name} eğitiliyor (Mesafe-Kodlamalı Küme-Farkında Mod)...")
                start_time = time.time()
                final_scores = np.zeros(n_samples_total)
                final_labels = np.zeros(n_samples_total)
                final_embeddings = None
                last_wrapper = None
                
                for c in range(k_clusters):
                    mask = (cluster_labels == c)
                    
                    if dataset_type == "time_series" and model_name in ["lstm_autoencoder", "supervised_xgboost", "supervised_lightgbm"]:
                        X_c = self.X_raw[mask]
                    else:
                        X_c = X_combined[mask]
                        
                    y_c = y[mask] if y is not None else None
                    
                    if len(X_c) < 5:
                        continue  # Çok küçük kümeleri atla
                        
                    factory = ModelFactory()
                    created = factory.create_models([model_name], n_features=X_c.shape[1])
                    if not created:
                        continue
                        
                    _, fresh_model, _ = created[0]
                    
                    if model_name == "hbos":
                        fresh_model.n_bins = max(2, min(10, len(X_c) // 5))
                    
                    is_supervised_c = registry_info.get("is_supervised", False)
                    
                    if is_supervised_c and y_c is not None:

                        scores_c = np.zeros(len(X_c))
                        labels_c = np.zeros(len(X_c))
                        
                        min_class_count = 0
                        counts = np.bincount(y_c.astype(int))
                        if len(counts) > 1:
                            min_class_count = np.min(counts)
                            
                        if min_class_count >= 5:
                            from sklearn.metrics import precision_score
                            from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
                            train_prec_list = []
                            val_prec_list = []
                            
                            dataset_type = config.get("dataset_type", "tabular")
                            if dataset_type == "time_series":
                                cv = TimeSeriesSplit(n_splits=5)
                                split_generator = cv.split(X_c)
                            else:
                                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                                split_generator = cv.split(X_c, y_c)
                                
                            for train_idx, val_idx in split_generator:
                                X_train, y_train = X_c[train_idx], y_c[train_idx]
                                X_val, y_val = X_c[val_idx], y_c[val_idx]
                                
                                factory_oof = ModelFactory()
                                fresh_model_oof = factory_oof.create_models([model_name], n_features=X_c.shape[1])[0][1]
                                wrapper_oof = ModelWrapper(model_name, fresh_model_oof, registry_info)
                                wrapper_oof.fit(X_train, y_train)
                                
                                scores_c[val_idx] = wrapper_oof.predict_anomaly_scores(X_val)
                                labels_c[val_idx] = wrapper_oof.predict_labels(X_val)
                                
                                train_labels = wrapper_oof.predict_labels(X_train)
                                train_prec_list.append(precision_score(y_train, train_labels, zero_division=0))
                                val_prec_list.append(precision_score(y_val, labels_c[val_idx], zero_division=0))
                                
                            cv_name = "Time-Series CV" if dataset_type == "time_series" else "K-Fold"
                            print(f"      [{cv_name}] Eğitim Precision: {np.mean(train_prec_list):.2%} | Test Precision: {np.mean(val_prec_list):.2%}")
                            wrapper = wrapper_oof
                        else:
                            wrapper = ModelWrapper(model_name, fresh_model, registry_info)
                            wrapper.fit(X_c, y_c)
                            scores_c = wrapper.get_anomaly_scores()
                            labels_c = wrapper.predict_labels(X_c)
                    else:
                        wrapper = ModelWrapper(model_name, fresh_model, registry_info)
                        wrapper.fit(X_c, y_c)
                        scores_c = wrapper.get_anomaly_scores()
                        labels_c = np.zeros(len(X_c))
                    
                    # YEREL MIN-MAX NORMALİZASYONU İPTAL: Ham skor doğrudan atanıyor
                        
                    final_scores[mask] = scores_c
                    if is_supervised_c:
                        final_labels[mask] = labels_c
                    last_wrapper = wrapper
                    
                    # Embeddingleri birleştir
                    if registry_info.get("supports_embedding", False) or hasattr(wrapper.model, "embeddings_"):
                        emb = None
                        if hasattr(wrapper.model, "embeddings_") and wrapper.model.embeddings_ is not None:
                            emb = wrapper.model.embeddings_
                        else:
                            try:
                                emb = wrapper.get_embeddings()
                            except Exception:
                                pass
                        
                        if emb is not None:
                            if final_embeddings is None:
                                final_embeddings = np.zeros((n_samples_total, emb.shape[1]))
                            final_embeddings[mask] = emb
    
                # GLOBAL NORMALİZASYON (Tüm kümelerden toplanan ham skorlar üzerinde)
                scores = final_scores
                if len(scores) > 1:
                    global_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
                    scores = global_scaler.fit_transform(scores.reshape(-1, 1)).flatten()
                wrapper = last_wrapper if last_wrapper is not None else wrapper

                if final_embeddings is not None and wrapper is not None:
                    if hasattr(wrapper, "model") and wrapper.model is not None:
                        wrapper.model.embeddings_ = final_embeddings
                
                is_supervised = False
                if wrapper is not None and hasattr(wrapper, "registry_info"):
                    is_supervised = wrapper.registry_info.get("is_supervised", False)
                
                if is_supervised:
                    labels = final_labels
                else:
                    # --- GERÇEK DİNAMİK EŞİKLEME (MAD YÖNTEMİ) ---
                    median_score = np.median(scores)
                    
                    # Medyandan mutlak sapmaların medyanını al
                    mad = np.median(np.abs(scores - median_score))
                    
                    # Eğer MAD 0 çıkarsa (çok yoğun yığılma varsa) standart sapmayı yedek olarak kullan
                    if mad == 0:
                        mad = np.std(scores)
                        
                    # Dinamik Eşik: Medyan + 3 * MAD (Genellikle 3 veya 5 çarpanı kullanılır)
                    # Veri setinin skor yayılımına göre threshold kendi kendine genişler veya daralır.
                    dynamic_threshold = median_score + (5 * mad)
                    
                    labels = (scores >= dynamic_threshold).astype(int)
                    
                    dynamic_contam = np.sum(labels) / len(labels)
                    print(f"      [DİNAMİK EŞİK] MAD Yöntemi ile Threshold: {dynamic_threshold:.4f}, Hesaplanan Anomali Oranı: {dynamic_contam:.2%}")
                
                # Her ihtimale karşı 0 tespit edildiyse
                if np.sum(labels) == 0 and hasattr(wrapper.model, 'labels_'):
                    if len(wrapper.model.labels_) == len(labels):
                        labels = wrapper.model.labels_.astype(int)

                result.model_results[model_name] = {
                    "scores": scores,
                    "labels": labels,
                    "is_supervised": is_supervised
                }

                # Modelin kendi ürettiği embedding'i varsa kaydet
                if final_embeddings is not None:
                    result.model_embeddings[model_name] = final_embeddings

                all_scores.append(scores)
                self.wrappers.append(wrapper)

                elapsed_time = time.time() - start_time
                print(
                    f"    [OK] Tamamlandı - Süre: {elapsed_time:.1f}s, Ortalama Skor: {scores.mean():.4f}, "
                    f"Tespit Edilen Anomali Sayısı: {np.sum(labels)}"
                )

            except Exception as e:
                print(f"    [HATA] {model_name} eğitilirken sorun oluştu: {e}")

        # Adım 4 (Ensemble) Kapatıldı

        print("\n" + "=" * 60)
        print("FAZ 2 TAMAMLANDI")
        print("=" * 60)

        # Tabloyu Oluşturma
        results_df = result.to_dataframe()

        # Terminalde Temiz Gösterim İçin Pandas Ayarları
        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_colwidth", None)
        pd.set_option("display.width", 1000)

        # Ortak Anomalileri (En az 1 algoritmanın '1' dediği satırlar) filtrele
        anomalies_df = results_df[results_df["anomaly_vote_count"] > 0]
        
        print("\n" + "-" * 60)
        print(f"Toplam {len(anomalies_df)} adet anomali içeren satır tespit edildi.")
        
        if not anomalies_df.empty:
            # Görüntüleme için dinamik sütunları seç (row_id, vote_count ve label'lar)
            cols_to_show = ["row_id", "anomaly_vote_count"] + [col for col in results_df.columns if col.startswith("label_")]
            # En çok algoritmanın "Anomali" dediği satırları en üste alacak şekilde sırala
            anomalies_df = anomalies_df.sort_values(by="anomaly_vote_count", ascending=False)
            print(anomalies_df[cols_to_show].head(15)) # İlk 15 satırı göster
        else:
            print("Hiçbir algoritma anomali tespit etmedi.")
        print("-" * 60)

        # Dashboard / İnceleme için CSV'ye kaydetme kaldırıldı
        # results_df.to_csv("faz2_anomali_sonuclari.csv", index=False, encoding="utf-8")
        # print(f"\n[BİLGİ] Tüm analiz detayları 'faz2_anomali_sonuclari.csv' dosyasına kaydedildi.")

        return result