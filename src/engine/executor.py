"""
Executor — Faz 2 Orkestratörü.

Ham veri (DataFrame) + Faz 1 Router çıktısını (config dict) alır ve:
  1. Preprocessor ile veriyi ön işlemden geçirir
  2. ModelFactory ile uygun PyOD modellerini oluşturur
  3. ModelWrapper ile eğitir
  4. Her gözlem için anomali skoru + embedding üretir
  5. Sonuçları tek bir DataFrame olarak döndürür
"""

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

    # Her model için sonuçlar
    model_results: Dict[str, Dict[str, Any]] = field(
        default_factory=dict
    )

    # Birleştirilmiş sonuçlar (ensemble)
    combined_scores: Optional[np.ndarray] = None
    combined_embeddings: Optional[np.ndarray] = None

    # Orijinal satır indeksleri
    row_ids: Optional[pd.Index] = None

    # Ön işlem bilgileri
    feature_names: List[str] = field(default_factory=list)
    n_features: int = 0

    def to_dataframe(self) -> pd.DataFrame:
        """
        Sonuçları Faz 3'e aktarılacak DataFrame formatına dönüştürür.

        Çıktı sütunları:
          - row_id:         Orijinal satır indeksi
          - anomaly_score:  Birleştirilmiş anomali skoru (0.0 – 1.0)
          - embedding_0..N: Embedding vektörünün her boyutu
          - model_scores:   Her modelin ayrı skoru (dict)
        """

        result_df = pd.DataFrame()

        result_df["row_id"] = (
            self.row_ids
            if self.row_ids is not None
            else range(len(self.combined_scores))
        )

        result_df["anomaly_score"] = self.combined_scores

        # Embedding sütunları
        if self.combined_embeddings is not None:
            embedding_cols = [
                f"embedding_{i}"
                for i in range(self.combined_embeddings.shape[1])
            ]
            embedding_df = pd.DataFrame(
                self.combined_embeddings,
                columns=embedding_cols
            )
            result_df = pd.concat(
                [result_df.reset_index(drop=True),
                 embedding_df.reset_index(drop=True)],
                axis=1
            )

        # Her modelin ayrı skoru
        for model_name, model_data in self.model_results.items():
            result_df[f"score_{model_name}"] = model_data["scores"]

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
        config: Dict[str, Any]
    ) -> EngineResult:
        """
        Faz 2 pipeline'ının tamamını çalıştırır.

        Args:
            df:     Ham DataFrame (orijinal veri)
            config: Faz 1 Router çıktısı (preprocessing_steps,
                    recommended_models, metadata vs.)

        Returns:
            EngineResult: Anomali skorları, embedding'ler ve
                          model bazlı sonuçlar
        """

        print("\n" + "=" * 60)
        print("FAZ 2: Anomali Tespiti Motoru Başlatılıyor")
        print("=" * 60)

        # ---------------------------------------------------------
        # Adım 1: Ön İşleme
        # ---------------------------------------------------------

        print("\n[Adım 1] Veri ön işleme...")

        preprocessing_steps = config.get("preprocessing_steps", {})

        X, row_ids = self.preprocessor.fit_transform(
            df,
            preprocessing_steps,
            meta=config.get("metadata")
        )

        n_samples, n_features = X.shape

        print(
            f"  Ön işlem tamamlandı: "
            f"{n_samples} satır × {n_features} özellik"
        )
        print(
            f"  Özellikler: "
            f"{self.preprocessor.feature_names}"
        )

        # ---------------------------------------------------------
        # Adım 2: Model Oluşturma
        # ---------------------------------------------------------

        print("\n[Adım 2] Modeller oluşturuluyor...")

        recommended = config.get("recommended_models", [])

        model_tuples = self.factory.create_models(
            recommended_models=recommended,
            n_features=n_features
        )

        # Fallback: Önerilen modeller PyOD'de tanımlı değilse
        # genel amaçlı modellere düş
        if not model_tuples:
            print(
                "  [Fallback] Önerilen modeller bulunamadı, "
                "genel amaçlı modellere geçiliyor..."
            )
            fallback_models = ["isolation_forest", "ecod"]
            model_tuples = self.factory.create_models(
                recommended_models=fallback_models,
                n_features=n_features
            )

        if not model_tuples:
            raise RuntimeError(
                "Hiçbir model oluşturulamadı. "
                f"Önerilen modeller: {recommended}"
            )

        # ---------------------------------------------------------
        # Adım 3: Model Eğitimi ve Sonuç Toplama
        # ---------------------------------------------------------

        print("\n[Adım 3] Modeller eğitiliyor...\n")

        result = EngineResult(
            row_ids=row_ids,
            feature_names=self.preprocessor.feature_names,
            n_features=n_features
        )

        all_scores = []
        all_embeddings = []

        for model_name, model, registry_info in model_tuples:

            print(f"  > {model_name} egitiliyor...")

            wrapper = ModelWrapper(
                model_name=model_name,
                model=model,
                registry_info=registry_info,
                embedding_dim=16
            )

            try:
                wrapper.fit(X)

                scores = wrapper.get_anomaly_scores()
                embeddings = wrapper.get_embeddings()

                result.model_results[model_name] = {
                    "scores": scores,
                    "embeddings": embeddings,
                    "category": registry_info.get("category"),
                    "supports_native_embedding":
                        registry_info.get("supports_embedding", False)
                }

                all_scores.append(scores)
                all_embeddings.append(embeddings)

                self.wrappers.append(wrapper)

                print(
                    f"    [OK] Tamamlandi - "
                    f"Ortalama skor: {scores.mean():.4f}, "
                    f"Embedding boyutu: {embeddings.shape[1]}"
                )

            except Exception as e:
                print(f"    [HATA] {e}")

        # ---------------------------------------------------------
        # Adım 4: Ensemble (Birleştirme)
        # ---------------------------------------------------------

        if all_scores:

            print("\n[Adım 4] Sonuçlar birleştiriliyor...")

            # Anomali skorları: Modellerin ortalaması
            result.combined_scores = np.mean(
                np.column_stack(all_scores),
                axis=1
            )

            # Embedding'ler: En iyi modelin embedding'ini kullan
            # Öncelik: DL modeli > ML modeli
            best_embedding = None

            for model_name, model_data in result.model_results.items():
                if model_data["supports_native_embedding"]:
                    best_embedding = model_data["embeddings"]
                    print(
                        f"  Embedding kaynağı: {model_name} "
                        f"(DL — native)"
                    )
                    break

            if best_embedding is None and all_embeddings:
                best_embedding = all_embeddings[0]
                first_model = list(result.model_results.keys())[0]
                print(
                    f"  Embedding kaynağı: {first_model} "
                    f"(PCA fallback)"
                )

            result.combined_embeddings = best_embedding

        # ---------------------------------------------------------
        # Özet
        # ---------------------------------------------------------

        print("\n" + "=" * 60)
        print("FAZ 2 TAMAMLANDI")
        print("=" * 60)

        if result.combined_scores is not None:
            print(
                f"  Toplam gözlem:       {len(result.combined_scores)}"
            )
            print(
                f"  Ortalama skor:       "
                f"{result.combined_scores.mean():.4f}"
            )
            print(
                f"  Skor std:            "
                f"{result.combined_scores.std():.4f}"
            )

            # İstatistiksel Dinamik Eşik (Z-Score / 3-Sigma Kuralı)
            # Verinin kirlilik oranını (contamination) varsaymak yerine,
            # anomali skorlarının istatistiksel dağılımını (sapmasını) kullanırız.
            mu = result.combined_scores.mean()
            sigma = result.combined_scores.std()
            
            # Kural: Ortalamadan 3 Standart Sapma uzaklaşan noktalar anomalidir.
            # Eşik (Threshold) = μ + 3σ
            threshold = float(mu + (3 * sigma))
            
            anomaly_count = (
                result.combined_scores >= threshold
            ).sum()

            print(
                f"  Anomali (>={threshold:.4f}):    "
                f"{anomaly_count} / {len(result.combined_scores)} "
                f"({anomaly_count / len(result.combined_scores) * 100:.2f}%)"
            )

        if result.combined_embeddings is not None:
            print(
                f"  Embedding boyutu:    "
                f"{result.combined_embeddings.shape[1]}"
            )

        print(
            f"  Eğitilen modeller:   "
            f"{list(result.model_results.keys())}"
        )

        return result
