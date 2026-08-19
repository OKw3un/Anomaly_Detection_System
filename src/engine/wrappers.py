"""
Wrappers — PyOD modellerinin çıktılarını tek tipleştiren evrensel kılıf.

Her modelden aynı arayüzle:
  - fit(X)
  - get_anomaly_scores() → normalize edilmiş skorlar (0.0 – 1.0)
  - get_embeddings()     → gizli temsil vektörleri (latent space)

Geleneksel ML modelleri embedding üretmediği için bu durumda
PCA ile boyut küçültülmüş veri embedding olarak döndürülür.
"""

import numpy as np
from typing import Any, Dict, Optional

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


class ModelWrapper:
    """
    Tek bir PyOD modelini saran, anomali skoru ve embedding
    çıktılarını standartlaştıran kılıf sınıfı.
    """

    def __init__(
        self,
        model_name: str,
        model: Any,
        registry_info: Dict[str, Any],
        embedding_dim: int = 16
    ):
        self.model_name = model_name
        self.model = model
        self.registry_info = registry_info
        self.embedding_dim = embedding_dim

        self._is_fitted = False
        self._X: Optional[np.ndarray] = None
        self._anomaly_scores: Optional[np.ndarray] = None
        self._embeddings: Optional[np.ndarray] = None

    # ==========================================================
    # Public API
    # ==========================================================

    def fit(self, X: np.ndarray) -> "ModelWrapper":
        """
        Modeli eğitir ve anomali skorlarını hesaplar.

        Args:
            X: (n_samples, n_features) ön işlemden geçmiş veri matrisi
        """

        self._X = X

        # PyOD modelleri fit() ile eğitilir
        self.model.fit(X)

        # PyOD: decision_scores_ → ham anomali skorları (eğitim verisi)
        raw_scores = self.model.decision_scores_

        # Min-Max normalizasyonu → [0.0, 1.0] aralığına getir
        scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        self._anomaly_scores = (
            scaler.fit_transform(
                raw_scores.reshape(-1, 1)
            ).flatten()
        )

        # Embedding hesapla
        self._embeddings = self._extract_embeddings(X)

        self._is_fitted = True

        return self

    def get_anomaly_scores(self) -> np.ndarray:
        """
        Normalize edilmiş anomali skorlarını döndürür.
        Yüksek skor = daha anomalik.

        Returns:
            (n_samples,) numpy dizisi, [0.0, 1.0] aralığında
        """

        if not self._is_fitted:
            raise RuntimeError(
                "Model henüz eğitilmedi. Önce fit() çağırın."
            )

        return self._anomaly_scores

    def get_embeddings(self) -> np.ndarray:
        """
        Her gözlem için sıkıştırılmış temsil vektörünü döndürür.

        DL modelleri: Darboğaz (bottleneck) katman çıktısı
        ML modelleri: PCA ile boyut küçültülmüş veri

        Returns:
            (n_samples, embedding_dim) numpy dizisi
        """

        if not self._is_fitted:
            raise RuntimeError(
                "Model henüz eğitilmedi. Önce fit() çağırın."
            )

        return self._embeddings

    # ==========================================================
    # Internal — Embedding Extraction
    # ==========================================================

    def _extract_embeddings(self, X: np.ndarray) -> np.ndarray:
        """
        Model türüne göre embedding çıkarır.

        DL modelleri (AutoEncoder, VAE, DeepSVDD):
            PyOD modelinin iç sinir ağından darboğaz katman çıktısını alır.

        Geleneksel ML modelleri (IForest, LOF, ECOD, OCSVM):
            PCA ile verinin boyutunu embedding_dim'e indirger.
        """

        supports = self.registry_info.get("supports_embedding", False)

        if supports:
            return self._extract_dl_embeddings(X)
        else:
            return self._extract_pca_embeddings(X)

    def _extract_dl_embeddings(self, X: np.ndarray) -> np.ndarray:
        """
        Derin öğrenme modellerinin encoder kısmından
        darboğaz (bottleneck) çıktısını alır.
        """

        try:
            # PyOD AutoEncoder / VAE modelleri
            # encoding_dim_ veya intermediate model ile erişim

            # Yöntem 1: PyOD modelinin iç encoding modeline eriş
            if hasattr(self.model, "model_") and self.model.model_ is not None:

                inner_model = self.model.model_

                # Keras/TF modeli ise ara katman çıktısını al
                try:
                    import torch

                    if hasattr(inner_model, "encoder"):
                        # PyTorch tabanlı modeller
                        inner_model.eval()
                        with torch.no_grad():
                            tensor_X = torch.FloatTensor(X)
                            embeddings = (
                                inner_model.encoder(tensor_X)
                                .cpu()
                                .numpy()
                            )
                        return embeddings

                except ImportError:
                    pass

            # Yöntem 2: Fallback — model'in iç nöron yapısından
            # bottleneck boyutunu al ve PCA ile eşle
            print(
                f"[Wrapper] {self.model_name}: DL embedding "
                f"çıkarılamadı, PCA fallback kullanılıyor."
            )
            return self._extract_pca_embeddings(X)

        except Exception as e:
            print(
                f"[Wrapper] {self.model_name}: Embedding hatası "
                f"({e}), PCA fallback kullanılıyor."
            )
            return self._extract_pca_embeddings(X)

    def _extract_pca_embeddings(self, X: np.ndarray) -> np.ndarray:
        """
        PCA ile verinin boyutunu embedding_dim'e indirger.
        Geleneksel ML modelleri için embedding olarak kullanılır.
        """

        n_samples, n_features = X.shape

        target_dim = min(self.embedding_dim, n_features, n_samples)

        if target_dim < 1:
            target_dim = 1

        pca = PCA(n_components=target_dim, random_state=42)
        embeddings = pca.fit_transform(X)

        return embeddings

    # ==========================================================
    # Properties
    # ==========================================================

    @property
    def is_deep_learning(self) -> bool:
        """Modelin derin öğrenme tabanlı olup olmadığını döndürür."""
        return self.registry_info.get("category") == "deep_learning"

    @property
    def model_category(self) -> str:
        """Modelin kategorisini döndürür."""
        return self.registry_info.get("category", "unknown")

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"ModelWrapper(name={self.model_name}, "
            f"category={self.model_category}, "
            f"status={status})"
        )
