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

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "ModelWrapper":
        """
        Modeli eğitir ve anomali skorlarını hesaplar.

        Args:
            X: (n_samples, n_features) ön işlemden geçmiş veri matrisi
            y: (Optional) Gözetimli modeller için etiketler
        """

        self._X = X

        # PyOD modelleri veya gözetimli modeller fit() ile eğitilir
        is_supervised = self.registry_info.get("is_supervised", False)
        
        import warnings
        with warnings.catch_warnings():
            # XGBOD supervised olmasına rağmen PyOD base.py yüzünden uyarı veriyor. Gizleyelim.
            warnings.filterwarnings("ignore", category=UserWarning)
            try:
                if is_supervised and y is not None:
                    self.model.fit(X, y)
                else:
                    self.model.fit(X)
            except TypeError:
                # Parametre uyuşmazlığı durumunda (eski versiyonlar) fallback
                self.model.fit(X)

        # Skorları al
        module_name = self.registry_info.get("module", "")
        is_sklearn_api = any(pkg in module_name for pkg in ["sklearn", "xgboost", "lightgbm"])
        
        if is_sklearn_api and hasattr(self.model, "predict_proba"):
            # Scikit-Learn / XGBoost / LightGBM Supervised Modeller
            # Sınıf 1 (Anomaly/Fraud) olasılığını skor olarak alıyoruz
            raw_scores = self.model.predict_proba(X)[:, 1]
        else:
            # PyOD Modelleri (Unsupervised veya XGBOD)
            raw_scores = self.model.decision_scores_

        # MinMaxScaler İPTAL: Ham skorlar doğrudan döndürülmeli ki Global Normalizasyon yapılabilsin.
        self._anomaly_scores = raw_scores

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

    def predict_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Eğitilmiş model ile yeni (görülmemiş) veri üzerinde anomali skorlarını tahmin eder.
        Cross Validation (Out-of-Fold) tahminleri için kullanılır.
        """
        if not self._is_fitted:
            raise RuntimeError("Model henüz eğitilmedi. Önce fit() çağırın.")
            
        module_name = self.registry_info.get("module", "")
        is_sklearn_api = any(pkg in module_name for pkg in ["sklearn", "xgboost", "lightgbm"])
        
        if is_sklearn_api and hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        elif hasattr(self.model, "decision_function"):
            return self.model.decision_function(X)
        else:
            raise NotImplementedError(f"{self.model_name} modeli predict_anomaly_scores desteklemiyor.")

    def predict_labels(self, X: np.ndarray) -> np.ndarray:
        """
        Eğitilmiş model ile yeni veri üzerinde kesin etiket (0 veya 1) tahmin eder.
        """
        if not self._is_fitted:
            raise RuntimeError("Model henüz eğitilmedi. Önce fit() çağırın.")
            
        if hasattr(self.model, "predict"):
            return self.model.predict(X).astype(int)
        else:
            scores = self.predict_anomaly_scores(X)
            return (scores >= 0.5).astype(int)

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
            # 1. Eğer modelin kendine has kaydedilmiş bir embedding matrisi varsa ve veriyle aynı boyuttaysa direkt al
            # (Özellikle LSTMAutoEncoder gibi sliding window yapan custom modeller için kritik)
            if hasattr(self.model, "embeddings_") and self.model.embeddings_ is not None:
                if len(self.model.embeddings_) == len(X):
                    return self.model.embeddings_
                else:
                    print(f"      [Debug] {self.model_name}: len(embeddings_)={len(self.model.embeddings_)}, len(X)={len(X)} uyuşmuyor.")
            else:
                print(f"      [Debug] {self.model_name}: modelin embeddings_ özniteliği yok veya None.")

            # 2. PyOD AutoEncoder / VAE modelleri
            # Yeni sürüm: 'model', eski sürüm: 'model_'
            inner_model = None
            if hasattr(self.model, "model") and self.model.model is not None:
                inner_model = self.model.model
            elif hasattr(self.model, "model_") and self.model.model_ is not None:
                inner_model = self.model.model_

            if inner_model is not None:
                try:
                    import torch

                    if hasattr(inner_model, "encoder"):
                        # PyTorch tabanlı modeller
                        inner_model.eval()
                        with torch.no_grad():
                            tensor_X = torch.FloatTensor(X)
                            raw_out = inner_model.encoder(tensor_X)
                            
                            # Eğer çıktı bir tuple ise (örn: LSTM'den dönen (output, (h, c))), sadece tensor olan kısmı al
                            if isinstance(raw_out, tuple):
                                raw_out = raw_out[0]
                                # Eğer hala çok boyutluysa (örn. (Batch, Seq, Features)), sadece son zaman adımını al
                                if raw_out.dim() == 3:
                                    raw_out = raw_out[:, -1, :]
                                    
                            # VAE encoder çıktısı [mu, logvar] birleşik olabilir
                            # mu kısmını al (ilk yarısı)
                            category = self.registry_info.get("category", "")
                            if "vae" in self.model_name.lower() or self.registry_info.get("class", "") == "VAE":
                                latent_dim = raw_out.shape[1] // 2
                                if latent_dim > 0:
                                    embeddings = raw_out[:, :latent_dim].cpu().numpy()
                                else:
                                    embeddings = raw_out.cpu().numpy()
                            else:
                                embeddings = raw_out.cpu().numpy()
                        return embeddings

                except ImportError:
                    pass

            # Fallback — PCA
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
