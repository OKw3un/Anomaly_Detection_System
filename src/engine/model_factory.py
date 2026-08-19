"""
Model Factory — Faz 1 Router çıktısındaki recommended_models listesine göre
PyOD model nesnelerini yapılandırarak döndürür.

Her model adı (string) → PyOD sınıfı + varsayılan hiperparametreler eşlemesi
bu dosyada tanımlıdır. Bilinmeyen model adları atlanır.
"""

from typing import Dict, Any, List, Optional, Tuple


# =================================================================
# PyOD Model Registry
# =================================================================
# Her girdi:
#   model_key  →  (import_path, class_name, default_kwargs, supports_embedding)
#
# supports_embedding: True ise model bir darboğaz (bottleneck) katmanından
# embedding üretebilir. False ise embedding olarak ön işlemden geçmiş
# veri veya PCA kullanılır.
# =================================================================

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {

    # ---------------------------------------------------------
    # Geleneksel ML — Embedding üretmez
    # ---------------------------------------------------------

    "isolation_forest": {
        "module": "pyod.models.iforest",
        "class": "IForest",
        "kwargs": {
            "n_estimators": 200,
            "contamination": 0.05,
            "random_state": 42
        },
        "supports_embedding": False,
        "category": "traditional"
    },

    "ecod": {
        "module": "pyod.models.ecod",
        "class": "ECOD",
        "kwargs": {
            "contamination": 0.05
        },
        "supports_embedding": False,
        "category": "traditional"
    },

    "lof": {
        "module": "pyod.models.lof",
        "class": "LOF",
        "kwargs": {
            "n_neighbors": 20,
            "contamination": 0.05
        },
        "supports_embedding": False,
        "category": "traditional"
    },

    "one_class_svm": {
        "module": "pyod.models.ocsvm",
        "class": "OCSVM",
        "kwargs": {
            "contamination": 0.05
        },
        "supports_embedding": False,
        "category": "traditional"
    },

    # ---------------------------------------------------------
    # Derin Öğrenme — Embedding üretir
    # ---------------------------------------------------------

    "autoencoder": {
        "module": "pyod.models.auto_encoder",
        "class": "AutoEncoder",
        "kwargs": {
            "hidden_neurons": [64, 32, 16, 32, 64],
            "epochs": 50,
            "batch_size": 64,
            "contamination": 0.05,
            "preprocessing": False  # Biz zaten ön işlem yaptık
        },
        "supports_embedding": True,
        "category": "deep_learning",
        "bottleneck_index": 2  # hidden_neurons[2] = 16
    },

    "vae": {
        "module": "pyod.models.vae",
        "class": "VAE",
        "kwargs": {
            "encoder_neurons": [64, 32, 16],
            "decoder_neurons": [16, 32, 64],
            "epochs": 50,
            "batch_size": 64,
            "contamination": 0.05,
            "preprocessing": False
        },
        "supports_embedding": True,
        "category": "deep_learning",
        "bottleneck_index": -1  # Son encoder katmanı (latent_dim)
    },

    "deep_svdd": {
        "module": "pyod.models.deep_svdd",
        "class": "DeepSVDD",
        "kwargs": {
            "n_features": None,  # Çalışma zamanında ayarlanacak
            "hidden_neurons": [64, 32],
            "epochs": 50,
            "batch_size": 64,
            "contamination": 0.05,
            "preprocessing": False
        },
        "supports_embedding": True,
        "category": "deep_learning",
        "bottleneck_index": -1
    },

    # ---------------------------------------------------------
    # Yarı-gözetimli (Semi-supervised) sekanslar — gelecekte
    # ---------------------------------------------------------

    "deep_svdd_sequential": {
        "module": "pyod.models.deep_svdd",
        "class": "DeepSVDD",
        "kwargs": {
            "n_features": None,
            "hidden_neurons": [128, 64, 32],
            "epochs": 80,
            "batch_size": 64,
            "contamination": 0.05,
            "preprocessing": False
        },
        "supports_embedding": True,
        "category": "deep_learning",
        "bottleneck_index": -1
    },
}


class ModelFactory:
    """Router'ın önerdiği model adlarından PyOD nesneleri üretir."""

    def create_models(
        self,
        recommended_models: List[str],
        n_features: Optional[int] = None
    ) -> List[Tuple[str, Any, Dict[str, Any]]]:
        """
        Önerilen model adlarından PyOD nesnelerini oluşturur.

        Args:
            recommended_models: Router'dan gelen model adları listesi
            n_features: Girdi özellik sayısı (DL modelleri için gerekli)

        Returns:
            Liste: [(model_adı, model_nesnesi, registry_bilgisi), ...]
        """

        created = []

        for model_name in recommended_models:

            if model_name not in MODEL_REGISTRY:
                print(
                    f"[ModelFactory] Bilinmeyen model atlanıyor: "
                    f"{model_name}"
                )
                continue

            registry = MODEL_REGISTRY[model_name]

            try:
                # Dinamik import
                import importlib
                module = importlib.import_module(registry["module"])
                cls = getattr(module, registry["class"])

                # Kwargs kopyala
                kwargs = dict(registry["kwargs"])

                # n_features sadece DL modelleri için
                # (registry'de n_features: None olarak tanımlananlar)
                if (
                    "n_features" in kwargs
                    and kwargs["n_features"] is None
                    and n_features is not None
                ):
                    kwargs["n_features"] = n_features
                elif (
                    "n_features" in kwargs
                    and kwargs["n_features"] is None
                ):
                    # n_features bilinmiyorsa anahtarı kaldır
                    del kwargs["n_features"]

                # DL modelleri için hidden_neurons[0] > n_features
                # olmamalı kontrolü
                if "hidden_neurons" in kwargs and n_features is not None:
                    neurons = kwargs["hidden_neurons"]
                    if neurons[0] > n_features * 4:
                        # Ölçekle
                        scale = max(1, n_features // 2)
                        kwargs["hidden_neurons"] = [
                            max(scale, n // 2) for n in neurons
                        ]

                model = cls(**kwargs)

                created.append((model_name, model, registry))

                print(
                    f"[ModelFactory] Model oluşturuldu: {model_name} "
                    f"({registry['class']})"
                )

            except Exception as e:
                print(
                    f"[ModelFactory] Model oluşturulamadı: "
                    f"{model_name} — {e}"
                )

        return created

    @staticmethod
    def get_registry_info(model_name: str) -> Optional[Dict[str, Any]]:
        """Bir modelin registry bilgilerini döndürür."""
        return MODEL_REGISTRY.get(model_name)

    @staticmethod
    def list_available_models() -> List[str]:
        """Mevcut tüm model adlarını döndürür."""
        return list(MODEL_REGISTRY.keys())
