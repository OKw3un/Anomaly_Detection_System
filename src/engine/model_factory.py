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
    "pca": {
        "module": "pyod.models.pca",
        "class": "PCA",
        "kwargs": {
            "contamination": 0.05,
            "n_components": 15  # 0-Varyans/Sıfıra bölünme hatasını önlemek için boyut sınırlandırıldı
        },
        "supports_embedding": False,
        "category": "traditional"
    },

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
            "n_neighbors": 100,
            "contamination": 0.05
        },
        "supports_embedding": False,
        "category": "traditional"
    },

    "ocsvm": {
        "module": "pyod.models.ocsvm",
        "class": "OCSVM",
        "kwargs": {
            "contamination": 0.05,
            "kernel": "rbf",
            "gamma": 0.005
        },
        "supports_embedding": False,
        "category": "traditional"
    },

    "copod": {
        "module": "pyod.models.copod",
        "class": "COPOD",
        "kwargs": {"contamination": 0.05},
        "supports_embedding": False,
        "category": "traditional"
    },
    "hbos": {
        "module": "pyod.models.hbos",
        "class": "HBOS",
        "kwargs": {"contamination": 0.05, "n_bins": 20},
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
            "hidden_neuron_list": [64, 32, 16, 32, 64],
            "epoch_num": 50,
            "batch_size": 64,
            "contamination": 0.05,
            "preprocessing": False  # Biz zaten ön işlem yaptık
        },
        "supports_embedding": True,
        "category": "deep_learning",
        "bottleneck_index": 2  # hidden_neuron_list[2] = 16
    },

    "vae": {
        "module": "pyod.models.vae",
        "class": "VAE",
        "kwargs": {
            "encoder_neuron_list": [64, 32, 16],
            "decoder_neuron_list": [16, 32, 64],
            "epoch_num": 50,
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
    # Supervised / Semi-supervised Modeller (YENİ EKLENDİ)
    # ---------------------------------------------------------

    "random_forest": {
        "module": "sklearn.ensemble",
        "class": "RandomForestClassifier",
        "kwargs": {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "random_state": 42,
            "n_jobs": -1
        },
        "supports_embedding": False,
        "category": "supervised",
        "is_supervised": True,
        "requires_y": True
    },

    "xgbod": {
        "module": "pyod.models.xgbod",
        "class": "XGBOD",
        "kwargs": {
            "n_jobs": -1,
            "random_state": 42
        },
        "supports_embedding": False,
        "category": "semi_supervised",
        "is_supervised": True,
        "requires_y": True
    },

    "lightgbm": {
        "module": "lightgbm",
        "class": "LGBMClassifier",
        "kwargs": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "num_leaves": 31,
            "max_depth": 5,
            "random_state": 42,
            "n_jobs": -1
        },
        "supports_embedding": False,
        "category": "supervised",
        "is_supervised": True,
        "requires_y": True
    },

    "xgboost": {
        "module": "xgboost",
        "class": "XGBClassifier",
        "kwargs": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 5,
            "random_state": 42,
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "n_jobs": -1
        },
        "supports_embedding": False,
        "category": "supervised",
        "is_supervised": True,
        "requires_y": True
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

    # ---------------------------------------------------------
    # Zaman Serisi Modelleri
    # ---------------------------------------------------------
    "lstm_autoencoder": {
        "module": "src.engine.custom_models",
        "class": "LSTMAutoEncoder",
        "kwargs": {
            "seq_len": 21,
            "hidden_dim": 32,
            "epochs": 25,
            "batch_size": 256,
            "contamination": 0.00087,
            "invert_score": True
        },
        "supports_embedding": True,
        "category": "time_series"
    },


    "supervised_xgboost": {
        "module": "src.engine.custom_models",
        "class": "SupervisedXGBoost",
        "kwargs": {
            "seq_len": 21,
            "contamination": 0.00087
        },
        "supports_embedding": False,
        "category": "time_series",
        "is_supervised": True,
        "requires_y": True
    },

    "supervised_lightgbm": {
        "module": "src.engine.custom_models",
        "class": "SupervisedLightGBM",
        "kwargs": {
            "seq_len": 21,
            "contamination": 0.00087,
            "is_unbalance": True,
            "min_child_samples": 5
        },
        "supports_embedding": False,
        "category": "time_series",
        "is_supervised": True,
        "requires_y": True
    },

    # ---------------------------------------------------------
    # Grafik / Ağ (Graph/Network) Modelleri
    # ---------------------------------------------------------

    "graph_dominant": {
        "module": "src.engine.custom_models",
        "class": "GraphDominantDetector",
        "kwargs": {
            "hid_dim": 32,
            "num_layers": 3,
            "epoch": 40,
            "weight": 0.5,
            "k_neighbors": 5,
            "contamination": 0.05
        },
        "supports_embedding": True,
        "category": "graph"
    }
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

                # DL modellerinde aşırı öğrenmeyi engellemek için ağ ölçeklendirmesi (Bottleneck)
                if n_features is not None:
                    neuron_keys = ["hidden_neuron_list", "hidden_neurons", "encoder_neuron_list", "decoder_neuron_list"]
                    for key in neuron_keys:
                        if key in kwargs:
                            neurons = kwargs[key]
                            # Eğer girdi boyutu 32'den küçükse (Örn: 16 Dengeli Uzay), 
                            # ağı kesinlikle daralt ki ezberlemesin!
                            if n_features <= 32:
                                # Ağın ilk katmanı genelde genişletir, ama girdi çok küçükse 
                                # en büyük katmanı minimum 16 veya girdi boyutu kadar koru
                                scale = max(16, n_features)
                                ratio = scale / float(max(neurons[0], 1))
                                
                                new_neurons = []
                                for n in neurons:
                                    # Ağın tamamen sıfırlanmasını (collapse) önlemek için minimum 4 nöron bırak
                                    scaled_n = max(4, int(n * ratio))
                                    new_neurons.append(scaled_n)
                                kwargs[key] = new_neurons
                                print(f"    [ModelFactory] {model_name} {key} darboğazı aktifleştirildi: {new_neurons}")
                            elif neurons[0] > n_features * 4:
                                # Eski ölçeklendirme kuralı (Çok büyük girdi vs aşırı büyük nöron)
                                scale = max(1, n_features // 2)
                                kwargs[key] = [max(scale, n // 2) for n in neurons]
                                
                # PCA Modeli için dinamik n_components koruması
                if model_name == "pca" and n_features is not None:
                    if "n_components" in kwargs and kwargs["n_components"] >= n_features:
                        # n_components özelliği, mevcut özellik sayısından küçük olmalıdır
                        kwargs["n_components"] = max(1, n_features - 1)
                        print(f"    [ModelFactory] pca n_components darboğazı aktifleştirildi: {kwargs['n_components']}")

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
