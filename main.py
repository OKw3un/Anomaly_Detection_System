import pandas as pd

from src.profiler.profiler import DataProfiler
from src.profiler.router import ModelRouter
from src.engine.executor import AnomalyEngine

def main():

    import os
    
    # =========================================================
    # 1. Load Dataset
    # =========================================================

    file_path = "data/timeseries/TimeSeries.csv"
    df = pd.read_csv(file_path)

    print(f"Dataset loaded: {df.shape}")

    # =========================================================
    # 2. Profile Dataset (Faz 1)
    # =========================================================

    profiler = DataProfiler(df, data_dir=os.path.dirname(file_path))

    meta_vector = profiler.profile()

    print("\n--- Meta-Feature Vector ---")

    print(
        meta_vector.to_dict()
    )

    # =========================================================
    # 3. Model Routing (Faz 1)
    # =========================================================

    router = ModelRouter()

    config = router.route(
        meta_vector
    )

    print("\n--- Model Routing Output ---")

    print(config)

    # =========================================================
    # 4. Dataset Summary (Faz 1)
    # =========================================================

    print("\n--- Dataset Summary ---")

    print(
        f"Dataset Type: "
        f"{meta_vector.dataset_type}"
    )

    print(
        f"Rows: "
        f"{meta_vector.total_rows}"
    )

    print(
        f"Columns: "
        f"{meta_vector.total_columns}"
    )

    print(
        f"Continuous Columns: "
        f"{len(meta_vector.continuous_cols)}"
    )

    print(
        f"Categorical Columns: "
        f"{len(meta_vector.categorical_cols)}"
    )

    print(
        f"ID Columns: "
        f"{meta_vector.id_cols}"
    )

    print(
        f"Label Columns: "
        f"{meta_vector.label_cols}"
    )

    print(
        f"Temporal Candidates: "
        f"{meta_vector.temporal_candidate_cols}"
    )

    print(
        f"High Dimensionality: "
        f"{meta_vector.high_dimensionality}"
    )

    print(
        f"Missing Ratio: "
        f"{meta_vector.missing_data_ratio:.4f}"
    )

    # =========================================================
    # 5. Anomaly Detection Engine (Faz 2)
    # =========================================================

    engine = AnomalyEngine()
    
    # Veri setinin türüne göre uygun modelleri belirleyelim (Router'ı eziyoruz)
    dataset_type = meta_vector.dataset_type
    
    if dataset_type in ["text", "log"]:
        suitable_models = ["isolation_forest", "lof", "pca", "ecod", "copod", "hbos", "ocsvm", "autoencoder", "vae", "deep_svdd"]
        print(f"  [Router Override] {dataset_type} veri seti tespit edildi. Sadece hazır Tabular modeller kullanılacak.")
    elif dataset_type == "time_series":
        # Feature Engineering eklendiği için artık tüm standart Tabular modeller (Özellikle Ağaç tabanlı Supervised modeller) zaman serisi gibi çalışabilir.
        # Ayrıca kendi yazdığımız özel (custom) supervised modeller de korundu.
        suitable_models = ["xgboost", "lightgbm", "random_forest", "isolation_forest", "ecod", "pca", "copod", "hbos", "lstm_autoencoder", "supervised_xgboost", "supervised_lightgbm"]
        print(f"  [Router Override] {dataset_type} veri seti tespit edildi. TS Feature Engineering ile zenginleştirilmiş standart modeller ve özel Supervised modeller kullanılacak.")
    else:
        # Hızlı sonuç alabilmek için geçici olarak DL modelleri ve OCSVM devre dışı bırakıldı.
        # Gözetimli (Supervised) modeller eklendi.
        suitable_models = ["isolation_forest", "ecod", "pca", "lof", "copod", "hbos", "random_forest", "lightgbm", "xgboost"]
        print(f"  [Router Override] {dataset_type} veri seti tespit edildi. İstatistiksel ve Supervised modeller çalıştırılacak.")
        
    config["recommended_models"] = suitable_models

    # Eğer Time Series ise etiketleri (y) oku
    y_labels = None
    if dataset_type == "time_series":
        try:
            labels_df = pd.read_csv("data/timeseries/labelsTimeSeries.csv")
            if "label" in labels_df.columns:
                y_labels = labels_df["label"].values
            elif labels_df.shape[1] == 1:
                y_labels = labels_df.iloc[:, 0].values
        except Exception as e:
            print(f"Etiketler okunamadı: {e}")
    else:
        # Tabular veri setlerinde varsa Class veya label sütununu y_labels olarak al
        if "Class" in df.columns:
            y_labels = df["Class"].values
        elif "label" in df.columns:
            y_labels = df["label"].values
        elif "income" in df.columns:
            # Adult dataset için özel etiket (">50K" anomali/pozitif sınıf kabul ediliyor)
            y_labels = (df["income"].astype(str).str.contains(">50K")).astype(int).values
            df = df.drop(columns=["income"]) # Eğitime sızmaması için sil
            print("  [OK] 'income' sütunu y_labels olarak ayrıldı ve eğitim setinden (df) çıkartıldı.")


    result = engine.run(df, config, y=y_labels)

    # =========================================================
    # 6. Sonuçların Kaydedilmesi
    # =========================================================

    result_df = result.to_dataframe()
    # print("  [OK] Faz 2 Sonuçları bellekte tutuluyor (CSV'ye kaydedilmedi).")
    
    # =========================================================
    # 7. Model Doğrulama ve XAI (Faz 3)
    # =========================================================
    
    from src.validation.validator import ResultValidator
    
    # Tüm datasetler için ayrı klasörleme mantığı
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.join("reports", dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"  [OK] Raporlar '{output_dir}' klasörüne yönlendirildi.")

    validator = ResultValidator(output_dir=output_dir)
    
    # Her bir modelin kendi skoru için dinamik Top-K İncelemesi
    model_names = [col.replace("score_", "") for col in result_df.columns if col.startswith("score_")]
    
    # Zaman serisi veri seti için gerçek etiketleri DataFrame'e ekle
    # (Validator'ın Top-K değerlendirmesinde kullanması için)
    if dataset_type == "time_series" and y_labels is not None:
        df["label"] = y_labels
        
    for model_name in model_names:
        validator.top_k_inspection(original_df=df, result_df=result_df, k=15, model_name=model_name)
    
    # 3. PCA ile Uzamsal Kontrol
    if result.X_transformed is not None:
        
        # Her bir modelin kendi kararları için dinamik Uzamsal Kontrol
        for model_name in model_names:
            validator.spatial_check_pca(result.X_transformed, result_df, model_name=model_name, dataset_type=dataset_type)
            
        # Gerçek etiketler (True Labels) ile Karşılaştırma Raporu (Sadece Time Series için)
        if dataset_type == "time_series":
            labels_path = "data/timeseries/labelsTimeSeries.csv"
            validator.evaluate_with_true_labels(result_df, labels_path, model_names)
            
        # Anomali Skorları Dağılım Grafiği (Elbow Metodu)
        if dataset_type != "tabular":
            validator.plot_score_distributions(result_df, model_names)
        
        # 4. SHAP Analizi (Isolation Forest varsa)
if __name__ == "__main__":
    main()
