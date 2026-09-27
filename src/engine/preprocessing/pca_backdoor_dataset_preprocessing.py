"""
Preprocessor — Faz 1 Router çıktısındaki preprocessing_steps sözlüğünü
alarak ham DataFrame'i modele beslenebilir sayısal matrise dönüştürür.

Desteklenen işlemler:
  - Eksik veri doldurma (imputation)
  - Kategorik kodlama (one-hot / ordinal)
  - Sürekli değişken ölçekleme (StandardScaler / RobustScaler)
  - ID ve Label sütunlarını ayırma
  - Mevsimsellik çıkarma ve trend arındırma
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer


class Preprocessor:
    """Faz 1 Router çıktısına göre dinamik ön işleme pipeline'ı."""

    def __init__(self):
        self.scaler: Optional[RobustScaler] = None
        self.encoder: Optional[OrdinalEncoder] = None
        self.imputer: Optional[SimpleImputer] = None

        # Fit sonrası saklanan sütun bilgileri
        self._feature_cols: List[str] = []
        self._continuous_cols: List[str] = []
        self._categorical_cols: List[str] = []

    # ==========================================================
    # Public API
    # ==========================================================

    def fit_transform(
        self,
        df: pd.DataFrame,
        preprocessing_steps: Dict[str, Any],
        meta: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, pd.Index]:
        """
        Ham DataFrame'i alır, ön işlemden geçirir ve sayısal matris döndürür.

        Returns:
            X:        (n_samples, n_features) numpy dizisi
            row_ids:  Orijinal satır indeksleri (sonuçları eşlemek için)
        """

        working_df = df.copy()
        row_ids = working_df.index.copy()

        # ---------------------------------------------------------
        # 1. ID ve Label sütunlarını ayır
        # ---------------------------------------------------------

        exclude_cols: List[str] = []

        id_cols = preprocessing_steps.get("exclude_id_columns", [])
        label_cols = preprocessing_steps.get("exclude_label_columns", [])
        text_cols = preprocessing_steps.get("text_columns", [])

        exclude_cols.extend(
            col for col in id_cols if col in working_df.columns
        )
        exclude_cols.extend(
            col for col in label_cols if col in working_df.columns
        )

        # Benzersizleştir
        exclude_cols = list(dict.fromkeys(exclude_cols))

        working_df = working_df.drop(
            columns=[c for c in exclude_cols if c in working_df.columns],
            errors="ignore"
        )

        # ---------------------------------------------------------
        # 2. Datetime sütunlarını sayısala dönüştür
        # ---------------------------------------------------------

        # Sadece Kaggle CreditCard veri setine özgü olacak şekilde,
        # Time saniye sayacını Günün Saati (Hour) değişkenine çevirelim.
        if "Time" in working_df.columns and "V1" in working_df.columns and "V28" in working_df.columns and pd.api.types.is_numeric_dtype(working_df["Time"]):
            print("  [Preprocessor] Creditcard veri seti algılandı. 'Time' sütunu 'Saat (Hour)' değişkenine dönüştürülüyor...")
            working_df["Hour"] = (working_df["Time"] // 3600) % 24

        for col in working_df.columns:
            if pd.api.types.is_datetime64_any_dtype(working_df[col]):
                working_df[col] = (
                    working_df[col]
                    .astype(np.int64) // 10**9  # Unix timestamp (saniye)
                )

        # ---------------------------------------------------------
        # 3. Text/Log Sütunlarını Vektörleştir (BERT / TF-IDF)
        # ---------------------------------------------------------
       
        bert_available = False
        if len(text_cols) > 0:
            try:
                from sentence_transformers import SentenceTransformer
                bert_available = True
                print("  [Preprocessor] sentence-transformers kütüphanesi bulundu! Metinler BERT ile (Semantik) vektörleştirilecek.")
                # Hızlı ve hafif bir model olan MiniLM kullanıyoruz
                bert_model = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                bert_available = False
                print("  [Preprocessor] UYARI: 'sentence-transformers' yüklü değil! TF-IDF (Kelime frekansı) yöntemine geri dönülüyor (Fallback).")
                print("  [Preprocessor] Çok daha akıllı sonuçlar için terminalden 'pip install sentence-transformers' komutunu çalıştırabilirsiniz.")

        for col in text_cols:
            if col in working_df.columns:
                working_df[col] = working_df[col].fillna("")
               
                if bert_available:
                    print(f"    > '{col}' sütunu BERT ile 384-boyutlu semantik vektöre çevriliyor...")
                    # Metinleri BERT embeddinglerine dönüştür (Shape: [N, 384])
                    embeddings = bert_model.encode(working_df[col].tolist(), show_progress_bar=False)
                   
                    bert_cols = [f"{col}_bert_{i}" for i in range(embeddings.shape[1])]
                    bert_df = pd.DataFrame(
                        embeddings,
                        columns=bert_cols,
                        index=working_df.index
                    )
                   
                    working_df = pd.concat([working_df, bert_df], axis=1)
                    working_df = working_df.drop(columns=[col])
                else:
                    # Eskisi gibi TF-IDF Fallback
                    # max_features=15 ile boyut patlamasını engelle
                    vectorizer = TfidfVectorizer(max_features=15)
                    try:
                        tfidf_matrix = vectorizer.fit_transform(working_df[col]).toarray()
                        vocab = vectorizer.get_feature_names_out()
                       
                        tfidf_cols = [f"{col}_tfidf_{w}" for w in vocab]
                        tfidf_df = pd.DataFrame(
                            tfidf_matrix,
                            columns=tfidf_cols,
                            index=working_df.index
                        )
                       
                        working_df = pd.concat([working_df, tfidf_df], axis=1)
                        working_df = working_df.drop(columns=[col])
                       
                    except ValueError:
                        working_df = working_df.drop(columns=[col])

        # ---------------------------------------------------------
        # 4. Sütun türlerini tespit et
        # ---------------------------------------------------------

        self._categorical_cols = [
            col for col in working_df.columns
            if (
                pd.api.types.is_object_dtype(working_df[col])
                or pd.api.types.is_string_dtype(working_df[col])
                or pd.api.types.is_categorical_dtype(working_df[col])
            )
        ]

        self._continuous_cols = [
            col for col in working_df.columns
            if col not in self._categorical_cols
            and pd.api.types.is_numeric_dtype(working_df[col])
        ]

        # ---------------------------------------------------------
        # 5. Eksik veri doldurma (Imputation)
        # ---------------------------------------------------------

        if preprocessing_steps.get("impute_missing", False):
            is_ts = meta.get("is_time_series", False) if meta else False
           
            if is_ts:
                # Zaman serisi için ffill ve bfill
                working_df = working_df.ffill().bfill()
            else:
                self.imputer = SimpleImputer(strategy="median")

                if self._continuous_cols:
                    working_df[self._continuous_cols] = (
                        self.imputer.fit_transform(
                            working_df[self._continuous_cols]
                        )
                    )

                # Kategorikler için en sık görülen değer
                for col in self._categorical_cols:
                    if working_df[col].isnull().any():
                        mode_val = working_df[col].mode()
                        if len(mode_val) > 0:
                            working_df[col] = working_df[col].fillna(
                                mode_val.iloc[0]
                            )

        # ---------------------------------------------------------
        # 6. Kategorik kodlama
        # ---------------------------------------------------------

        if (
            preprocessing_steps.get("encode_categorical", False)
            and self._categorical_cols
        ):
            # Adaptif Kategorik Kodlama (Kardinaliteye Göre)
            for col in self._categorical_cols:
                nunique = working_df[col].nunique()
               
                if nunique <= 5:
                    # Low Cardinality -> One-Hot Encoding
                    # drop_first=True eklenerek Dummy Variable Trap (Tam Çoklu Doğrusallık) önlendi
                    dummies = pd.get_dummies(working_df[col], prefix=col, dummy_na=False, drop_first=True).astype(np.float32)
                    working_df = pd.concat([working_df, dummies], axis=1)
                    working_df = working_df.drop(columns=[col])
                else:
                    # High Cardinality -> Frequency Encoding
                    freq = working_df[col].value_counts(normalize=True)
                    working_df[col] = working_df[col].map(freq).astype(np.float32)
                    working_df[col] = working_df[col].fillna(0.0)

        else:
            # Kategorik sütunlar encode edilmeyecekse düşür
            working_df = working_df.drop(
                columns=self._categorical_cols,
                errors="ignore"
            )

        # ---------------------------------------------------------
        # 7. Sürekli değişken ölçekleme
        # ---------------------------------------------------------

        # Artık tüm sütunlar sayısal olmalı.
        # ANCAK TF-IDF (_tfidf_) ve One-Hot (_ namespace_) gibi zaten normalize
        # edilmiş olan seyrek (sparse) sütunları RobustScaler'a SOKMAMALIYIZ!
        numeric_cols = [
            col for col in working_df.columns
            if pd.api.types.is_numeric_dtype(working_df[col])
            and "_tfidf_" not in col
            and (not col.startswith(tuple(c + "_" for c in self._categorical_cols)))
        ]

        if (
            preprocessing_steps.get("scale_continuous", False)
            and numeric_cols
        ):
            self.scaler = RobustScaler()

            working_df[numeric_cols] = (
                self.scaler.fit_transform(
                    working_df[numeric_cols]
                )
            )

        # ---------------------------------------------------------
        # 7.5. Sabit (Sıfır Varyanslı) Sütunları Çıkar (PCA Hatasını Önlemek İçin)
        # ---------------------------------------------------------
       
        numeric_df = working_df.select_dtypes(include=[np.number])
        # Standart sapması 0 olan (hiç değişmeyen) sütunlar PCA gibi modelleri çökertir
        zero_var_cols = numeric_df.columns[numeric_df.std(ddof=0) == 0].tolist()
       
        if zero_var_cols:
            print(f"  [Preprocessor] {len(zero_var_cols)} adet sıfır varyanslı sütun tespit edildi ve çıkarılıyor (PCA çökmesini önlemek için).")
            working_df = working_df.drop(columns=zero_var_cols)

        # ---------------------------------------------------------
        # 8. Sonuç
        # ---------------------------------------------------------

        self._feature_cols = list(working_df.columns)

        X = working_df.values.astype(np.float32)

        return X, row_ids

    # ==========================================================
    # Getter
    # ==========================================================

    @property
    def feature_names(self) -> List[str]:
        """Ön işlem sonrası kullanılan özellik isimlerini döndürür."""
        return self._feature_cols 
