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

        for col in working_df.columns:
            if pd.api.types.is_datetime64_any_dtype(working_df[col]):
                working_df[col] = (
                    working_df[col]
                    .astype(np.int64) // 10**9  # Unix timestamp (saniye)
                )

        # ---------------------------------------------------------
        # 3. Text/Log Sütunlarını Vektörleştir (TF-IDF)
        # ---------------------------------------------------------

        for col in text_cols:
            if col in working_df.columns:
                working_df[col] = working_df[col].fillna("")
                # max_features=15 ile boyut patlamasını engelle
                vectorizer = TfidfVectorizer(max_features=15)
                
                try:
                    tfidf_matrix = vectorizer.fit_transform(working_df[col]).toarray()
                    vocab = vectorizer.get_feature_names_out()
                    
                    # Yeni sütun isimleri oluştur
                    tfidf_cols = [f"{col}_tfidf_{w}" for w in vocab]
                    tfidf_df = pd.DataFrame(
                        tfidf_matrix, 
                        columns=tfidf_cols, 
                        index=working_df.index
                    )
                    
                    # Orijinal matrise ekle ve ham metin sütununu düşür
                    working_df = pd.concat([working_df, tfidf_df], axis=1)
                    working_df = working_df.drop(columns=[col])
                    
                except ValueError:
                    # Sütun tamamen boşsa vb. hataları atla
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
                    dummies = pd.get_dummies(working_df[col], prefix=col, dummy_na=False).astype(np.float32)
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
