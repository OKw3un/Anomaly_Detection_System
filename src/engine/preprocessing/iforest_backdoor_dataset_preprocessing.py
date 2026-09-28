"""
IForest × UNSW-NB15 Backdoor Preprocessor
==========================================
Isolation Forest'a özel, UNSW-NB15 backdoor veri setine optimize edilmiş
ön işleme pipeline'ı.

Tasarım Prensipleri:
  - Ölçeklendirme YOK  → IForest ağaç tabanlıdır, scale-invariant'tır
  - One-Hot YOK        → Sparse binary sütunlar rastgele split'i zehirler
  - Log Transform VAR  → Çarpık dağılımları düzleştirip anlamlı split sağlar
  - Oran Özellikleri   → Korelasyon anomalilerini tek sütuna sıkıştırır
  - Düşük Varyans Filtresi → Neredeyse sabit sütunları temizler
  - Korelasyon Filtresi    → Tekrar eden bilgiyi eler
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer


class IForestBackdoorPreprocessor:
    """Isolation Forest + UNSW-NB15 Backdoor veri setine özel preprocessor."""

    def __init__(
        self,
        variance_threshold: float = 0.01,
        correlation_threshold: float = 0.85,
        skewness_threshold: float = 3.0
    ):
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.skewness_threshold = skewness_threshold

        self.imputer: Optional[SimpleImputer] = None
        self._feature_cols: List[str] = []
        self._dropped_corr_cols: List[str] = []

    # ==========================================================
    # Düzeltilmiş fit_transform Akışı
    # ==========================================================
    def fit_transform(
        self,
        df: pd.DataFrame,
        preprocessing_steps: Dict[str, Any],
        meta: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, pd.Index]:

        working_df = df.copy()
        row_ids = working_df.index.copy()

        # 1. Label ve ID sütunlarını ayır
        label_cols = preprocessing_steps.get("exclude_label_columns", [])
        id_cols = preprocessing_steps.get("exclude_id_columns", [])
        exclude_cols = list(dict.fromkeys(id_cols + label_cols))
        working_df = working_df.drop(columns=[c for c in exclude_cols if c in working_df.columns], errors="ignore")

        # 2. One-Hot sütunlarını frekans kodlamasına dönüştür
        working_df = self._collapse_onehot_groups(working_df)

        # 3. Kalan kategorik sütunları frekans kodla
        working_df = self._frequency_encode_categoricals(working_df)

        # 4. Eksik veri doldurma (Imputation)
        if preprocessing_steps.get("impute_missing", False):
            numeric_cols = working_df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                self.imputer = SimpleImputer(strategy="median")
                working_df[numeric_cols] = self.imputer.fit_transform(working_df[numeric_cols])
        working_df = working_df.fillna(0)

        # ---------------------------------------------------------
        # KRİTİK DEĞİŞİKLİK:
        # 5. ADIM: ÖNCE Feature Engineering (Ham değerler üzerinden)
        # ---------------------------------------------------------
        working_df = self._create_ratio_features(working_df)

        # ---------------------------------------------------------
        # 6. ADIM: SONRA Log Transform (Türetilen oranlar dahil)
        # ---------------------------------------------------------
        working_df = self._log_transform_skewed(working_df)

        # 7. Düşük Varyans Filtresi
        working_df = self._filter_low_variance(working_df)

        # 8. Korelasyon Filtresi (Türetilen özellikleri koruyarak)
        working_df = self._filter_high_correlation(working_df)

        # 9. Sonuç Matrisi
        self._feature_cols = list(working_df.columns)
        X = working_df.values.astype(np.float32)

        print(f"  [IForest Preprocessor] Final: {X.shape[0]} satır × {X.shape[1]} özellik")
        return X, row_ids

    # ==========================================================
    # Private — One-Hot Çözme
    # ==========================================================

    def _collapse_onehot_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        proto=*, service=*, state=* gibi one-hot gruplarını tespit eder.
        Her gruptan en sık kullanılan max_keep tanesini TUTAR,
        geri kalanları düşürür. Tutulanları da frekans kodlar.
        """
        onehot_prefixes = ["proto=", "service=", "state="]

        for prefix in onehot_prefixes:
            oh_cols = [c for c in df.columns if c.startswith(prefix)]

            if not oh_cols:
                continue

            # Orijinal kategorik sütunu geri oluştur
            category_name = prefix.rstrip("=")
            df[category_name] = df[oh_cols].idxmax(axis=1).str.replace(prefix, "", regex=False)

            # One-hot sütunlarını düşür
            df = df.drop(columns=oh_cols)

            # Frekans kodlama uygula
            freq = df[category_name].value_counts(normalize=True)
            df[category_name] = df[category_name].map(freq).astype(np.float32)

            print(f"  [IForest Preprocessor] '{prefix}*' ({len(oh_cols)} one-hot sütun) -> 1 frekans sütununa dönüştürüldü")

        return df

    # ==========================================================
    # Private — Kategorik Frekans Kodlama
    # ==========================================================

    def _frequency_encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Kalan tüm kategorik sütunları frekans kodlar."""
        cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()

        for col in cat_cols:
            freq = df[col].value_counts(normalize=True)
            df[col] = df[col].map(freq).astype(np.float32).fillna(0.0)
            print(f"  [IForest Preprocessor] '{col}' kategorik -> frekans kodlandı")

        return df

    # ==========================================================
    # Geliştirilmiş Oran ve Feature Engineering Fonksiyonu
    # ==========================================================
    def _create_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Backdoor trafiğine özel oran ve yoğunluk özellikleri türetir.
        """
        new_features = {}

        # 1. Trafik Asimetrisi (Backdoor C2 trafiği gidiş-dönüş asimetrisi gösterir)
        if "sbytes" in df.columns and "dbytes" in df.columns:
            new_features["ratio_bytes"] = (df["sbytes"] + 1) / (df["dbytes"] + 1)

        if "spkts" in df.columns and "dpkts" in df.columns:
            new_features["ratio_pkts"] = (df["spkts"] + 1) / (df["dpkts"] + 1)

        # Paket başına düşen yük
        if "sbytes" in df.columns and "spkts" in df.columns:
            new_features["bytes_per_spkt"] = df["sbytes"] / (df["spkts"] + 1)
        if "dbytes" in df.columns and "dpkts" in df.columns:
            new_features["bytes_per_dpkt"] = df["dbytes"] / (df["dpkts"] + 1)

        # 2. TTL ve TCP Manipülasyonları
        if "sttl" in df.columns and "dttl" in df.columns:
            new_features["ttl_diff"] = (df["sttl"] - df["dttl"]).abs()
            new_features["ttl_ratio"] = (df["sttl"] + 1) / (df["dttl"] + 1)

        if "swin" in df.columns and "dwin" in df.columns:
            new_features["win_ratio"] = (df["swin"] + 1) / (df["dwin"] + 1)

        # 3. BACKDOOR KRİTİK: Bağlantı Frekansı ve Yoğunluğu (UNSW-NB15 ct_* sütunları)
        # Backdoor hedefte sürekli aynı port/servis üzerinden bağlantı kurar
        if "ct_srv_src" in df.columns and "ct_dst_src_ltm" in df.columns:
            new_features["ratio_srv_dst"] = (df["ct_srv_src"] + 1) / (df["ct_dst_src_ltm"] + 1)

        if "ct_state_ttl" in df.columns and "ct_srv_src" in df.columns:
            new_features["ratio_state_srv"] = (df["ct_state_ttl"] + 1) / (df["ct_srv_src"] + 1)

        # Paket boyutu farkı
        if "smean" in df.columns and "dmean" in df.columns:
            new_features["mean_pkt_diff"] = (df["smean"] - df["dmean"]).abs()

        if new_features:
            new_df = pd.DataFrame(new_features, index=df.index)
            # Türetilen özellikleri tablonun başına alıyoruz ki korelasyon filtresi
            # ham özellikleri elesin, türettiğimiz zengin oranları korusun!
            df = pd.concat([new_df, df], axis=1)
            print(f"  [IForest Preprocessor] {len(new_features)} adet oran özelliği eklendi.")

        return df


    # ==========================================================
    # Geliştirilmiş Log Transform (Eşik 1.5'e çekildi)
    # ==========================================================
    def _log_transform_skewed(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Oranlar ve ham özellikler dahil sağa çarpık tüm sütunlara log1p uygular.
        Eşik 3.0 yerine 1.5 tutularak manifoldun merkezde sıkışması önlenir.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        transformed = []

        # skewness_threshold değerini 1.5 olarak kullanmak ağ trafiğinde daha dengeli dağılım sağlar
        effective_threshold = min(self.skewness_threshold, 1.5)

        for col in numeric_cols:
            # 0/1 binary veya one-hot sütunları atla
            if df[col].nunique() <= 2:
                continue

            if (df[col] >= 0).all():
                skew_val = float(df[col].skew())
                if skew_val > effective_threshold:
                    df[col] = np.log1p(df[col])
                    transformed.append(col)

        if transformed:
            print(f"  [IForest Preprocessor] Log1p uygulanan özellik sayısı: {len(transformed)}")

        return df

    # ==========================================================
    # Private — Düşük Varyans Filtresi
    # ==========================================================

    def _filter_low_variance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Varyansı çok düşük olan sütunları filtreler.
        proto=argus gibi neredeyse tamamen 0 olan sütunlar
        IForest'ın rastgele dal seçimini zehirler.
        """
        before_count = df.shape[1]

        selector = VarianceThreshold(threshold=self.variance_threshold)
        X_filtered = selector.fit_transform(df.values)

        kept_cols = df.columns[selector.get_support()].tolist()
        dropped_cols = df.columns[~selector.get_support()].tolist()

        df = pd.DataFrame(X_filtered, columns=kept_cols, index=df.index)

        dropped_count = before_count - df.shape[1]
        if dropped_count > 0:
            print(f"  [IForest Preprocessor] Düşük varyans filtresi: {dropped_count} sütun çıkarıldı (varyans < {self.variance_threshold})")

        return df

    # ==========================================================
    # Private — Korelasyon Filtresi
    # ==========================================================

    def _filter_high_correlation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pearson korelasyonu > threshold olan sütun çiftlerinden birini düşürür.
        IForest rastgele sütun seçtiği için aynı bilgiyi taşıyan
        3 farklı sütun olması, o bilgiye haksız ağırlık verir.
        """
        corr_matrix = df.corr().abs()

        # Üst üçgen matris (kendisiyle ve alt üçgenle tekrarı önle)
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Eşikten yüksek korelasyona sahip sütunları bul
        to_drop = [
            col for col in upper_tri.columns
            if any(upper_tri[col] > self.correlation_threshold)
        ]

        self._dropped_corr_cols = to_drop

        if to_drop:
            df = df.drop(columns=to_drop)
            print(f"  [IForest Preprocessor] Korelasyon filtresi: {len(to_drop)} sütun çıkarıldı (Pearson > {self.correlation_threshold}): {to_drop[:10]}{'...' if len(to_drop) > 10 else ''}")

        return df

    # ==========================================================
    # Getter
    # ==========================================================

    @property
    def feature_names(self) -> List[str]:
        """Ön işlem sonrası kullanılan özellik isimlerini döndürür."""
        return self._feature_cols
