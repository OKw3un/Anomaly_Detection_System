"""
PCA × UNSW-NB15 Backdoor Preprocessor
======================================
PCA (Principal Component Analysis) tabanlı anomali tespiti için özel,
UNSW-NB15 Backdoor veri setine optimize edilmiş ön işleme (preprocessing) modülü.

Tasarım Prensipleri & Uygulanan Kurallar:
  1. Tamamen Unsupervised:
     - ID, Label, Class sütunları pipeline'dan tamamen çıkarılır.
     - Label veya Class bilgisi içeren hiçbir istatistik (KS-testi vb.)
       feature seçiminde veya türetiminde kullanılmaz.
  2. Kategorik Bilgi Korunur:
     - proto, service, state sütunları one-hot kodlama ile korunur.
     - Asla idxmax / frequency encoding ile tek bir sütuna indirgenmez.
     - Mevcut one-hot grupları doğrudan korunur.
  3. Log1p Dönüşümü:
     - Yalnızca pozitif ve belirgin sağa çarpık (skew > skewness_threshold)
       sayısal özelliklere uygulanır.
     - Binary (one-hot) sütunlar log dönüşümüne sokulmaz.
  4. Eksik Değer Doldurma (Imputation):
     - Sayısal değişkenler -> Median (Medyan)
     - Kategorik değişkenler -> Mode (Mod)
     - Zaman serisi özel durumu -> ffill().bfill()
  5. Ölçeklendirme (Scaling):
     - RobustScaler (varsayılan) ve StandardScaler desteği.
     - PCA her zaman scaling sonrası uygulanır.
  6. Sabit Sütun Temizliği:
     - Körlemesine VarianceThreshold(0.001) UYGULANMAZ (nadir anomali sinyallerini silmemek için).
     - Yalnızca standart sapması kesin olarak 0 olan (tamamen sabit) sütunlar elenir.
  7. Kademeli Test ve Ablation Desteği:
     - use_log_transform, scaler_type, scale_onehot parametreleri ile
       adım adım karşılaştırma olanağı sunar.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer


class PCABackdoorPreprocessor:
    """UNSW-NB15 Backdoor veri seti ve PCA anomali tespit modeline özel preprocessor."""

    # Sıkça karşılaşılan ID ve Label sütun adları (Unsupervised garanti)
    DEFAULT_EXCLUDE_CANDIDATES = {
        "id", "ID", "Id", "label", "Label", "class", "Class",
        "attack_cat", "Attack_cat", "uuid", "guid", "index"
    }

    def __init__(
        self,
        scaler_type: str = "robust",            # "standard" | "robust" | "none"
        use_log_transform: bool = True,        # Sağa çarpık değişkenlere log1p uygula
        skewness_threshold: float = 2.0,       # log1p için skew eşiği (skew > 2)
        scale_onehot: bool = False,            # One-hot sütunları da ölçekle mi (False: 0/1 kalır)
        only_numeric_features: bool = False,   # True ise one-hot/kategorik sütunlar elenir, sadece continuous sayısal sütunlar verilir
        **kwargs
    ):
        self.scaler_type = str(scaler_type).lower()
        self.use_log_transform = use_log_transform
        self.skewness_threshold = skewness_threshold
        self.scale_onehot = scale_onehot
        self.only_numeric_features = kwargs.get("numeric_only", only_numeric_features)

        self.scaler: Optional[Any] = None
        self.imputer: Optional[SimpleImputer] = None

        self._feature_cols: List[str] = []
        self._dropped_zero_var_cols: List[str] = []
        self._log_transformed_cols: List[str] = []
        self._scaled_cols: List[str] = []

    # ==========================================================
    # Public API
    # ==========================================================

    def fit_transform(
        self,
        df: pd.DataFrame,
        preprocessing_steps: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, pd.Index]:
        """
        Ham DataFrame'i alır, PCA ve Backdoor veri setine özel
        unsupervised pipeline'dan geçirir ve sayısal matris döndürür.

        Returns:
            X:        (n_samples, n_features) numpy float32 dizisi
            row_ids:  Orijinal satır indeksleri
        """
        if preprocessing_steps is None:
            preprocessing_steps = {}

        working_df = df.copy()
        row_ids = working_df.index.copy()

        # ---------------------------------------------------------
        # 1. ID ve Label sütunlarını çıkar (Tamamen Unsupervised)
        # ---------------------------------------------------------
        working_df = self._drop_id_and_label_columns(working_df, preprocessing_steps)

        # ---------------------------------------------------------
        # 2. Eksik Veri Doldurma (Imputation)
        # ---------------------------------------------------------
        working_df = self._impute_missing(working_df, meta)

        # ---------------------------------------------------------
        # 3. Kategorik Değişkenleri Koru (One-Hot Encoding)
        #    (Frekans kodlamaya indirgenmez, kategori kimliği korunur)
        # ---------------------------------------------------------
        working_df = self._preserve_and_encode_categoricals(working_df)

        # ---------------------------------------------------------
        # 4. Log1p Dönüşümü
        #    Yalnızca pozitif ve sağa çarpık (skew > 2) numeric sütunlara
        # ---------------------------------------------------------
        if self.use_log_transform:
            working_df = self._apply_log1p_transform(working_df)

        # ---------------------------------------------------------
        # 5. Sıfır Varyanslı (Tamamen Sabit) Sütunları Çıkar
        #    VarianceThreshold(0.001) körlemesine uygulanmaz!
        #    Yalnızca sayısal kararlılık için std == 0 olanlar çıkarılır.
        # ---------------------------------------------------------
        working_df = self._drop_zero_variance_columns(working_df)

        # ---------------------------------------------------------
        # 6. Ölçekleme (RobustScaler veya StandardScaler)
        #    PCA scaling sonrası uygulanır.
        # ---------------------------------------------------------
        working_df = self._apply_scaling(working_df)

        # ---------------------------------------------------------
        # 7. Sonuç Matrisi Hazırlığı
        # ---------------------------------------------------------
        self._feature_cols = list(working_df.columns)
        X = working_df.values.astype(np.float32)

        print(
            f"  [PCABackdoorPreprocessor] Tamamlandı: "
            f"{X.shape[0]} satır × {X.shape[1]} özellik "
            f"(Scaler: {self.scaler_type}, Log: {self.use_log_transform}, ScaleOneHot: {self.scale_onehot}, OnlyNumeric: {self.only_numeric_features})"
        )

        return X, row_ids

    # ==========================================================
    # Private — Adımlar
    # ==========================================================

    def _drop_id_and_label_columns(
        self,
        df: pd.DataFrame,
        preprocessing_steps: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        ID, Label ve Class sütunlarını tamamen ayıklar.
        Modelin unsupervised kalmasını sağlar, kesinlikle PCA'ya sızdırılmaz.
        """
        exclude_cols = set()

        for col in preprocessing_steps.get("exclude_id_columns", []):
            if col in df.columns:
                exclude_cols.add(col)
        for col in preprocessing_steps.get("exclude_label_columns", []):
            if col in df.columns:
                exclude_cols.add(col)

        for col in df.columns:
            if col.lower() in {c.lower() for c in self.DEFAULT_EXCLUDE_CANDIDATES}:
                exclude_cols.add(col)

        if exclude_cols:
            print(f"  [PCABackdoorPreprocessor] ID/Label sütunları çıkarıldı: {sorted(list(exclude_cols))}")
            df = df.drop(columns=list(exclude_cols), errors="ignore")

        return df

    def _impute_missing(
        self,
        df: pd.DataFrame,
        meta: Optional[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Eksik değerleri doldurur:
          - Zaman serisi varsa: ffill().bfill()
          - Sayısal değişkenler: Median
          - Kategorik değişkenler: Mode
        """
        is_ts = meta.get("is_time_series", False) if meta else False

        if is_ts:
            df = df.ffill().bfill()
            return df

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols and df[numeric_cols].isnull().any().any():
            self.imputer = SimpleImputer(strategy="median")
            df[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])

        cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        for col in cat_cols:
            if df[col].isnull().any():
                mode_series = df[col].mode(dropna=True)
                mode_val = mode_series.iloc[0] if len(mode_series) > 0 else "UNKNOWN"
                df[col] = df[col].fillna(mode_val)

        if df.isnull().any().any():
            df = df.fillna(0)

        return df

    def _preserve_and_encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        only_numeric_features=True ise tüm kategorik ve one-hot sütunları eler,
        PCA'ya yalnızca continuous sayısal değişkenlerin gitmesini sağlar.

        only_numeric_features=False ise proto, service, state gibi kategorik
        değişkenlerin kimliğini korur (One-Hot Encoding).
        """
        if self.only_numeric_features:
            cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
            onehot_cols = [
                c for c in df.columns
                if any(c.startswith(p) for p in ["proto=", "service=", "state="])
            ]
            drop_cols = list(set(cat_cols + onehot_cols))
            if drop_cols:
                print(
                    f"  [PCABackdoorPreprocessor] only_numeric_features=True: "
                    f"{len(drop_cols)} adet kategorik/one-hot sütun çıkarıldı."
                )
                df = df.drop(columns=drop_cols, errors="ignore")
            return df

        cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()

        if cat_cols:
            print(f"  [PCABackdoorPreprocessor] Kategorik sütunlar One-Hot kodlanıyor: {cat_cols}")
            dummies = pd.get_dummies(df[cat_cols], prefix=cat_cols, drop_first=False, dtype=np.float32)
            df = df.drop(columns=cat_cols)
            df = pd.concat([df, dummies], axis=1)

        return df

    def _apply_log1p_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Yalnızca pozitif, ikili olmayan (nunique > 2) ve belirgin sağa çarpık
        (skew > skewness_threshold) sayısal değişkenlere np.log1p uygular.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        transformed = []

        for col in numeric_cols:
            if df[col].nunique() <= 2:
                continue

            if (df[col] >= 0).all():
                skew_val = float(df[col].skew())
                if skew_val > self.skewness_threshold:
                    df[col] = np.log1p(df[col].astype(np.float32))
                    transformed.append(col)

        self._log_transformed_cols = transformed
        if transformed:
            sample_shown = transformed[:6]
            more_str = f" (+{len(transformed) - 6} daha)" if len(transformed) > 6 else ""
            print(
                f"  [PCABackdoorPreprocessor] Log1p uygulandı ({len(transformed)} sütun, skew > {self.skewness_threshold}): "
                f"{sample_shown}{more_str}"
            )

        return df

    def _drop_zero_variance_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Körlemesine VarianceThreshold(0.001) UYGULAMAZ.
        Sadece sayısal olarak hiçbir değişimi olmayan (std == 0) sütunları çıkarır.
        Nadir anomali göstergesi olabilecek seyrek sütunlar korunur.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        zero_var_cols = [c for c in numeric_cols if df[c].std(ddof=0) == 0.0]

        self._dropped_zero_var_cols = zero_var_cols
        if zero_var_cols:
            print(
                f"  [PCABackdoorPreprocessor] {len(zero_var_cols)} adet sıfır varyanslı (sabit) sütun çıkarıldı: "
                f"{zero_var_cols}"
            )
            df = df.drop(columns=zero_var_cols)

        return df

    def _apply_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ölçeklendirme uygular:
          - scaler_type == 'standard': StandardScaler
          - scaler_type == 'robust': RobustScaler
          - scaler_type == 'none': Ölçekleme yapılmaz

        scale_onehot=False ise one-hot/ikili sütunlar 0/1 olarak korunur,
        yalnızca sürekli sayısal değişkenler ölçeklenir.
        """
        if self.scaler_type == "none":
            return df

        if self.scale_onehot or self.only_numeric_features:
            cols_to_scale = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            cols_to_scale = [
                c for c in df.select_dtypes(include=[np.number]).columns
                if df[c].nunique() > 2
                and not any(c.startswith(p) for p in ["proto=", "service=", "state="])
            ]

        self._scaled_cols = cols_to_scale

        if cols_to_scale:
            if self.scaler_type == "robust":
                self.scaler = RobustScaler()
                print(f"  [PCABackdoorPreprocessor] RobustScaler uygulanıyor ({len(cols_to_scale)} sütun)...")
            else:
                self.scaler = StandardScaler()
                print(f"  [PCABackdoorPreprocessor] StandardScaler uygulanıyor ({len(cols_to_scale)} sütun)...")

            df[cols_to_scale] = self.scaler.fit_transform(df[cols_to_scale]).astype(np.float32)

        return df

    # ==========================================================
    # Getters
    # ==========================================================

    @property
    def feature_names(self) -> List[str]:
        """Ön işlem sonrası kullanılan özellik isimlerini döndürür."""
        return self._feature_cols


# Geriye dönük uyumluluk için alias
Preprocessor = PCABackdoorPreprocessor
