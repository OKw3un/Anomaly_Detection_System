import re
import pandas as pd
import numpy as np

try:
    from statsmodels.tsa.stattools import adfuller, acf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

from src.profiler.meta_feature import MetaFeatureVector


class DataProfiler:

    def __init__(
        self,
        df: pd.DataFrame,
        categorical_threshold: float = 0.05,
        high_cardinality_threshold: int = 20,
        text_length_threshold: int = 30,
        data_dir: str = None,
        max_sample_size: int = 15000,
        dataset_name: str = None
    ):
        self.dataset_name = dataset_name
        self.original_rows = df.shape[0]
        self.full_missing_ratio = float(df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) if df.shape[0] > 0 and df.shape[1] > 0 else 0.0
        if max_sample_size is not None and self.original_rows > max_sample_size:
            # Zaman serisi bütünlüğünü bozmamak için rastgele değil, sıralı kesit (tail) alıyoruz
            self.df = df.tail(max_sample_size).copy()
        else:
            self.df = df
            
        self.cat_threshold = categorical_threshold
        self.high_cardinality_threshold = high_cardinality_threshold
        self.text_length_threshold = text_length_threshold
        self.data_dir = data_dir
        self.max_sample_size = max_sample_size

    # =========================================================
    # Public API
    # =========================================================

    def profile(self) -> MetaFeatureVector:

        total_rows, total_cols = self.df.shape

        if total_rows == 0 or total_cols == 0:
            raise ValueError("Dataset is empty.")

        # -----------------------------------------------------
        # Basic statistics
        # -----------------------------------------------------

        missing_ratio = self.full_missing_ratio

        feature_to_sample_ratio = (
            total_cols / self.original_rows
        )

        # -----------------------------------------------------
        # Column groups
        # -----------------------------------------------------

        timestamp_cols = []
        temporal_candidate_cols = []

        continuous_cols = []
        categorical_cols = []
        text_cols = []
        id_cols = []
        label_cols = []

        cols_with_missing_values = []

        column_cardinality = {}
        column_cardinality_ratio = {}
        high_cardinality_cols = []

        # -----------------------------------------------------
        # First: Label detection
        # -----------------------------------------------------
        label_cols = self._detect_label_columns()

        # -----------------------------------------------------
        # Column profiling
        # -----------------------------------------------------

        for col in self.df.columns:

            series = self.df[col]

            unique_count = series.nunique(dropna=True)

            unique_ratio = (
                unique_count / total_rows
                if total_rows > 0
                else 0
            )

            column_cardinality[col] = unique_count
            column_cardinality_ratio[col] = unique_ratio

            # -------------------------------------------------
            # Missing values
            # -------------------------------------------------

            if series.isnull().any():
                cols_with_missing_values.append(col)

            # -------------------------------------------------
            # Label columns
            # -------------------------------------------------

            if col in label_cols:
                continue

            # -------------------------------------------------
            # Timestamp detection
            # -------------------------------------------------

            if pd.api.types.is_datetime64_any_dtype(series):

                timestamp_cols.append(col)
                temporal_candidate_cols.append(col)

                continue

            # -------------------------------------------------
            # Object / String columns
            # -------------------------------------------------

            if (
                pd.api.types.is_object_dtype(series)
                or pd.api.types.is_string_dtype(series)
            ):

                # ---------------------------------------------
                # Try datetime detection
                # ---------------------------------------------

                non_null = series.dropna()

                if len(non_null) > 0:

                    sample = non_null.head(100)

                    converted = pd.to_datetime(
                        sample,
                        format="mixed",
                        errors="coerce"
                    )

                    valid_ratio = (
                        converted.notna().sum()
                        / len(sample)
                    )

                    if valid_ratio >= 0.5:

                        timestamp_cols.append(col)
                        temporal_candidate_cols.append(col)

                        continue

                # ---------------------------------------------
                # ID detection for string columns
                # ---------------------------------------------
                # ID detection must come BEFORE text detection
                # because UUIDs are long and can be misclassified as text.

                if self._looks_like_id(
                    col,
                    unique_ratio,
                    series
                ):

                    id_cols.append(col)

                    continue

                # ---------------------------------------------
                # Text detection
                # ---------------------------------------------

                avg_text_length = (
                    non_null.astype(str)
                    .str.len()
                    .mean()
                    if len(non_null) > 0
                    else 0
                )

                if (
                    avg_text_length >= self.text_length_threshold
                    or self._looks_like_text(col, series)
                ):

                    text_cols.append(col)

                    continue

                # ---------------------------------------------
                # Otherwise categorical
                # ---------------------------------------------

                categorical_cols.append(col)

                continue

            # -------------------------------------------------
            # Numeric columns
            # -------------------------------------------------

            if pd.api.types.is_numeric_dtype(series):

                # ---------------------------------------------
                # Numeric ID detection
                # ---------------------------------------------

                if self._looks_like_id(
                    col,
                    unique_ratio,
                    series
                ):

                    id_cols.append(col)

                # ---------------------------------------------
                # Numeric temporal candidate
                # ---------------------------------------------

                elif self._looks_temporal(col, series):

                    temporal_candidate_cols.append(col)

                    # Numeric time values are still features.
                    # They are NOT automatically timestamp_cols.

                    continuous_cols.append(col)

                # ---------------------------------------------
                # Categorical vs continuous numeric
                # ---------------------------------------------

                elif self._is_numeric_categorical(
                    series,
                    unique_count,
                    total_rows
                ):

                    categorical_cols.append(col)

                # ---------------------------------------------
                # Continuous numeric column
                # ---------------------------------------------

                else:

                    continuous_cols.append(col)

                continue

            # -------------------------------------------------
            # Fallback
            # -------------------------------------------------

            categorical_cols.append(col)

        # -----------------------------------------------------
        # High cardinality
        # -----------------------------------------------------

        for col in self.df.columns:

            if col in label_cols:
                continue

            if col in id_cols:
                continue

            if col in text_cols:
                continue

            cardinality = column_cardinality[col]

            if cardinality > self.high_cardinality_threshold:

                high_cardinality_cols.append(col)

        # -----------------------------------------------------
        # Dataset type
        # -----------------------------------------------------

        dataset_type = self._detect_dataset_type(
            timestamp_cols=timestamp_cols,
            temporal_candidate_cols=temporal_candidate_cols,
            text_cols=text_cols,
            continuous_cols=continuous_cols
        )

        # -----------------------------------------------------
        # Ratios
        # -----------------------------------------------------

        numeric_ratio = (
            len(continuous_cols) / total_cols
        )

        categorical_ratio = (
            len(categorical_cols) / total_cols
        )

        text_ratio = (
            len(text_cols) / total_cols
        )

        id_ratio = (
            len(id_cols) / total_cols
        )

        # -----------------------------------------------------
        # Dimensionality
        # -----------------------------------------------------

        high_dimensionality = self._is_high_dimensional(
            total_rows,
            total_cols
        )

        # -----------------------------------------------------
        # Temporal information
        # -----------------------------------------------------

        is_time_series = self._is_time_series(
            temporal_candidate_cols,
            timestamp_cols,
            continuous_cols
        )

        temporal_regularity = self._temporal_regularity(
            temporal_candidate_cols
        )

        # -----------------------------------------------------
        # Entity information
        # -----------------------------------------------------

        observations_per_entity = (
            self._observations_per_entity(id_cols)
        )

        # -----------------------------------------------------
        # Temporal min/max
        # -----------------------------------------------------

        temporal_min = {}
        temporal_max = {}

        for col in temporal_candidate_cols:

            _series = self.df[col].dropna()

            if len(_series) == 0:
                continue

            try:

                if (
                    pd.api.types.is_object_dtype(_series)
                    or pd.api.types.is_string_dtype(_series)
                ):
                    _series = pd.to_datetime(
                        _series,
                        format="mixed",
                        errors="coerce"
                    ).dropna()

                if len(_series) > 0:
                    temporal_min[col] = str(
                        _series.min()
                    )
                    temporal_max[col] = str(
                        _series.max()
                    )

            except Exception:
                continue

        # -----------------------------------------------------
        # Anomaly Characteristics (Behavioral)
        # -----------------------------------------------------

        supervision_level, imbalance_ratio = self._detect_supervision_level(
            label_cols
        )

        has_trend, has_seasonality, is_stationary, trend_cols, seasonality_cols, stationary_cols = self._detect_temporal_characteristics(
            temporal_candidate_cols,
            timestamp_cols,
            continuous_cols,
            is_time_series
        )

        anomaly_characteristics = self._detect_anomaly_characteristics(
            dataset_type,
            has_seasonality,
            has_trend,
            text_cols,
            categorical_cols,
            id_cols,
            continuous_cols
        )

        # -----------------------------------------------------
        # Return MetaFeatureVector
        # -----------------------------------------------------

        return MetaFeatureVector(

            total_rows=self.original_rows,
            total_columns=total_cols,

            dataset_type=dataset_type,

            missing_data_ratio=missing_ratio,
            feature_to_sample_ratio=feature_to_sample_ratio,

            high_dimensionality=high_dimensionality,

            continuous_cols=continuous_cols,
            categorical_cols=categorical_cols,
            timestamp_cols=timestamp_cols,
            text_cols=text_cols,
            id_cols=id_cols,

            temporal_candidate_cols=temporal_candidate_cols,
            label_cols=label_cols,

            numeric_ratio=numeric_ratio,
            categorical_ratio=categorical_ratio,
            text_ratio=text_ratio,
            id_ratio=id_ratio,

            column_cardinality=column_cardinality,
            column_cardinality_ratio=column_cardinality_ratio,

            high_cardinality_cols=high_cardinality_cols,

            is_time_series=is_time_series,
            temporal_regularity=temporal_regularity,

            temporal_min=temporal_min,
            temporal_max=temporal_max,

            observations_per_entity=observations_per_entity,
            
            anomaly_characteristics=anomaly_characteristics,
            supervision_level=supervision_level,
            imbalance_ratio=imbalance_ratio,
            has_trend=has_trend,
            has_seasonality=has_seasonality,
            is_stationary=is_stationary,
            trend_cols=trend_cols,
            seasonality_cols=seasonality_cols,
            stationary_cols=stationary_cols
        )

    # =========================================================
    # ID Detection
    # =========================================================

    def _looks_like_id(
        self,
        col: str,
        unique_ratio: float,
        series: pd.Series
    ) -> bool:

        name = col.lower().strip()

        id_patterns = [
            "id",
            "uuid",
            "identifier",
            "customer_id",
            "user_id",
            "account_id",
            "transaction_id",
            "record_id"
        ]

        # ---------------------------------------------
        # Strong name-based signal
        # ---------------------------------------------

        name_match = any(
            re.search(
                rf"(^|[_\-\s]){re.escape(pattern)}($|[_\-\s])",
                name
            )
            for pattern in id_patterns
        )

        if name_match and unique_ratio >= 0.5:
            return True

        # ---------------------------------------------
        # UUID Pattern Detection
        # ---------------------------------------------

        uuid_pattern = re.compile(
            r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
            re.IGNORECASE
        )

        sample = series.dropna().head(20)
        if len(sample) > 0:
            uuid_match_ratio = sum(
                bool(uuid_pattern.match(str(v)))
                for v in sample
            ) / len(sample)

            if uuid_match_ratio >= 0.8:
                return True

        # ---------------------------------------------
        # Very high cardinality
        #
        # IMPORTANT:
        # Small datasets are excluded.
        # A 5-row dataset naturally has 100%
        # uniqueness for many normal features.
        #
        # We also reject very long strings (e.g. log messages)
        # unless they matched the UUID pattern above.
        # ---------------------------------------------

        avg_len = series.dropna().astype(str).str.len().mean()
        if avg_len > 50:
            return False

        if (
            len(series) >= 100
            and unique_ratio >= 0.99
        ):
            return True

        return False

    # =========================================================
    # Text Detection
    # =========================================================

    def _looks_like_text(
        self,
        col: str,
        series: pd.Series
    ) -> bool:

        name = col.lower()

        text_keywords = [
            "text",
            "message",
            "description",
            "comment",
            "content",
            "log",
            "error",
            "event"
        ]

        if any(
            keyword in name
            for keyword in text_keywords
        ):
            return True

        non_null = series.dropna()

        if len(non_null) == 0:
            return False

        avg_length = (
            non_null.astype(str)
            .str.len()
            .mean()
        )

        return avg_length >= self.text_length_threshold

    # =========================================================
    # Temporal Detection
    # =========================================================

    def _looks_temporal(
        self,
        col: str,
        series: pd.Series
    ) -> bool:

        name = col.lower()

        temporal_keywords = [
            "time",
            "timestamp",
            "date",
            "datetime",
            "year",
            "month",
            "day"
        ]

        return any(
            keyword in name
            for keyword in temporal_keywords
        )

    # =========================================================
    # Label Detection
    # =========================================================

    def _detect_label_columns(self):

        label_cols = []

        # -------------------------------------------------
        # Graph structural columns to exclude
        # -------------------------------------------------

        graph_structural_cols = (
            self._detect_graph_structural_cols()
        )

        # -------------------------------------------------
        # Keyword-based detection (word-boundary)
        # -------------------------------------------------

        label_keywords = [
            "label",
            "target",
            "class",
            "classification",
            "response",
            "ground_truth",
            "groundtruth",
            "anomaly",
            "fraud",
            "is_fraud",
            "is_anomaly",
            "behavior",
            "outlier"
        ]

        for col in self.df.columns:

            name = col.lower().strip()

            if col in graph_structural_cols:
                continue

            tokens = re.split(r'[_\-\s.]+', name)

            if any(
                keyword == name
                or any(
                    keyword == token
                    for token in tokens
                )
                for keyword in label_keywords
            ):

                label_cols.append(col)

        # -------------------------------------------------
        # Binary label candidate detection
        # -------------------------------------------------

        for col in self.df.columns:

            if col in label_cols:
                continue

            if col in graph_structural_cols:
                continue

            series = self.df[col].dropna()
            unique_count = series.nunique()

            if unique_count != 2:
                continue

            is_binary_label = False

            # Numeric 0/1 — sadece son sütunsa

            if pd.api.types.is_numeric_dtype(series):

                values = set(series.unique())
                cols_list = list(self.df.columns)

                if (
                    values == {0, 1}
                    and col == cols_list[-1]
                ):
                    is_binary_label = True

            # String — karşılaştırma operatörleri veya son sütun

            elif (
                pd.api.types.is_object_dtype(series)
                or pd.api.types.is_string_dtype(series)
            ):
                cols_list = list(self.df.columns)
                values = series.unique()

                if any(
                    re.search(r'[<>=!]', str(v))
                    for v in values
                ):
                    is_binary_label = True
                elif col == cols_list[-1]:
                    # String binary and is the last column
                    is_binary_label = True

            if is_binary_label:
                label_cols.append(col)

        return label_cols

    # =========================================================
    # Dataset Type Detection
    # =========================================================

    def _detect_dataset_type(
        self,
        timestamp_cols,
        temporal_candidate_cols,
        text_cols,
        continuous_cols=None
    ):

        source_candidates = [
            col for col in self.df.columns
            if col.lower() in [
                "source",
                "source_id",
                "src",
                "from",
                "fromid"
            ]
        ]

        target_candidates = [
            col for col in self.df.columns
            if col.lower() in [
                "target",
                "target_id",
                "dst",
                "to",
                "toid"
            ]
        ]

        # Graph
        if source_candidates and target_candidates:
            return "graph"

        # Check if the specific provided file is a graph JSON
        if self.dataset_name and self.dataset_name.endswith(".json") and "graph" in self.dataset_name.lower():
            return "graph"

        # Text / logs
        if len(text_cols) > 0:
            total = len(self.df.columns)
            text_ratio = len(text_cols) / total

            # Veri setinin çoğunluğu text ise veya en az 2 text sütun varsa
            if text_ratio >= 0.3 or len(text_cols) >= 2:
                return "text"

        # Actual datetime
        if len(timestamp_cols) > 0:
            return "time_series"

        # Implicit time-series
        # Timestamp sütunu yok ama veri yapısal olarak
        # zaman serisine benziyor (satır sırasına bağımlı)

        if continuous_cols and self._looks_like_implicit_time_series(
            continuous_cols
        ):
            return "time_series"

        return "tabular"

    # =========================================================
    # Time-Series Detection
    # =========================================================

    def _is_time_series(
        self,
        temporal_candidate_cols,
        timestamp_cols,
        continuous_cols=None
    ):

        # Explicit timestamp column
        if len(timestamp_cols) > 0:
            return True

        # Numeric temporal candidate
        if len(temporal_candidate_cols) > 0:

            for col in temporal_candidate_cols:

                series = self.df[col].dropna()

                if len(series) < 10:
                    continue

                if series.nunique() < 3:
                    continue

                return True

        # Implicit time-series
        # Timestamp sütunu yok ama veri satır sırasına bağımlı

        if continuous_cols and self._looks_like_implicit_time_series(
            continuous_cols
        ):
            return True

        return False

    # =========================================================
    # Temporal Regularity
    # =========================================================

    def _temporal_regularity(
        self,
        temporal_candidate_cols
    ):

        if len(temporal_candidate_cols) == 0:
            return "none"

        for col in temporal_candidate_cols:

            series = self.df[col].dropna()

            if len(series) < 3:
                continue

            try:

                # String/object → datetime dönüşümü

                if (
                    pd.api.types.is_object_dtype(series)
                    or pd.api.types.is_string_dtype(series)
                ):
                    series = pd.to_datetime(
                        series,
                        format="mixed",
                        errors="coerce"
                    ).dropna()

                    if len(series) < 3:
                        continue

                sorted_values = series.sort_values()

                differences = (
                    sorted_values
                    .diff()
                    .dropna()
                )

                if len(differences) == 0:
                    continue

                # Datetime → saniyeye çevir

                if pd.api.types.is_datetime64_any_dtype(
                    sorted_values
                ):
                    differences = (
                        differences.dt.total_seconds()
                    )

                unique_differences = (
                    differences.nunique()
                )

                if unique_differences == 1:
                    return "regular"

                # Neredeyse düzenli kontrol (CV < %10)

                mean_diff = differences.mean()

                if mean_diff != 0:

                    cv = differences.std() / mean_diff

                    if cv < 0.1:
                        return "near_regular"

                return "irregular"

            except Exception:
                continue

        return "unknown"

    # =========================================================
    # High Dimensionality
    # =========================================================

    def _is_high_dimensional(
        self,
        total_rows,
        total_cols
    ):

        # Absolute dimensionality
        if total_cols >= 100:
            return True

        # Feature/sample ratio
        # Minimum satır eşiği: küçük veri setlerinde
        # ratio doğal olarak yüksek olur

        if total_rows >= 50:

            ratio = total_cols / total_rows

            if ratio >= 0.1:
                return True

        return False

    # =========================================================
    # Entity Statistics
    # =========================================================

    def _observations_per_entity(
        self,
        id_cols
    ):

        if len(id_cols) == 0:
            return 0.0

        try:

            entity_counts = (
                self.df
                .groupby(id_cols)
                .size()
            )

            if len(entity_counts) == 0:
                return 0.0

            return float(
                entity_counts.mean()
            )

        except Exception:

            return 0.0

    # =========================================================
    # Numeric Categorical Detection
    # =========================================================

    def _is_numeric_categorical(
        self,
        series: pd.Series,
        unique_count: int,
        total_rows: int
    ) -> bool:

        # -------------------------------------------------
        # Layer 1: Binary — kesin categorical
        # Örnek: 0/1 flag, yes/no encoded
        # -------------------------------------------------

        if unique_count <= 2:
            return True

        # -------------------------------------------------
        # Layer 2: Float dtype → büyük ihtimalle continuous
        # Ölçümler, miktarlar, koordinatlar vb.
        # -------------------------------------------------

        if pd.api.types.is_float_dtype(series):
            return False

        # -------------------------------------------------
        # Layer 3: Integer dtype — detaylı analiz
        # -------------------------------------------------

        non_null = series.dropna()

        if len(non_null) == 0:
            return True

        value_range = non_null.max() - non_null.min()

        # Geniş değer aralığı + yeterli unique → continuous
        # Örnek: age [17-90], capital.gain [0-99999]

        if value_range > 20 and unique_count > 5:
            return False

        # Çok sayıda unique değer → continuous
        # Adaptif eşik: veri setinin boyutuna göre

        adaptive_threshold = max(
            10,
            min(20, int(total_rows * 0.1))
        )

        if unique_count > adaptive_threshold:
            return False

        # Kalan → categorical
        # Örnek: education.num [1-16], rating [1-5]

        return True

    # =========================================================
    # Graph Structural Column Detection
    # =========================================================

    def _detect_graph_structural_cols(self):

        structural = set()

        source_names = {
            "source", "source_id", "src", "from"
        }

        target_names = {
            "target", "target_id", "dst", "to"
        }

        sources = [
            c for c in self.df.columns
            if c.lower() in source_names
        ]

        targets = [
            c for c in self.df.columns
            if c.lower() in target_names
        ]

        # Her ikisi de varsa graph yapısı olarak kabul et

        if sources and targets:
            structural.update(sources)
            structural.update(targets)

        return structural

    # =========================================================
    # Implicit Time-Series Detection
    # =========================================================

    def _looks_like_implicit_time_series(
        self,
        continuous_cols
    ) -> bool:
        """
        Timestamp sütunu olmayan implicit zaman serilerini tespit eder.

        Strateji:
        1. Veri setinin büyük çoğunluğu numeric olmalı
        2. Yeterli satır sayısı olmalı
        3. Birden fazla sütunda yüksek lag-1 otokorelasyon
           olmalı (ardışık değerler birbirine yakın)
        """

        total_rows, total_cols = self.df.shape

        # -------------------------------------------------
        # Koşul 1: Yeterli veri
        # Küçük veri setlerinde otokorelasyon anlamlı değil
        # -------------------------------------------------

        if total_rows < 50:
            return False

        # -------------------------------------------------
        # Koşul 2: Çoğunlukla numeric sütunlar
        # Tabular veriler genelde karışık tiptedir
        # -------------------------------------------------

        numeric_ratio = len(continuous_cols) / total_cols

        if numeric_ratio < 0.7:
            return False

        # -------------------------------------------------
        # Koşul 3: Lag-1 otokorelasyon analizi
        # Zaman serilerinde ardışık değerler güçlü korelasyon
        # gösterir. Tabular verilerde bu yapı yoktur.
        # -------------------------------------------------

        # En fazla 8 sütun kontrol et (performans için)
        cols_to_check = continuous_cols[:8]
        high_autocorr_count = 0

        for col in cols_to_check:

            series = self.df[col].dropna()

            if len(series) < 20:
                continue

            try:
                autocorr = series.autocorr(lag=1)

                # Yüksek otokorelasyon: |r| > 0.7
                if autocorr is not None and abs(autocorr) > 0.7:
                    high_autocorr_count += 1

            except Exception:
                continue

        # -------------------------------------------------
        # Karar: Kontrol edilen sütunların çoğunda
        # yüksek otokorelasyon varsa → implicit time-series
        # -------------------------------------------------

        if len(cols_to_check) == 0:
            return False

        autocorr_ratio = (
            high_autocorr_count / len(cols_to_check)
        )

        return autocorr_ratio >= 0.5

    # =========================================================
    # Anomaly Characteristics & Supervision
    # =========================================================

    def _detect_supervision_level(self, label_cols: list) -> tuple:
        if not label_cols:
            return "unsupervised", 0.0
            
        # Analyze the first label column found
        col = label_cols[0]
        series = self.df[col].dropna()
        
        if len(series) == 0:
            return "unsupervised", 0.0
            
        value_counts = series.value_counts(normalize=True)
        
        if len(value_counts) == 1:
            return "semi-supervised", 0.0
            
        # Assuming the minority class is the anomaly
        imbalance_ratio = value_counts.min()
        
        if imbalance_ratio < 0.05:
            # Highly imbalanced, often treated as semi-supervised or requires specific tuning
            return "semi-supervised", float(imbalance_ratio)
            
        return "supervised", float(imbalance_ratio)

    def _detect_temporal_characteristics(
        self, 
        temporal_candidate_cols: list, 
        timestamp_cols: list, 
        continuous_cols: list,
        is_time_series: bool
    ) -> tuple:
        has_trend = False
        has_seasonality = False
        is_stationary = False
        trend_cols = []
        seasonality_cols = []
        stationary_cols = []
        
        if not is_time_series:
            return has_trend, has_seasonality, is_stationary, trend_cols, seasonality_cols, stationary_cols
            
        if not continuous_cols:
            return has_trend, has_seasonality, is_stationary, trend_cols, seasonality_cols, stationary_cols
            
        # Test up to 20 continuous columns to avoid performance issues
        cols_to_test = continuous_cols[:20]
        
        valid_cols = 0
        
        for target_col in cols_to_test:
            series = self.df[target_col].dropna()
            
            if len(series) < 50:
                continue
                
            valid_cols += 1
            col_stationary = False
            col_trend = False
            col_seasonality = False
            
            if HAS_STATSMODELS:
                # Use statsmodels
                try:
                    # Stationarity (ADF Test)
                    result = adfuller(series.values)
                    if result[1] < 0.05:
                        col_stationary = True
                    
                    # Seasonality & Trend (ACF)
                    acf_vals = acf(series.values, nlags=min(40, len(series) // 2), fft=True)
                    
                    # Simple heuristic for trend: slow decay in ACF
                    if acf_vals[1] > 0.7 and acf_vals[5] > 0.4:
                        col_trend = True
                        
                    # Simple heuristic for seasonality: significant peaks at lags > 1
                    for i in range(2, len(acf_vals) - 1):
                        if acf_vals[i] > acf_vals[i-1] and acf_vals[i] > acf_vals[i+1] and acf_vals[i] > 0.3:
                            col_seasonality = True
                            break
                except Exception:
                    pass
            else:
                # Fallback to pandas
                try:
                    # Stationarity guess using split variance
                    half = len(series) // 2
                    var1, var2 = series.iloc[:half].var(), series.iloc[half:].var()
                    if var1 > 0:
                        ratio = var2 / var1
                        if 0.5 < ratio < 2.0:
                            col_stationary = True
                    
                    # Trend guess using rolling mean variance
                    window = min(20, len(series) // 5)
                    rolling_mean = series.rolling(window=window).mean().dropna()
                    if rolling_mean.var() > (series.var() * 0.5):
                        col_trend = True
                        
                    # Seasonality guess using simple autocorr at common lags
                    for lag in [7, 12, 24]:
                        if lag < len(series) // 2:
                            if abs(series.autocorr(lag)) > 0.4:
                                col_seasonality = True
                                break
                except Exception:
                    pass
                    
            if col_stationary:
                stationary_cols.append(target_col)
            if col_trend:
                trend_cols.append(target_col)
            if col_seasonality:
                seasonality_cols.append(target_col)
                
        if valid_cols > 0:
            # Consider it stationary if at least half of the valid tested columns are stationary
            is_stationary = (len(stationary_cols) >= (valid_cols / 2.0))
            # Consider dataset to have trend/seasonality if >= 20% of evaluated cols exhibit it
            has_trend = (len(trend_cols) >= (valid_cols * 0.20))
            has_seasonality = (len(seasonality_cols) >= (valid_cols * 0.20))
                
        return has_trend, has_seasonality, is_stationary, trend_cols, seasonality_cols, stationary_cols

    def _detect_anomaly_characteristics(
        self, 
        dataset_type: str, 
        has_seasonality: bool, 
        has_trend: bool,
        text_cols: list,
        categorical_cols: list,
        id_cols: list,
        continuous_cols: list
    ) -> list:
        characteristics = set()
        
        # 1. Contextual anomalies: having categorical 'context' and continuous 'behavior'
        if len(categorical_cols) > 0 and len(continuous_cols) > 0:
            characteristics.add("contextual")
            
        # 2. Point anomalies are almost always possible
        characteristics.add("point")
        
        # 3. Statistical Collective Anomaly Detection
        is_collective = False
        
        # Rule-based fallback (metadata)
        if dataset_type in ["time_series", "graph", "text"] or has_seasonality or has_trend:
            is_collective = True
            
        # Statistical verification if not already found (checks up to 5 continuous cols for performance)
        if not is_collective and HAS_STATSMODELS and len(continuous_cols) > 0:
            for target_col in continuous_cols[:5]:
                series = self.df[target_col].dropna().values
                n = len(series)
                
                if n < 30:
                    continue
                    
                try:
                    # Zamansal Bağımlılık Testi (Otokorelasyon - ACF)
                    autocorr_values = acf(series, nlags=min(10, n // 4), fft=True)[1:]
                    mean_autocorr = np.mean(np.abs(autocorr_values))
                    
                    # Küresel Sapma (Global Z-Score)
                    global_mean = np.mean(series)
                    global_std = np.std(series) + 1e-8
                    global_z_scores = np.abs((series - global_mean) / global_std)
                    
                    # Yerel Pencere Sapması (Rolling Window Z-Score)
                    window_size = max(5, min(50, n // 20))
                    series_pd = pd.Series(series)
                    rolling_mean = series_pd.rolling(window=window_size, min_periods=1).mean()
                    rolling_std = series_pd.rolling(window=window_size, min_periods=1).std().fillna(1e-8)
                    local_z_scores = np.abs((series - rolling_mean) / rolling_std)
                    
                    # Küresel olarak NORMAL ama yerel olarak SAPMIŞ noktalar (Kollektif Anomali Adayları)
                    collective_candidates = (global_z_scores <= 3.0) & (local_z_scores > 3.0)
                    collective_outlier_ratio = np.sum(collective_candidates) / n
                    
                    has_time_index = (dataset_type == "time_series")
                    
                    # Karar: Otokorelasyon yüksekse veya yerel sapma %1'den fazlaysa kolektif anomali vardır
                    if mean_autocorr >= 0.25 or has_time_index:
                        if collective_outlier_ratio > 0.01:
                            is_collective = True
                            break
                            
                except Exception:
                    continue
                    
        if is_collective:
            characteristics.add("collective")
            
        return list(characteristics)