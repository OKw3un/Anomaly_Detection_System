import numpy as np
import pandas as pd
from typing import List

class TimeSeriesFeatureEngineer:
    """
    Zaman serisi verilerine özellik mühendisliği (Feature Engineering) uygular.
    - Gecikme (Lag) özellikleri
    - Hareketli pencere (Rolling window) istatistikleri (mean, std, vb.)
    - Zaman serisi ayrıştırması (Decomposition - Trend, Seasonality, Residual)
    """
    
    def __init__(self, period: int = 21, lags: List[int] = [1, 2, 3]):
        self.period = period
        self.lags = lags
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Girdi DataFrame'ine zaman serisi özelliklerini ekleyerek yeni bir DataFrame döndürür.
        """
        print(f"  [TS Feature Engineer] Zaman serisi özellik mühendisliği başlatıldı (Periyot: {self.period})")
        df_out = df.copy()
        
        # Sadece sayısal sütunlar üzerinde işlem yap
        numeric_cols = df_out.select_dtypes(include=[np.number]).columns.tolist()
        
        # 1. Lag Features & Rolling Windows
        print(f"    > Gecikme (Lag) ve Hareketli (Rolling) istatistikler hesaplanıyor...")
        
        # Sütun ekleme işlemlerinden kaynaklı pandas "PerformanceWarning: DataFrame is highly fragmented"
        # hatasını çözmek için tüm yeni özellikleri bir sözlükte toplayıp tek seferde birleştireceğiz.
        new_features = {}
        
        for col in numeric_cols:
            # Lag özellikleri
            for lag in self.lags:
                new_features[f"{col}_lag_{lag}"] = df_out[col].shift(lag)
            
            # Rolling istatistikler
            new_features[f"{col}_roll_mean_{self.period}"] = df_out[col].rolling(window=self.period, min_periods=1).mean()
            new_features[f"{col}_roll_std_{self.period}"] = df_out[col].rolling(window=self.period, min_periods=1).std().fillna(0)
            new_features[f"{col}_roll_min_{self.period}"] = df_out[col].rolling(window=self.period, min_periods=1).min()
            new_features[f"{col}_roll_max_{self.period}"] = df_out[col].rolling(window=self.period, min_periods=1).max()
        
        # 2. Time Series Decomposition (Trend + Seasonality + Residual)
        print(f"    > Zaman serisi ayrıştırması (Decomposition) yapılarak Residual bileşenleri çıkarılıyor...")
        for col in numeric_cols:
            new_features[f"{col}_residual"] = df_out[col] - new_features[f"{col}_roll_mean_{self.period}"]

        # Tüm yeni özellikleri tek seferde DataFrame'e ekle (Böylece bellek parçalanmaz / fragmented olmaz)
        new_features_df = pd.DataFrame(new_features)
        df_out = pd.concat([df_out, new_features_df], axis=1)

        # Shift işlemlerinden dolayı oluşan NaN değerleri doldur
        df_out = df_out.bfill().fillna(0)
        
        print(f"  [TS Feature Engineer] Özellik mühendisliği tamamlandı. Sütun sayısı {len(df.columns)} -> {len(df_out.columns)} oldu.")
        return df_out
