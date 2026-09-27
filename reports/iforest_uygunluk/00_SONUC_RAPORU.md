# IForest Uygunluk Analizi — Sonuç Raporu

**Veri Seti:** `data\UNSW_NB15_traintest_backdoor.csv`
**Tarih:** 2026-09-27 16:57

---

## Analiz Skorları

| Analiz | Skor |
|--------|------|
| Feature Dağılım Ayrışması | 10.0/10 ██████████ |
| KS Testi Ayrışması | 10.0/10 ██████████ |
| KNN Local Density İzolasyonu | 0.5/10 ░░░░░░░░░░ |

**Genel Uygunluk Skoru: 6.8 / 10**

---

## Karar

✅ IForest bu veri seti için UYGUN

Anomaliler normal veriden yeterince ayrışmış durumda.
IForest'ın izolasyon prensibi bu veri setinde işe yarayabilir.

---

## Üretilen Dosyalar

| Dosya | Açıklama |
|-------|----------|
| `01_feature_dagilim_karsilastirma.csv` | Normal vs Anomaly median/IQR karşılaştırması |
| `02_ks_testi_sonuclari.csv` | Kolmogorov-Smirnov test sonuçları |
| `02_ks_top6_histograms.png` | En farklı 6 feature'ın dağılım histogramları |
| `03_pca_projection.png` | PCA 2D görselleştirme |
| `03_umap_projection.png` | UMAP 2D görselleştirme |
| `04_knn_local_density.csv` | KNN mesafe karşılaştırması |
| `04_knn_mesafe_dagilimi.png` | KNN mesafe dağılım histogramları |