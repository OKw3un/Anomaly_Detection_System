"""
IForest Uygunluk Analizi — Backdoor Veri Seti
===============================================
Bu script, Isolation Forest algoritmasının UNSW-NB15 Backdoor veri setine
ne kadar uygun olduğunu 4 farklı yöntemle analiz eder:

  1. Öznitelik Dağılım Karşılaştırması (Normal vs Anomaly median/iqr tablosu)
  2. Kolmogorov-Smirnov Testi (Feature Separability — her sütun için p-value)
  3. PCA & UMAP Görselleştirme (2D projeksiyon — ayrışma var mı?)
  4. KNN Local Density Analizi (Anomaliler izole mi, gömülü mü?)

Kullanım:
  python iforest_uygunluk_analizi.py

Çıktılar reports/iforest_uygunluk/ klasörüne kaydedilir.
"""

import os
import sys
import warnings
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # GUI olmadan çalışsın
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap

from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import NearestNeighbors

# Projenin kök dizinine göre import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.engine.preprocessing.iforest_backdoor_dataset_preprocessing import (
    IForestBackdoorPreprocessor,
)

# ────────────────────────────────────────────────────────────────
# Ayarlar
# ────────────────────────────────────────────────────────────────
CSV_PATH = os.path.join("data", "UNSW_NB15_traintest_backdoor.csv")
OUTPUT_DIR = os.path.join("reports", "iforest_uygunluk")
LABEL_COL = "class"          # 0 = Normal, 1 = Backdoor (Anomaly)
KNN_K_VALUES = [5, 10, 20]   # KNN analizinde bakılacak k değerleri

# Matplotlib Türkçe karakter desteği
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data() -> pd.DataFrame:
    """CSV'yi yükler, temel bilgiyi basar."""
    print("=" * 70)
    print("  IForest Uygunluk Analizi — Backdoor Veri Seti")
    print("=" * 70)
    print(f"\n📂 Veri seti: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    n_normal = (df[LABEL_COL] == 0).sum()
    n_anomaly = (df[LABEL_COL] == 1).sum()

    print(f"   Toplam satır  : {len(df):,}")
    print(f"   Normal  (y=0) : {n_normal:,} ({n_normal/len(df)*100:.1f}%)")
    print(f"   Anomaly (y=1) : {n_anomaly:,} ({n_anomaly/len(df)*100:.1f}%)")
    return df


def preprocess(df: pd.DataFrame):
    """
    IForestBackdoorPreprocessor ile veriyi dönüştürür.
    Label sütununu ayırıp, X matrisi + y etiketleri + feature isimlerini döndürür.
    """
    print("\n" + "─" * 70)
    print("  Ön İşleme (IForestBackdoorPreprocessor)")
    print("─" * 70)

    y = df[LABEL_COL].values.copy()

    preprocessor = IForestBackdoorPreprocessor(
        variance_threshold=0.01,
        correlation_threshold=0.85,
        skewness_threshold=3.0,
    )

    preprocessing_steps = {
        "exclude_id_columns": [],
        "exclude_label_columns": [LABEL_COL],
        "impute_missing": True,
        "encode_categorical": True,
        "scale_continuous": False,    # IForest için ölçekleme yapılmaz
    }

    X, _ = preprocessor.fit_transform(df, preprocessing_steps)
    feature_names = preprocessor.feature_names

    print(f"\n   Sonuç matris boyutu: {X.shape}")
    print(f"   Özellik sayısı: {len(feature_names)}")

    return X, y, feature_names


# ═══════════════════════════════════════════════════════════════
# ANALİZ 1: Öznitelik Dağılım Karşılaştırması
# ═══════════════════════════════════════════════════════════════
def analiz_1_feature_dagilimi(X, y, feature_names):
    """
    Her feature için normal vs anomaly verilerinin median, IQR (Q1-Q3),
    min, max değerlerini karşılaştırır.

    Eğer çoğu feature'da normal ≈ anomaly ise IForest'ın başarılı
    olmasını beklemek zaten zor.
    """
    print("\n" + "═" * 70)
    print("  ANALİZ 1: Öznitelik Dağılım Karşılaştırması (Normal vs Anomaly)")
    print("═" * 70)

    normal_mask = y == 0
    anomaly_mask = y == 1

    rows = []
    for i, fname in enumerate(feature_names):
        normal_vals = X[normal_mask, i]
        anomaly_vals = X[anomaly_mask, i]

        n_med = np.median(normal_vals)
        a_med = np.median(anomaly_vals)
        n_q1, n_q3 = np.percentile(normal_vals, [25, 75])
        a_q1, a_q3 = np.percentile(anomaly_vals, [25, 75])

        # Median farkını normalize et (ne kadar farklı?)
        denom = max(abs(n_med), abs(a_med), 1e-9)
        relative_diff = abs(n_med - a_med) / denom

        rows.append({
            "Feature": fname,
            "Normal Median": round(n_med, 4),
            "Normal IQR": f"[{n_q1:.2f}, {n_q3:.2f}]",
            "Anomaly Median": round(a_med, 4),
            "Anomaly IQR": f"[{a_q1:.2f}, {a_q3:.2f}]",
            "Relative Diff": round(relative_diff, 4),
        })

    result_df = pd.DataFrame(rows).sort_values("Relative Diff", ascending=False)

    # Konsola en önemli 30 feature'ı göster
    print("\n  En farklı 30 feature (Relative Diff'e göre sıralı):")
    print("  " + "─" * 95)
    print(f"  {'Feature':<30} {'Normal Med':>12} {'Anomaly Med':>12} {'Rel. Diff':>10} {'Normal IQR':>20} {'Anomaly IQR':>20}")
    print("  " + "─" * 95)
    for _, row in result_df.head(30).iterrows():
        diff_marker = "🔴" if row["Relative Diff"] > 0.5 else ("🟡" if row["Relative Diff"] > 0.1 else "🟢")
        print(f"  {diff_marker} {row['Feature']:<28} {row['Normal Median']:>12.4f} {row['Anomaly Median']:>12.4f} {row['Relative Diff']:>10.4f} {row['Normal IQR']:>20} {row['Anomaly IQR']:>20}")

    # Ayrışma özeti
    high_diff = (result_df["Relative Diff"] > 0.5).sum()
    mid_diff = ((result_df["Relative Diff"] > 0.1) & (result_df["Relative Diff"] <= 0.5)).sum()
    low_diff = (result_df["Relative Diff"] <= 0.1).sum()

    print(f"\n  📊 Özet:")
    print(f"     🔴 Güçlü ayrışma (Diff > 0.5) : {high_diff} feature")
    print(f"     🟡 Orta ayrışma  (0.1 < Diff ≤ 0.5) : {mid_diff} feature")
    print(f"     🟢 Zayıf/yok    (Diff ≤ 0.1) : {low_diff} feature")

    if high_diff == 0:
        print("\n  ⚠️  UYARI: Hiçbir feature'da güçlü median ayrışması yok!")
        print("     IForest'ın tek başına bu veri setinde başarılı olması çok düşük ihtimal.")

    # CSV olarak kaydet
    csv_path = os.path.join(OUTPUT_DIR, "01_feature_dagilim_karsilastirma.csv")
    result_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n  💾 Detaylı tablo kaydedildi: {csv_path}")

    return result_df


# ═══════════════════════════════════════════════════════════════
# ANALİZ 2: Kolmogorov-Smirnov (KS) Testi
# ═══════════════════════════════════════════════════════════════
def analiz_2_ks_testi(X, y, feature_names):
    """
    Her sürekli feature için normal ve anomaly dağılımları arasında
    Kolmogorov-Smirnov testi uygular.

    KS İstatistiği (D):
      - 0'a yakın → dağılımlar neredeyse aynı (IForest için kötü)
      - 1'e yakın → dağılımlar tamamen farklı (IForest için iyi)

    p-value:
      - > 0.05 → Fark istatistiksel olarak anlamsız
      - < 0.05 → Fark istatistiksel olarak anlamlı
    """
    print("\n" + "═" * 70)
    print("  ANALİZ 2: Kolmogorov-Smirnov (KS) Testi")
    print("═" * 70)

    normal_mask = y == 0
    anomaly_mask = y == 1

    rows = []
    for i, fname in enumerate(feature_names):
        normal_vals = X[normal_mask, i]
        anomaly_vals = X[anomaly_mask, i]

        ks_stat, p_value = ks_2samp(normal_vals, anomaly_vals)

        rows.append({
            "Feature": fname,
            "KS Statistic (D)": round(ks_stat, 6),
            "p-value": p_value,
            "Significant (p<0.05)": "✅ Evet" if p_value < 0.05 else "❌ Hayır",
        })

    result_df = pd.DataFrame(rows).sort_values("KS Statistic (D)", ascending=False)

    # Konsola en anlamlı 30 feature
    print("\n  En yüksek KS istatistiğine sahip 30 feature:")
    print("  " + "─" * 75)
    print(f"  {'Feature':<30} {'KS Stat (D)':>12} {'p-value':>15} {'Anlamlı?':>10}")
    print("  " + "─" * 75)
    for _, row in result_df.head(30).iterrows():
        p_str = f"{row['p-value']:.2e}" if row['p-value'] < 0.001 else f"{row['p-value']:.6f}"
        marker = "🔴" if row["KS Statistic (D)"] > 0.3 else ("🟡" if row["KS Statistic (D)"] > 0.1 else "🟢")
        print(f"  {marker} {row['Feature']:<28} {row['KS Statistic (D)']:>12.6f} {p_str:>15} {row['Significant (p<0.05)']:>10}")

    # Özet
    significant_count = (result_df["p-value"] < 0.05).sum()
    high_ks = (result_df["KS Statistic (D)"] > 0.3).sum()
    mid_ks = ((result_df["KS Statistic (D)"] > 0.1) & (result_df["KS Statistic (D)"] <= 0.3)).sum()
    low_ks = (result_df["KS Statistic (D)"] <= 0.1).sum()

    print(f"\n  📊 Özet:")
    print(f"     İstatistiksel olarak anlamlı (p < 0.05): {significant_count}/{len(result_df)} feature")
    print(f"     🔴 Güçlü ayrışma (D > 0.3) : {high_ks} feature")
    print(f"     🟡 Orta ayrışma  (0.1 < D ≤ 0.3) : {mid_ks} feature")
    print(f"     🟢 Zayıf/yok    (D ≤ 0.1) : {low_ks} feature")

    print("\n  📖 KS Testi Yorumu:")
    print("     KS istatistiği (D), iki kümülatif dağılım fonksiyonu arasındaki")
    print("     maksimum dikey mesafedir.")
    print("     D = 0 → Dağılımlar tamamen örtüşüyor (ayırt edilemez)")
    print("     D = 1 → Dağılımlar hiç örtüşmüyor (mükemmel ayrışma)")
    if high_ks == 0:
        print("\n  ⚠️  UYARI: Hiçbir feature'da D > 0.3 yok!")
        print("     Bu, backdoor trafiğin normal trafikle iç içe geçtiğini gösterir.")

    # CSV kaydet
    csv_path = os.path.join(OUTPUT_DIR, "02_ks_testi_sonuclari.csv")
    result_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n  💾 Detaylı tablo kaydedildi: {csv_path}")

    # En yüksek KS'e sahip top 6 feature'ın histogram görselleştirmesi
    _plot_top_ks_histograms(X, y, feature_names, result_df)

    return result_df


def _plot_top_ks_histograms(X, y, feature_names, ks_df):
    """En iyi ayrışan 6 feature'ın dağılım histogramlarını çizer."""
    top_features = ks_df.head(6)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("KS Testine Göre En Farklı 6 Öznitelik — Normal vs Anomaly Dağılımları",
                 fontsize=14, fontweight="bold", y=0.98)

    normal_mask = y == 0
    anomaly_mask = y == 1

    for idx, (_, row) in enumerate(top_features.iterrows()):
        ax = axes[idx // 3, idx % 3]
        fname = row["Feature"]
        f_idx = feature_names.index(fname)

        normal_vals = X[normal_mask, f_idx]
        anomaly_vals = X[anomaly_mask, f_idx]

        # Ortak bins oluştur
        all_vals = np.concatenate([normal_vals, anomaly_vals])
        bins = np.linspace(np.percentile(all_vals, 1), np.percentile(all_vals, 99), 60)

        ax.hist(normal_vals, bins=bins, alpha=0.6, color="#3498db", label="Normal", density=True)
        ax.hist(anomaly_vals, bins=bins, alpha=0.6, color="#e74c3c", label="Anomaly", density=True)

        ax.set_title(f"{fname}\nKS D = {row['KS Statistic (D)']:.4f}", fontsize=10)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_path = os.path.join(OUTPUT_DIR, "02_ks_top6_histograms.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📈 Top 6 KS histogram: {fig_path}")


# ═══════════════════════════════════════════════════════════════
# ANALİZ 3: PCA & UMAP Görselleştirme
# ═══════════════════════════════════════════════════════════════
def analiz_3_pca_umap(X, y, feature_names):
    """
    PCA (2 bileşen) ve UMAP (2 bileşen) ile veriyi 2D'ye indirger.
    Normal ve anomaly noktalarını farklı renklerle görselleştirir.

    Eğer anomaliler normal noktaların arasına dağılmışsa → IForest için zor
    Eğer anomaliler ayrı kümeler oluşturuyorsa → IForest için kolay
    """
    print("\n" + "═" * 70)
    print("  ANALİZ 3: PCA & UMAP Görselleştirme")
    print("═" * 70)

    normal_mask = y == 0
    anomaly_mask = y == 1

    # Ölçekleme (görselleştirme için, IForest için değil)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # ── PCA ──
    print("\n  🔄 PCA (2 bileşen) hesaplanıyor...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    explained_var = pca.explained_variance_ratio_
    print(f"     Açıklanan varyans: PC1={explained_var[0]*100:.1f}%, PC2={explained_var[1]*100:.1f}%")
    print(f"     Toplam: {sum(explained_var)*100:.1f}%")

    # PCA Plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Önce normalleri çiz (arka plan)
    ax.scatter(X_pca[normal_mask, 0], X_pca[normal_mask, 1],
               c="#3498db", alpha=0.15, s=5, label=f"Normal (n={normal_mask.sum():,})", zorder=1)
    # Sonra anomalileri çiz (ön plan)
    ax.scatter(X_pca[anomaly_mask, 0], X_pca[anomaly_mask, 1],
               c="#e74c3c", alpha=0.7, s=20, edgecolors="black", linewidths=0.3,
               label=f"Anomaly (n={anomaly_mask.sum():,})", zorder=2)

    ax.set_xlabel(f"PC1 ({explained_var[0]*100:.1f}% varyans)", fontsize=11)
    ax.set_ylabel(f"PC2 ({explained_var[1]*100:.1f}% varyans)", fontsize=11)
    ax.set_title("PCA Projeksiyonu — Normal vs Backdoor Anomaly", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")

    fig_path = os.path.join(OUTPUT_DIR, "03_pca_projection.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📈 PCA plot: {fig_path}")

    # ── UMAP ──
    print("\n  🔄 UMAP (2 bileşen) hesaplanıyor... (bu biraz sürebilir)")
    try:
        import umap

        # Büyük veri setleri için örnekleme
        max_samples = 15000
        if len(X_scaled) > max_samples:
            # Anomalilerin tümünü al, normallerden örnekle
            anomaly_idx = np.where(anomaly_mask)[0]
            normal_idx = np.where(normal_mask)[0]
            n_normal_sample = max_samples - len(anomaly_idx)
            rng = np.random.RandomState(42)
            sampled_normal_idx = rng.choice(normal_idx, size=n_normal_sample, replace=False)
            sampled_idx = np.concatenate([sampled_normal_idx, anomaly_idx])
            np.random.shuffle(sampled_idx)

            X_umap_input = X_scaled[sampled_idx]
            y_umap = y[sampled_idx]
            print(f"     UMAP için örnekleme: {len(sampled_idx):,} nokta ({n_normal_sample:,} normal + {len(anomaly_idx):,} anomaly)")
        else:
            X_umap_input = X_scaled
            y_umap = y

        reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3, random_state=42, verbose=False)
        X_umap = reducer.fit_transform(X_umap_input)

        umap_normal_mask = y_umap == 0
        umap_anomaly_mask = y_umap == 1

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(X_umap[umap_normal_mask, 0], X_umap[umap_normal_mask, 1],
                   c="#3498db", alpha=0.15, s=5, label=f"Normal (n={umap_normal_mask.sum():,})", zorder=1)
        ax.scatter(X_umap[umap_anomaly_mask, 0], X_umap[umap_anomaly_mask, 1],
                   c="#e74c3c", alpha=0.7, s=20, edgecolors="black", linewidths=0.3,
                   label=f"Anomaly (n={umap_anomaly_mask.sum():,})", zorder=2)

        ax.set_xlabel("UMAP-1", fontsize=11)
        ax.set_ylabel("UMAP-2", fontsize=11)
        ax.set_title("UMAP Projeksiyonu — Normal vs Backdoor Anomaly", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10, loc="upper right")

        fig_path = os.path.join(OUTPUT_DIR, "03_umap_projection.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  📈 UMAP plot: {fig_path}")

    except ImportError:
        print("  ⚠️  UMAP yüklü değil. 'pip install umap-learn' ile yükleyebilirsiniz.")
    except Exception as e:
        print(f"  ⚠️  UMAP hatası: {e}")


# ═══════════════════════════════════════════════════════════════
# ANALİZ 4: KNN Local Density Analizi
# ═══════════════════════════════════════════════════════════════
def analiz_4_knn_local_density(X, y, feature_names):
    """
    Her noktanın k-nearest-neighbor mesafesini hesaplar.
    Normal vs anomaly median KNN mesafelerini karşılaştırır.

    Eğer mesafeler benzer → Anomaliler normal veriye gömülmüş (IForest için zor)
    Eğer anomaly mesafesi çok büyük → Anomaliler izole (IForest için uygun)

    Ayrıca her anomaly noktasının en yakın k komşusunun kaçının normal olduğunu hesaplar.
    Bu, anomalilerin normal veriye ne kadar "sarıldığını" gösterir.
    """
    print("\n" + "═" * 70)
    print("  ANALİZ 4: KNN Local Density Analizi")
    print("═" * 70)

    normal_mask = y == 0
    anomaly_mask = y == 1

    # Ölçekleme (mesafe hesabı için gerekli)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"\n  Toplam nokta: {len(X_scaled):,}")
    print(f"  Normal: {normal_mask.sum():,} | Anomaly: {anomaly_mask.sum():,}")

    results = []

    for k in KNN_K_VALUES:
        print(f"\n  🔄 k={k} komşu hesaplanıyor...")
        t0 = time.time()

        nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", n_jobs=-1)
        nn.fit(X_scaled)
        distances, indices = nn.kneighbors(X_scaled)

        # İlk sütun kendisi (mesafe=0), onu atla
        knn_distances = distances[:, 1:]  # (n_samples, k)

        # Ortalama KNN mesafesi
        mean_knn_dist = knn_distances.mean(axis=1)  # Her nokta için ortalama

        normal_median_dist = np.median(mean_knn_dist[normal_mask])
        anomaly_median_dist = np.median(mean_knn_dist[anomaly_mask])
        normal_mean_dist = np.mean(mean_knn_dist[normal_mask])
        anomaly_mean_dist = np.mean(mean_knn_dist[anomaly_mask])

        ratio = anomaly_median_dist / max(normal_median_dist, 1e-9)

        elapsed = time.time() - t0

        print(f"     Normal  median KNN mesafesi: {normal_median_dist:.4f}")
        print(f"     Anomaly median KNN mesafesi: {anomaly_median_dist:.4f}")
        print(f"     Oran (Anomaly / Normal)    : {ratio:.2f}x")
        print(f"     Normal  mean  KNN mesafesi : {normal_mean_dist:.4f}")
        print(f"     Anomaly mean  KNN mesafesi : {anomaly_mean_dist:.4f}")
        print(f"     Süre: {elapsed:.1f}s")

        results.append({
            "k": k,
            "Normal Median Dist": round(normal_median_dist, 4),
            "Anomaly Median Dist": round(anomaly_median_dist, 4),
            "Ratio (A/N)": round(ratio, 2),
            "Normal Mean Dist": round(normal_mean_dist, 4),
            "Anomaly Mean Dist": round(anomaly_mean_dist, 4),
        })

        # ── Anomalilerin komşu analizi ──
        # Her anomaly noktasının en yakın k komşusunun kaçı normal?
        anomaly_indices = np.where(anomaly_mask)[0]
        neighbor_labels = y[indices[anomaly_indices, 1:]]  # (n_anomaly, k)
        normal_neighbor_ratio = (neighbor_labels == 0).mean()

        print(f"     Anomali komşularının Normal oranı: {normal_neighbor_ratio*100:.1f}%")
        if normal_neighbor_ratio > 0.8:
            print(f"     ⚠️  Anomalilerin komşularının %{normal_neighbor_ratio*100:.0f}'i normal! → Gömülmüş durumda")
        elif normal_neighbor_ratio > 0.5:
            print(f"     🟡 Anomalilerin komşularının %{normal_neighbor_ratio*100:.0f}'i normal. → Kısmen gömülmüş")
        else:
            print(f"     ✅ Anomalilerin çoğunluğu diğer anomalilerle komşu. → İzole küme")

    # Sonuç tablosu
    result_df = pd.DataFrame(results)
    print("\n  📊 KNN Mesafe Karşılaştırma Tablosu:")
    print("  " + "─" * 80)
    print(f"  {'k':>4} {'Normal Median':>16} {'Anomaly Median':>16} {'Oran (A/N)':>12} {'Normal Mean':>14} {'Anomaly Mean':>14}")
    print("  " + "─" * 80)
    for _, row in result_df.iterrows():
        print(f"  {row['k']:>4} {row['Normal Median Dist']:>16.4f} {row['Anomaly Median Dist']:>16.4f} {row['Ratio (A/N)']:>12.2f}x {row['Normal Mean Dist']:>14.4f} {row['Anomaly Mean Dist']:>14.4f}")

    # Yorum
    avg_ratio = np.mean([r["Ratio (A/N)"] for r in results])
    print(f"\n  📖 Yorum:")
    if avg_ratio < 1.3:
        print(f"     Ortalama oran: {avg_ratio:.2f}x")
        print(f"     ❌ Anomaliler normal veriye GÖMÜLMÜŞ durumda.")
        print(f"     IForest bu veri seti için UYGUN DEĞİL.")
        print(f"     Anomaliler normal noktaların arasında bulunuyor, IForest bunları izole edemez.")
    elif avg_ratio < 2.0:
        print(f"     Ortalama oran: {avg_ratio:.2f}x")
        print(f"     🟡 Anomaliler kısmen izole ama güçlü bir ayrışma yok.")
        print(f"     IForest kısmen işe yarayabilir ama yüksek performans beklemeyin.")
    else:
        print(f"     Ortalama oran: {avg_ratio:.2f}x")
        print(f"     ✅ Anomaliler ciddi şekilde izole!")
        print(f"     IForest bu veri seti için UYGUN olmalı.")

    # KNN dağılım grafikleri
    _plot_knn_distributions(X_scaled, y, KNN_K_VALUES)

    # CSV kaydet
    csv_path = os.path.join(OUTPUT_DIR, "04_knn_local_density.csv")
    result_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n  💾 KNN sonuçları: {csv_path}")

    return result_df


def _plot_knn_distributions(X_scaled, y, k_values):
    """Normal ve anomaly için KNN mesafe dağılımlarını histogram olarak çizer."""
    normal_mask = y == 0
    anomaly_mask = y == 1

    fig, axes = plt.subplots(1, len(k_values), figsize=(6 * len(k_values), 5))
    fig.suptitle("KNN Mesafe Dağılımları — Normal vs Anomaly",
                 fontsize=14, fontweight="bold")

    if len(k_values) == 1:
        axes = [axes]

    for ax, k in zip(axes, k_values):
        nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", n_jobs=-1)
        nn.fit(X_scaled)
        distances, _ = nn.kneighbors(X_scaled)
        mean_knn_dist = distances[:, 1:].mean(axis=1)

        normal_dists = mean_knn_dist[normal_mask]
        anomaly_dists = mean_knn_dist[anomaly_mask]

        # Bins: 1-99 percentile arası
        all_dists = np.concatenate([normal_dists, anomaly_dists])
        bins = np.linspace(np.percentile(all_dists, 1), np.percentile(all_dists, 99), 80)

        ax.hist(normal_dists, bins=bins, alpha=0.6, color="#3498db", label="Normal", density=True)
        ax.hist(anomaly_dists, bins=bins, alpha=0.6, color="#e74c3c", label="Anomaly", density=True)

        # Median çizgileri
        n_med = np.median(normal_dists)
        a_med = np.median(anomaly_dists)
        ax.axvline(n_med, color="#2980b9", linestyle="--", linewidth=2, label=f"Normal Med: {n_med:.3f}")
        ax.axvline(a_med, color="#c0392b", linestyle="--", linewidth=2, label=f"Anomaly Med: {a_med:.3f}")

        ax.set_title(f"k = {k}", fontsize=12)
        ax.set_xlabel("Ortalama KNN Mesafesi", fontsize=10)
        ax.set_ylabel("Yoğunluk", fontsize=10)
        ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig_path = os.path.join(OUTPUT_DIR, "04_knn_mesafe_dagilimi.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📈 KNN mesafe dağılımı: {fig_path}")


# ═══════════════════════════════════════════════════════════════
# GENEL SONUÇ RAPORU
# ═══════════════════════════════════════════════════════════════
def genel_sonuc(feat_df, ks_df, knn_df):
    """Tüm analizlerin bir arada değerlendirildiği özet rapor."""
    print("\n" + "█" * 70)
    print("  GENEL SONUÇ RAPORU — IForest Uygunluk Değerlendirmesi")
    print("█" * 70)

    scores = []  # Her analiz 0-10 arası puan

    # 1. Feature dağılımı skoru
    high_diff = (feat_df["Relative Diff"] > 0.5).sum()
    total_features = len(feat_df)
    feat_score = min(10, (high_diff / max(total_features, 1)) * 100)
    scores.append(("Feature Dağılım Ayrışması", feat_score))
    print(f"\n  1️⃣  Feature Dağılım Ayrışması: {feat_score:.1f}/10")
    print(f"      {high_diff}/{total_features} feature'da güçlü median farkı var")

    # 2. KS testi skoru
    high_ks = (ks_df["KS Statistic (D)"] > 0.3).sum()
    ks_score = min(10, (high_ks / max(total_features, 1)) * 100)
    scores.append(("KS Testi Ayrışması", ks_score))
    print(f"\n  2️⃣  KS Testi Ayrışması: {ks_score:.1f}/10")
    print(f"      {high_ks}/{total_features} feature'da D > 0.3")

    # 3. KNN izolasyon skoru
    avg_ratio = knn_df["Ratio (A/N)"].mean()
    if avg_ratio >= 3.0:
        knn_score = 10.0
    elif avg_ratio >= 2.0:
        knn_score = 7.0
    elif avg_ratio >= 1.5:
        knn_score = 4.0
    elif avg_ratio >= 1.2:
        knn_score = 2.0
    else:
        knn_score = 0.5
    scores.append(("KNN Local Density İzolasyonu", knn_score))
    print(f"\n  3️⃣  KNN Local Density İzolasyonu: {knn_score:.1f}/10")
    print(f"      Ortalama mesafe oranı: {avg_ratio:.2f}x")

    # Toplam skor
    avg_score = np.mean([s[1] for s in scores])

    print(f"\n  {'─' * 50}")
    print(f"  📊 GENEL UYGUNLUK SKORU: {avg_score:.1f} / 10")
    print(f"  {'─' * 50}")

    if avg_score < 3:
        verdict = "❌ IForest bu veri seti için UYGUN DEĞİL"
        explanation = (
            "Backdoor trafiği, normal trafikle neredeyse aynı öznitelik dağılımlarına sahip.\n"
            "     Anomaliler lokal olarak normal veriye gömülmüş durumda.\n"
            "     IForest global izolasyon prensibiyle çalıştığı için bu tür\n"
            "     'stealth' anomalileri tespit etmesi çok zor.\n\n"
            "     ÖNERİLER:\n"
            "     • Supervised yöntemler (Random Forest, XGBoost) deneyin\n"
            "     • Semi-supervised yöntemler (DeepSVDD, AutoEncoder) deneyin\n"
            "     • Feature engineering ile daha ayrıştırıcı öznitelikler oluşturun\n"
            "     • LOF (Local Outlier Factor) gibi lokal yoğunluk tabanlı yöntemler deneyin"
        )
    elif avg_score < 6:
        verdict = "🟡 IForest kısmen uygun — yüksek performans beklemeyin"
        explanation = (
            "Bazı özniteliklerde ayrışma var ama çoğunluk örtüşüyor.\n"
            "     IForest kısmen başarılı olabilir ama precision/recall dengesi zor olacak.\n\n"
            "     ÖNERİLER:\n"
            "     • IForest'ı ensemble içinde kullanın (diğer algoritmalarla birlikte)\n"
            "     • Contamination oranını dikkatli ayarlayın\n"
            "     • Sadece en ayrışan feature'ları seçerek (feature selection) deneyin"
        )
    else:
        verdict = "✅ IForest bu veri seti için UYGUN"
        explanation = (
            "Anomaliler normal veriden yeterince ayrışmış durumda.\n"
            "     IForest'ın izolasyon prensibi bu veri setinde işe yarayabilir."
        )

    print(f"\n  🏆 KARAR: {verdict}")
    print(f"\n     {explanation}")

    # Raporu dosyaya kaydet
    _save_report(scores, avg_score, verdict, explanation)


def _save_report(scores, avg_score, verdict, explanation):
    """Markdown formatında rapor dosyası oluşturur."""
    report_path = os.path.join(OUTPUT_DIR, "00_SONUC_RAPORU.md")

    lines = [
        "# IForest Uygunluk Analizi — Sonuç Raporu",
        "",
        f"**Veri Seti:** `{CSV_PATH}`",
        f"**Tarih:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "---",
        "",
        "## Analiz Skorları",
        "",
        "| Analiz | Skor |",
        "|--------|------|",
    ]

    for name, score in scores:
        bar = "█" * int(score) + "░" * (10 - int(score))
        lines.append(f"| {name} | {score:.1f}/10 {bar} |")

    lines.extend([
        "",
        f"**Genel Uygunluk Skoru: {avg_score:.1f} / 10**",
        "",
        "---",
        "",
        f"## Karar",
        "",
        verdict,
        "",
        explanation.replace("     ", ""),
        "",
        "---",
        "",
        "## Üretilen Dosyalar",
        "",
        "| Dosya | Açıklama |",
        "|-------|----------|",
        "| `01_feature_dagilim_karsilastirma.csv` | Normal vs Anomaly median/IQR karşılaştırması |",
        "| `02_ks_testi_sonuclari.csv` | Kolmogorov-Smirnov test sonuçları |",
        "| `02_ks_top6_histograms.png` | En farklı 6 feature'ın dağılım histogramları |",
        "| `03_pca_projection.png` | PCA 2D görselleştirme |",
        "| `03_umap_projection.png` | UMAP 2D görselleştirme |",
        "| `04_knn_local_density.csv` | KNN mesafe karşılaştırması |",
        "| `04_knn_mesafe_dagilimi.png` | KNN mesafe dağılım histogramları |",
    ])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n  📄 Sonuç raporu: {report_path}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    ensure_output_dir()

    # Veri yükleme
    df = load_data()

    # Ön işleme (mevcut IForestBackdoorPreprocessor ile)
    X, y, feature_names = preprocess(df)

    # Analizler
    feat_df = analiz_1_feature_dagilimi(X, y, feature_names)
    ks_df = analiz_2_ks_testi(X, y, feature_names)
    analiz_3_pca_umap(X, y, feature_names)
    knn_df = analiz_4_knn_local_density(X, y, feature_names)

    # Genel sonuç
    genel_sonuc(feat_df, ks_df, knn_df)

    print("\n" + "=" * 70)
    print(f"  ✅ Tüm analizler tamamlandı! Sonuçlar: {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
