"""
Bank ve CelebA Veri Setleri için UMAP Manifold Analizi ve Görselleştirme
========================================================================
- Tüm anomalileri (%100) dahil eder.
- Temsili ve uygun çoklukta normal örnek seçer (stratified sampling).
- Yüksek çözünürlüklü UMAP 2D izdüşüm grafiklerini üretir.
- Komşuluk / ayrışabilirlik analizini hesaplar.
"""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
from sklearn.neighbors import NearestNeighbors

# Çıktı Dizinleri
WORKSPACE_REPORT_DIR = "reports/umap"
ARTIFACT_DIR = r"C:\Users\GHOST\.gemini\antigravity-ide\brain\b655bc6f-8b70-471c-8d8f-e1e1f8888a17"
os.makedirs(WORKSPACE_REPORT_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Dataset Konfigürasyonları
DATASETS = [
    {
        "name": "Bank Additional (Full Normalised)",
        "file": "data/bank-additional-full_normalised.csv",
        "output_prefix": "bank_umap",
        "n_normal_samples": 15000,
        "n_neighbors": 30,
        "min_dist": 0.3,
        "metric": "euclidean"
    },
    {
        "name": "CelebA Bald vs Non-Bald (Normalised)",
        "file": "data/celeba_baldvsnonbald_normalised.csv",
        "output_prefix": "celeba_umap",
        "n_normal_samples": 18000,
        "n_neighbors": 30,
        "min_dist": 0.3,
        "metric": "hamming"  # CelebA ikili/binary (0/1) özelliklerden oluştuğu için hamming veya euclidean
    }
]

def run_umap_pipeline(cfg):
    print("=" * 70)
    print(f"  Analiz Başlatılıyor: {cfg['name']}")
    print("=" * 70)

    # 1. Veri Yükleme
    df = pd.read_csv(cfg["file"])
    y = df["class"].values
    X = df.drop(columns=["class"]).values.astype(np.float32)

    anomaly_mask = (y == 1)
    normal_mask = (y == 0)

    n_anomalies_total = np.sum(anomaly_mask)
    n_normals_total = np.sum(normal_mask)
    print(f"  Toplam Kayıt: {len(df):,} | Normal: {n_normals_total:,} | Anomali: {n_anomalies_total:,}")

    # 2. Örnekleme: Bütün anomalileri AL, normallerden uygun çoklukta örnekle
    anomaly_indices = np.where(anomaly_mask)[0]
    normal_indices = np.where(normal_mask)[0]

    n_normal_to_sample = min(cfg["n_normal_samples"], len(normal_indices))
    rng = np.random.RandomState(42)
    sampled_normal_indices = rng.choice(normal_indices, size=n_normal_to_sample, replace=False)

    sampled_indices = np.concatenate([sampled_normal_indices, anomaly_indices])
    rng.shuffle(sampled_indices)

    X_sampled = X[sampled_indices]
    y_sampled = y[sampled_indices]

    n_sampled_normal = np.sum(y_sampled == 0)
    n_sampled_anomaly = np.sum(y_sampled == 1)
    print(f"  UMAP Giriş Matrisi: {len(X_sampled):,} satır × {X_sampled.shape[1]} özellik")
    print(f"    - Bütün Anomaliler (%100): {n_sampled_anomaly:,}")
    print(f"    - Örneklenen Normaller:    {n_sampled_normal:,} (Temsil Oranı: {n_sampled_normal/n_normals_total:.1%})")

    # 3. UMAP Fit & Transform
    print(f"  [UMAP] (n_neighbors={cfg['n_neighbors']}, min_dist={cfg['min_dist']}) hesaplaniyor...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=cfg["n_neighbors"],
        min_dist=cfg["min_dist"],
        metric="euclidean",
        random_state=42,
        verbose=False
    )
    X_umap = reducer.fit_transform(X_sampled)
    print("  [OK] UMAP hesaplamasi tamamlandi!")

    # 4. Ayrışabilirlik / Komşuluk Ölçümü (KNN Overlap in UMAP Space)
    print("  [KNN] Komsuluk ayrisabilirligi hesaplaniyor...")
    knn = NearestNeighbors(n_neighbors=15).fit(X_umap)
    distances, indices = knn.kneighbors(X_umap)

    # Her anomalinin en yakın 15 komşusunun kaçı normal?
    u_anomaly_indices = np.where(y_sampled == 1)[0]
    anomaly_neighbor_labels = y_sampled[indices[u_anomaly_indices, 1:]]  # 0. indeks noktanın kendisi
    normal_neighbor_ratio_for_anomalies = np.mean(anomaly_neighbor_labels == 0)

    print(f"  [Sonuc] Anomalilerin en yakin komsularindaki Normal orani: {normal_neighbor_ratio_for_anomalies:.1%}")
    if normal_neighbor_ratio_for_anomalies > 0.6:
        print("     -> Anomaliler normal veri kumelerinin arasina gomulmus (IForest/Unsupervised modeller icin zor).")
    else:
        print("     -> Anomaliler belirgin sekilde kendi kumelerini/adaciklarini olusturmus (Ayrismasi kolay).")

    # 5. Görselleştirme
    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)

    # Normaller (Mavi)
    ax.scatter(
        X_umap[y_sampled == 0, 0],
        X_umap[y_sampled == 0, 1],
        c="#2980b9",
        alpha=0.25,
        s=8,
        label=f"Normal (Örneklem: n={n_sampled_normal:,} / Toplam: {n_normals_total:,})",
        zorder=1
    )

    # Anomaliler (Kırmızı)
    ax.scatter(
        X_umap[y_sampled == 1, 0],
        X_umap[y_sampled == 1, 1],
        c="#e74c3c",
        alpha=0.85,
        s=24,
        edgecolors="black",
        linewidths=0.35,
        label=f"Anomaly (%100 Tamamı: n={n_sampled_anomaly:,})",
        zorder=2
    )

    ax.set_title(
        f"UMAP 2D Projeksiyonu — {cfg['name']}\n"
        f"(Tüm {n_sampled_anomaly:,} Anomali + {n_sampled_normal:,} Temsili Normal Veri | Boyut: {X.shape[1]} Özellik)",
        fontsize=13,
        fontweight="bold",
        pad=15
    )
    ax.set_xlabel("UMAP-1 (Manifold Eksen 1)", fontsize=11, labelpad=8)
    ax.set_ylabel("UMAP-2 (Manifold Eksen 2)", fontsize=11, labelpad=8)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=10, loc="best", frameon=True, framealpha=0.9)

    plt.tight_layout()

    # Dosyaları kaydet
    out_png_name = f"{cfg['output_prefix']}_projection.png"
    workspace_path = os.path.join(WORKSPACE_REPORT_DIR, out_png_name)
    artifact_path = os.path.join(ARTIFACT_DIR, out_png_name)

    fig.savefig(workspace_path, dpi=150, bbox_inches="tight")
    fig.savefig(artifact_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  [Kaydedildi] Workspace: {workspace_path}")
    print(f"  [Kaydedildi] Artifact:  {artifact_path}\n")

    return {
        "name": cfg["name"],
        "total_rows": len(df),
        "n_features": X.shape[1],
        "n_anomalies": n_sampled_anomaly,
        "n_normals_total": n_normals_total,
        "n_normals_sampled": n_sampled_normal,
        "normal_neighbor_ratio": normal_neighbor_ratio_for_anomalies,
        "workspace_path": workspace_path,
        "artifact_path": artifact_path
    }

if __name__ == "__main__":
    results = []
    for cfg in DATASETS:
        res = run_umap_pipeline(cfg)
        results.append(res)
    print("=" * 70)
    print("  Tüm UMAP analizleri başarıyla tamamlandı!")
    print("=" * 70)
