import os
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from sklearn.decomposition import PCA

class ResultValidator:
    def __init__(self, output_dir: str = "reports"):
        """
        Anomali tespiti sonuÃ§larÄ±nÄ± doÄŸrulamak ve aÃ§Ä±klanabilir hale getirmek iÃ§in 
        kullanÄ±lan doÄŸrulama modÃ¼lÃ¼. Ã‡Ä±ktÄ±lar belirtilen klasÃ¶re kaydedilir.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def top_k_inspection(self, original_df: pd.DataFrame, result_df: pd.DataFrame, k: int = 15, model_name: str = None):
        """
        Modelin '1' (Anomali) olarak etiketlediÄŸi TÃœM satÄ±rlarÄ± orijinal veriden Ã§ekerek kaydeder.
        EÄŸer model bazlÄ± bir sorgu yoksa (ensemble), en az 1 oy alan tÃ¼m anomaliler kaydedilir.
        """
        if model_name:
            print(f"\n--- [1] Modelin Tespit EttiÄŸi TÃ¼m Anomaliler [{model_name}] ---")
            sort_col = f"score_{model_name}"
            label_col = f"label_{model_name}"
        else:
            print(f"\n--- [1] Modelin Tespit EttiÄŸi TÃ¼m Anomaliler [Ortalama Skor] ---")
            sort_col = "mean_anomaly_score"
            label_col = "anomaly_vote_count"
            
        if sort_col not in result_df.columns:
            sort_col = "anomaly_vote_count"
            
        # 1 (Anomali) olan tÃ¼m satÄ±rlarÄ± filtrele
        if label_col in result_df.columns and model_name:
            anomalies_df = result_df[result_df[label_col] == 1].sort_values(by=sort_col, ascending=False)
        else:
            anomalies_df = result_df[result_df["anomaly_vote_count"] > 0].sort_values(by=sort_col, ascending=False)
            
        anomaly_indices = anomalies_df["row_id"].values
        
        if len(anomaly_indices) == 0:
            print(f"[BÄ°LGÄ°] {model_name or 'Ensemble'} hiÃ§bir anomali tespit etmedi.")
            return
            
        # Orijinal veriden bu satÄ±rlarÄ± Ã§ek ve skorlarÄ± ekle
        top_k_original = original_df.loc[anomaly_indices].copy()
        top_k_original[sort_col] = anomalies_df[sort_col].values
        top_k_original["anomaly_vote_count"] = anomalies_df["anomaly_vote_count"].values
        
        print(f"[{model_name or 'Ensemble'}] Toplam {len(anomaly_indices)} adet anomali bulundu.")
        
        # Eğer gerçek etiket (Class veya label) varsa, kaç tanesini doğru bildiğimizi ekrana yazalım
        target_col = None
        if "Class" in top_k_original.columns:
            target_col = "Class"
        elif "label" in top_k_original.columns:
            target_col = "label"
            
        if target_col:
            true_anomalies = top_k_original[target_col].sum()
            print(f"  -> Tespit edilen {len(anomaly_indices)} anomaliden {int(true_anomalies)} tanesi GERÇEKTEN anomali (Class=1)!")
            
            # Görüntüleme sırasında hedefin ortada kaybolmaması için öne alıp sadece belli başlı kolonları gösterelim
            cols_to_show = [target_col, sort_col, "anomaly_vote_count"] 
            # İlk 5 orijinal feature'u da ekleyelim ki tablo çok uzun olmasın (veya ekrana sığsın)
            cols_to_show += [c for c in top_k_original.columns if c not in cols_to_show][:5]
            print(top_k_original[cols_to_show].head(k).to_string())
        else:
            # Ekrana yine ilk 15'ini basalım çok kalabalık olmasın
            print(top_k_original.head(k).to_string())
        
        # CSV olarak TÜMÜNÜ kaydet
        if target_col:
            # Sütunları yeniden sıralayıp gerçek etiketi en sona alalım
            cols = [c for c in top_k_original.columns if c != target_col]
            cols.append(target_col)
            top_k_original = top_k_original[cols]
            # Sütunun adını da belirginleştirelim
            top_k_original = top_k_original.rename(columns={target_col: f"GERCEK_ETIKET_{target_col}"})
            
        suffix = f"_{model_name}" if model_name else "_ensemble"
        out_path = os.path.join(self.output_dir, f"top_k_anomalies{suffix}.csv")
        top_k_original.to_csv(out_path, index=False, encoding="utf-8")
        print(f"Tüm anomali listesi kaydedildi: {out_path}")
        
    def ensemble_validation(self, result_df: pd.DataFrame):
        """
        FarklÄ± mantÄ±kla Ã§alÄ±ÅŸan modellerin aynÄ± satÄ±rda anomali bulma (konsensÃ¼s) durumunu raporlar.
        """
        print(f"\n--- [3] Ã‡oklu Model MutabakatÄ± (Ensemble Validation) ---")
        consensus = result_df[result_df["anomaly_vote_count"] >= 2]
        
        print(f"En az 2 modelin 'Anomali' dediÄŸi (Kesin Anomali) satÄ±r sayÄ±sÄ±: {len(consensus)}")
        
        if len(consensus) > 0:
            out_path = os.path.join(self.output_dir, "consensus_anomalies.csv")
            consensus.to_csv(out_path, index=False, encoding="utf-8")
            print(f"Kesin anomaliler kaydedildi: {out_path}")
            
    def spatial_check_pca(self, X: np.ndarray, result_df: pd.DataFrame, model_name: str = None, dataset_type: str = "tabular"):
        """
        Veriyi 2 boyuta indirgeyerek (PCA), anomalilerin gÃ¶rsel olarak ana veri bulutunun 
        dÄ±ÅŸÄ±nda, seyrek alanlarda kalÄ±p kalmadÄ±ÄŸÄ±nÄ± kontrol etmeyi saÄŸlar.
        """
        if model_name:
            print(f"\n--- [4] Boyut Ä°ndirgeme ile Uzamsal Kontrol (PCA) [{model_name}] ---")
        else:
            print(f"\n--- [4] Boyut Ä°ndirgeme ile Uzamsal Kontrol (PCA) [Ensemble] ---")
            
        if not MATPLOTLIB_AVAILABLE:
            print("[UYARI] Uzamsal kontrol iÃ§in 'matplotlib' kÃ¼tÃ¼phanesi gereklidir. (pip install matplotlib)")
            return
            
        try:
            from sklearn.preprocessing import StandardScaler
            
            # PCA varyansa (bÃ¼yÃ¼klÃ¼ÄŸe) son derece duyarlÄ±dÄ±r. 
            # Adult veri setindeki 'capital-gain' gibi devasa uÃ§ deÄŸerleri olan sÃ¼tunlar 
            # grafiÄŸi tek baÅŸlarÄ±na domine edip artÄ± (+) veya haÃ§ ÅŸekli (dikey/yatay Ã§izgi) oluÅŸturabilirler.
            # Bunu Ã¶nlemek ve tÃ¼m Ã¶zelliklerin grafiÄŸe eÅŸit daÄŸÄ±lmasÄ±nÄ± saÄŸlamak iÃ§in 
            # EÄŸer X bir sparse (seyrek) matris ise Ã¶nce dense matrise Ã§evirmeliyiz
            # Ã‡Ã¼nkÃ¼ StandardScaler(with_mean=True) sparse matrislerde hata verir.
            import scipy.sparse as sp
            if sp.issparse(X):
                X_viz_input = X.toarray()
            else:
                X_viz_input = X
                
            # AutoEncoder/VAE gibi ReLU kullanan modellerin Ã§Ä±ktÄ±larÄ± hep pozitiftir.
            # with_mean=True ile veriyi 0'a merkezlemezsek PCA "L" ÅŸeklinde hatalÄ± Ã§izim yapar.
            scaler = StandardScaler(with_mean=True)
            X_viz = scaler.fit_transform(X_viz_input)
            
            pca = PCA(n_components=2, random_state=42)
            X_2d = pca.fit_transform(X_viz)
            
            # Hangi modelin anomali kararlarÄ±na bakacaÄŸÄ±z?
            if model_name and f"label_{model_name}" in result_df.columns:
                is_anomaly = result_df[f"label_{model_name}"] == 1
                title = f"PCA Uzamsal KontrolÃ¼ [{model_name}]"
                suffix = f"_{model_name}"
            else:
                is_anomaly = result_df["anomaly_vote_count"] > 0
                title = "PCA Uzamsal KontrolÃ¼ [TÃ¼m Modeller / Ortak Karar]"
                suffix = "_ensemble"
            
            if dataset_type == "time_series":
                plt.figure(figsize=(14, 6))
                x_axis = np.arange(len(X_2d))
                y_axis = X_2d[:, 0]
                
                plt.plot(x_axis, y_axis, color='blue', linewidth=1, alpha=0.8, label='Zaman AkÄ±ÅŸÄ± (PCA-1)')
                
                ymin, ymax = plt.ylim()
                plt.fill_between(x_axis, ymin, ymax, where=is_anomaly, color='red', alpha=0.3, label='Anomali')
                plt.ylim(ymin, ymax)
                
                plt.xlabel("Zaman (SatÄ±r Ä°ndeksi)")
                plt.ylabel("PCA BileÅŸeni 1")
            else:
                plt.figure(figsize=(10, 8))
                # Normaller (Mavi, kÃ¼Ã§Ã¼k saydam noktalar)
                plt.scatter(X_2d[~is_anomaly, 0], X_2d[~is_anomaly, 1], c='blue', alpha=0.2, label='Normal', s=10)
                
                # Anomaliler (KÄ±rmÄ±zÄ±, belirgin noktalar)
                plt.scatter(X_2d[is_anomaly, 0], X_2d[is_anomaly, 1], c='red', alpha=0.8, label='Anomali', s=40, marker='x')
                
                plt.xlabel("PCA BileÅŸeni 1")
                plt.ylabel("PCA BileÅŸeni 2")
            
            plt.title(title)
            
            # AynÄ± label'larÄ±n tekrarÄ±nÄ± Ã¶nlemek iÃ§in legend filtreleme
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())
            
            plt.grid(True, alpha=0.3)
            
            out_path = os.path.join(self.output_dir, f"pca_spatial_check{suffix}.png")
            plt.savefig(out_path, bbox_inches='tight', dpi=150)
            plt.close()
            
            print(f"PCA uzamsal kontrol grafiÄŸi kaydedildi: {out_path}")
        except Exception as e:
            print(f"PCA analizinde bir hata oluÅŸtu: {e}")

    def evaluate_with_true_labels(self, result_df: pd.DataFrame, labels_path: str, model_names: list):
        """
        Zaman serisi (veya herhangi bir) veri setinde gerÃ§ek etiketler (true labels) varsa, 
        modellerin tahmin performansÄ±nÄ± deÄŸerlendirir.
        """
        import os
        if not os.path.exists(labels_path):
            return
            
        print("\n--- [5] GerÃ§ek Etiketlere GÃ¶re Performans DeÄŸerlendirmesi ---")
        try:
            from sklearn.metrics import classification_report, confusion_matrix
            import pandas as pd
            
            true_labels_df = pd.read_csv(labels_path)
            
            # Etiket sÃ¼tununu bul (ilk sÃ¼tun veya 'label' adlÄ± sÃ¼tun)
            label_col = "label" if "label" in true_labels_df.columns else true_labels_df.columns[0]
            y_true = true_labels_df[label_col].values
            
            if len(y_true) != len(result_df):
                print(f"[UYARI] GerÃ§ek etiket boyutu ({len(y_true)}) ile tahmin boyutu ({len(result_df)}) eÅŸleÅŸmiyor!")
                return
                
            report_lines = ["# Zaman Serisi Modelleri Performans Raporu\n"]
            report_lines.append(f"Toplam Veri SayÄ±sÄ±: {len(y_true)}")
            report_lines.append(f"GerÃ§ek Anomali SayÄ±sÄ±: {sum(y_true == 1)}")
            report_lines.append(f"Normal Veri SayÄ±sÄ±: {sum(y_true == 0)}\n")
            
            mismatched_or_anomalous = result_df.copy()
            mismatched_or_anomalous["true_label"] = y_true
            
            for model in model_names:
                pred_col = f"label_{model}"
                if pred_col in result_df.columns:
                    y_pred = result_df[pred_col].values
                    
                    cm = confusion_matrix(y_true, y_pred)
                    if cm.shape == (2, 2):
                        tn, fp, fn, tp = cm.ravel()
                    else:
                        tn, fp, fn, tp = 0, 0, 0, 0
                        
                    toplam_bulunan = tp + fp
                    gercek_anomali_sayisi = tp + fn
                    
                    report_lines.append(f"## {model.upper()} Performansı")
                    report_lines.append(f"- **Toplam Bulduğu Anomali Sayısı:** {toplam_bulunan}")
                    report_lines.append(f"- **Doğru Tespit Ettiği (True Positive):** {tp}")
                    report_lines.append(f"- **Yanlış Alarm (False Positive):** {fp}")
                    report_lines.append(f"- **Kaçırdığı Anomali (False Negative):** {fn}")
                    report_lines.append(f"- **Geri Çağırma (Recall):** %{100 * tp / gercek_anomali_sayisi if gercek_anomali_sayisi > 0 else 0:.2f}")
                    report_lines.append(f"- **Hassasiyet (Precision):** %{100 * tp / toplam_bulunan if toplam_bulunan > 0 else 0:.2f}\n")
                    
                    report_lines.append("```text")
                    report_lines.append(classification_report(y_true, y_pred, zero_division=0))
                    report_lines.append("```\n")
            
            md_path = os.path.join(self.output_dir, "timeseries_evaluation_report.md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write("\n".join(report_lines))
            print(f"Performans raporu kaydedildi: {md_path}")
            
            # Ä°lgi Ã§ekici satÄ±rlarÄ± (GerÃ§ekte anomali veya modelin anomali bulduÄŸu) CSV olarak kaydet
            mask = (mismatched_or_anomalous["true_label"] == 1) | (mismatched_or_anomalous["anomaly_vote_count"] > 0)
            interesting_rows = mismatched_or_anomalous[mask]
            
            csv_path = os.path.join(self.output_dir, "timeseries_predictions_vs_true.csv")
            interesting_rows.to_csv(csv_path, index=False, encoding="utf-8")
            print(f"Anomali bulunan/olmasÄ± gereken satÄ±rlarÄ±n detayÄ± kaydedildi: {csv_path}")
            
        except Exception as e:
            print(f"Performans deÄŸerlendirme sÄ±rasÄ±nda hata oluÅŸtu: {e}")

    def plot_score_distributions(self, result_df: pd.DataFrame, model_names: list, top_k: int = 2000):
        """
        Modellerin Ã¼rettiÄŸi ham anomali skorlarÄ±nÄ± (decision_scores) bÃ¼yÃ¼kten kÃ¼Ã§Ã¼ÄŸe sÄ±ralayarak
        Elbow (KÄ±rÄ±lma) noktasÄ±nÄ± bulmak iÃ§in daÄŸÄ±lÄ±m grafiÄŸini Ã§izer.
        """
        if not MATPLOTLIB_AVAILABLE:
            return
            
        print("\n--- [6] Anomali SkorlarÄ± DaÄŸÄ±lÄ±mÄ± (Elbow Metodu) ---")
        
        for model in model_names:
            score_col = f"score_{model}"
            if score_col in result_df.columns:
                raw_scores = result_df[score_col].values
                sorted_scores = np.sort(raw_scores)[::-1]
                limit = min(top_k, len(sorted_scores))
                
                plt.figure(figsize=(10, 5))
                plt.plot(sorted_scores[:limit], color="blue", linewidth=2)
                
                plt.title(f"Anomali SkorlarÄ± DaÄŸÄ±lÄ±mÄ± [{model.upper()}] (Elbow Metodu)")
                plt.xlabel("SÄ±ralanmÄ±ÅŸ Veri Ä°ndeksi (En anomaliden en normale)")
                plt.ylabel("Anomali Skoru (Decision Score)")
                plt.grid(True, linestyle="--", alpha=0.6)
                
                out_path = os.path.join(self.output_dir, f"elbow_plot_{model}.png")
                plt.savefig(out_path, bbox_inches="tight", dpi=150)
                plt.close()
                print(f"Elbow grafiÄŸi kaydedildi: {out_path}")

