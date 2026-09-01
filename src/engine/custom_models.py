# custom_models.py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pyod.models.base import BaseDetector
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph

try:
    from torch_geometric.data import Data
    from pygod import DOMINANT
    PYGOD_AVAILABLE = True
except ImportError:
    PYGOD_AVAILABLE = False

# =================================================================
# Zaman Serisi iÃ§in Kayan Pencereli AutoEncoder / VAE SarmalayÄ±cÄ±larÄ±
# =================================================================
# PyOD'nin standart AutoEncoder ve VAE modelleri 2D tabular girdi bekler.
# Zaman serisi verisinde her satÄ±rÄ± baÄŸÄ±msÄ±z deÄŸerlendirmek temporal
# baÄŸlamÄ± kaybettirir. Bu sarmalayÄ±cÄ±lar:
#   1. Kayan pencere (sliding window) ile geÃ§miÅŸ bilgiyi yakalar
#   2. Pencereyi dÃ¼zleÅŸtirir (flatten) â†’ (N, seq_len * n_features)
#   3. Bu geniÅŸletilmiÅŸ vektÃ¶rÃ¼ PyOD modeline besler
#   4. Model, normal zaman pencerelerini yeniden oluÅŸturmayÄ± Ã¶ÄŸrenir
#   5. Yeniden oluÅŸturamadÄ±ÄŸÄ± pencereler â†’ anomali
# =================================================================





# =================================================================
# 1. PyTorch LSTM AutoEncoder AÄŸ Mimarisi
# =================================================================

class _LSTMAutoEncoderModule(nn.Module):
    """Encoder-Decoder LSTM Derin Ã–ÄŸrenme KatmanÄ±"""
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.seq_len = None  # forward anÄ±nda dinamik belirlenir
        self.hidden_dim = hidden_dim
        
        # Encoder: (Batch, Seq_Len, Input_Dim) -> (Batch, Hidden_Dim)
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Decoder: (Batch, Seq_Len, Hidden_Dim) -> (Batch, Seq_Len, Input_Dim)
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # 1. Kodlama (Encoding)
        _, (hidden, _) = self.encoder(x)
        latent_vector = hidden[-1]  # Son katmanÄ±n gizli durumu (Batch, Hidden_Dim)
        
        # 2. DarboÄŸazÄ± Zaman Boyutuna GeniÅŸletme (Repeat)
        repeated = latent_vector.unsqueeze(1).repeat(1, seq_len, 1)
        
        # 3. Geri OluÅŸturma (Decoding)
        dec_out, _ = self.decoder(repeated)
        reconstructed = self.output_layer(dec_out)
        
        return reconstructed, latent_vector


# =================================================================
# 2. PyOD Uyumlu Anomali DedektÃ¶rÃ¼ SÄ±nÄ±fÄ±
# =================================================================

class LSTMAutoEncoder(BaseDetector):
    """
    Zaman serisi iÃ§in kayan pencereli LSTM-AutoEncoder.
    Girdideki satÄ±r sayÄ±sÄ±nÄ± kaybetmemek iÃ§in baÅŸlangÄ±Ã§ indekslerine Ã¶n-dolgu (padding) yapar.
    """
    def __init__(
        self,
        seq_len: int = 10,
        hidden_dim: int = 32,
        num_layers: int = 2,
        epochs: int = 40,
        batch_size: int = 64,
        lr: float = 0.0001,
        contamination: float = 0.05,
        invert_score: bool = False,
        device: str = None,
        random_state: int = 42
    ):
        super().__init__(contamination=contamination)
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.invert_score = invert_score
        self.random_state = random_state
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.embeddings_ = None

    def _create_sliding_windows(self, X: np.ndarray) -> np.ndarray:
        """
        2D veriyi (N, Features) -> 3D pencereli tensÃ¶re (N, Seq_Len, Features) Ã§evirir.
        N satÄ±r sayÄ±sÄ±nÄ± korumak ve merkezi korumak iÃ§in geÃ§miÅŸ/gelecek dolgusu (padding) uygular.
        """
        n_samples, n_features = X.shape
        if n_samples < self.seq_len:
            raise ValueError(f"SatÄ±r sayÄ±sÄ± ({n_samples}) pencere boyutundan ({self.seq_len}) kÃ¼Ã§Ã¼k olamaz.")

        past_pad = self.seq_len // 2
        future_pad = self.seq_len - 1 - past_pad
        
        pad_past = np.repeat(X[:1], past_pad, axis=0)
        pad_future = np.repeat(X[-1:], future_pad, axis=0)
        padded_X = np.vstack([pad_past, X, pad_future])

        # Kayan pencereleri oluÅŸtur (Merkezli pencere)
        windows = []
        for i in range(n_samples):
            window = padded_X[i : i + self.seq_len]
            windows.append(window)

        return np.array(windows, dtype=np.float32)

    def fit(self, X, y=None):
        """Modeli eÄŸitir ve satÄ±r bazlÄ± rekonstrÃ¼ksiyon hatasÄ±nÄ± kaydeder."""
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # 1. 3D Pencereleme DÃ¶nÃ¼ÅŸÃ¼mÃ¼
        X_3d = self._create_sliding_windows(np.asarray(X, dtype=np.float32))
        n_samples, seq_len, input_dim = X_3d.shape

        # 2. Modeli BaÅŸlat
        self.model = _LSTMAutoEncoderModule(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss(reduction="none")

        dataset = TensorDataset(torch.from_numpy(X_3d))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # 3. EÄŸitim DÃ¶ngÃ¼sÃ¼
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for (batch_x,) in dataloader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                reconstructed, _ = self.model(batch_x)
                loss = criterion(reconstructed, batch_x).mean()
                loss.backward()
                # LSTM'in aniden sapıtmasını engellemek için Gradient Clipping eklendi
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            print(f"      [LSTMAutoEncoder] Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.6f}")

        # 4. Anomali SkorlarÄ±nÄ± ve Embedding'leri Ã‡Ä±karma
        self.model.eval()
        with torch.no_grad():
            eval_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
            all_scores = []
            all_embeddings = []

            for (batch_x,) in eval_loader:
                batch_x = batch_x.to(self.device)
                reconstructed, latent = self.model(batch_x)
                
                # Penceredeki son zaman adÄ±mÄ±nÄ±n MSE hatasÄ± (Point anomaly loss)
                step_error = criterion(reconstructed[:, -1, :], batch_x[:, -1, :]).mean(dim=1)
                
                all_scores.append(step_error.cpu().numpy())
                all_embeddings.append(latent.cpu().numpy())

        self.decision_scores_ = np.concatenate(all_scores)
        if getattr(self, 'invert_score', False):
            self.decision_scores_ = -self.decision_scores_
            
        self.embeddings_ = np.concatenate(all_embeddings, axis=0)

        # PyOD eÅŸik deÄŸeri ve label belirleme
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        """Yeni veriler iÃ§in anomali skoru Ã¼retir (Inference)."""
        self.model.eval()
        X_3d = self._create_sliding_windows(np.asarray(X, dtype=np.float32))
        dataset = TensorDataset(torch.from_numpy(X_3d))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        criterion = nn.MSELoss(reduction="none")

        scores = []
        with torch.no_grad():
            for (batch_x,) in dataloader:
                batch_x = batch_x.to(self.device)
                reconstructed, _ = self.model(batch_x)
                step_error = criterion(reconstructed[:, -1, :], batch_x[:, -1, :]).mean(dim=1)
                scores.append(step_error.cpu().numpy())

        final_scores = np.concatenate(scores)
        if getattr(self, 'invert_score', False):
            final_scores = -final_scores
        return final_scores


class GraphDominantDetector(BaseDetector):
    """
    PyGOD DOMINANT modelini PyOD BaseDetector arayÃ¼zÃ¼ne uyarlayan sÄ±nÄ±f.
    Hem graf nesnesi (torch_geometric.data.Data) hem de k-NN ile tÃ¼retilmiÅŸ 
    Ã¶znitelik matrislerini kabul eder.
    """
    def __init__(
        self,
        hid_dim: int = 64,
        num_layers: int = 4,
        epoch: int = 50,
        lr: float = 0.005,
        weight: float = 0.5,  # 0.5: YapÄ±sal hata ve Ã–znitelik hatasÄ±nÄ± eÅŸit aÄŸÄ±rlar
        k_neighbors: int = 5, # EÄŸer edge_index verilmezse k-NN ile graf Ã¼retilir
        contamination: float = 0.05,
        device: str = None,
        random_state: int = 42
    ):
        super().__init__(contamination=contamination)
        if not PYGOD_AVAILABLE:
            raise ImportError(
                "PyGOD veya PyTorch Geometric kurulu deÄŸil. "
                "LÃ¼tfen 'pip install pygod torch_geometric' Ã§alÄ±ÅŸtÄ±rÄ±n."
            )

        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.epoch = epoch
        self.lr = lr
        self.weight = weight
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.embeddings_ = None

    def _to_pyg_data(self, X) -> Data:
        """Girdiyi PyG Data formatÄ±na Ã§evirir."""
        if isinstance(X, Data):
            return X

        # EÄŸer X standart (N, Features) matrisi ise k-NN komÅŸuluk grafÄ± kur
        X_arr = np.asarray(X, dtype=np.float32)
        n_samples = X_arr.shape[0]

        adj = kneighbors_graph(
            X_arr, 
            n_neighbors=min(self.k_neighbors, n_samples - 1), 
            mode="connectivity", 
            include_self=False
        ).tocoo()

        edge_index = torch.tensor(
            np.vstack((adj.row, adj.col)), 
            dtype=torch.long
        )
        x_tensor = torch.tensor(X_arr, dtype=torch.float32)

        return Data(x=x_tensor, edge_index=edge_index)

    def fit(self, X, y=None):
        data = self._to_pyg_data(X)
        gpu_id = 0 if self.device == "cuda" or (isinstance(self.device, str) and "cuda" in self.device) else -1

        # PyGOD DOMINANT baÅŸlatma
        self.model = DOMINANT(
            hid_dim=self.hid_dim,
            num_layers=self.num_layers,
            epoch=self.epoch,
            lr=self.lr,
            weight=self.weight,
            contamination=self.contamination,
            gpu=gpu_id,
            verbose=0
        )

        # Modeli eÄŸit
        self.model.fit(data)

        # SkorlarÄ± ve etiketleri al
        self.decision_scores_ = self.model.decision_score_
        self.labels_ = self.model.label_

        # Modelin Ã¼rettiÄŸi dÃ¼ÄŸÃ¼m gÃ¶mmelerini (latent representations) al
        if hasattr(self.model, "emb") and self.model.emb is not None:
            self.embeddings_ = self.model.emb.detach().cpu().numpy()
        else:
            # Gömme katmanı doğrudan erişilemiyorsa düğüm özniteliklerini sakla
            self.embeddings_ = data.x.detach().cpu().numpy()

        self._process_decision_scores()
        return self

    def decision_function(self, X):
        data = self._to_pyg_data(X)
        scores, _ = self.model.predict(data, return_confidence=False)
        return scores
