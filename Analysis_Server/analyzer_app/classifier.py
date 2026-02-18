# analyzer/classifier.py
import torch
from pathlib import Path
import numpy as np
from .my_models import LitPhishingLSTM, CharCNN

BASE = Path(__file__).parent.parent / 'models'
embedding_matrix = np.load(BASE / 'embedding_matrix.npy')

class MessageClassifier:
    """
    Wrapper for LitPhishingLSTM LightningModule for inference.
    """
    def __init__(self, ckpt_path: Path, device: str = 'cpu'):
        self.device = torch.device(device)
        # Load model from checkpoint, passing embedding matrix
        self.model = LitPhishingLSTM.load_from_checkpoint(
            str(ckpt_path),
            embedding_matrix=embedding_matrix
        )
        self.model.to(self.device).eval()

    def predict(self, tokens_lengths):
        tokens, lengths = tokens_lengths
        tokens = tokens.to(self.device).unsqueeze(0)
        lengths = lengths.to(self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(tokens, lengths)
            p_phish = torch.sigmoid(logits)[0]  # scalar
            label = int(p_phish >= 0.5)

            conf = float(p_phish) if label == 1 else float(1.0 - p_phish)
        return label, conf

class UrlClassifier:
    """
    Wrapper for CharCNN LightningModule for inference.
    """
    def __init__(self, ckpt_path: Path, device: str = 'cpu'):
        self.device = torch.device(device)
        self.model = CharCNN.load_from_checkpoint(str(ckpt_path))
        self.model.to(self.device).eval()

    def predict(self, seq_tensor):

        tensor = seq_tensor.to(self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(tensor)
            p_phish = torch.sigmoid(logits)[0]
            label = int(p_phish >= 0.5)
            conf = float(p_phish) if label == 1 else float(1.0 - p_phish)
        return label, conf

msg_clf = MessageClassifier(BASE / 'best-phishing-lstm.ckpt')
url_clf = UrlClassifier(  BASE / 'best-phishing-url.ckpt')
