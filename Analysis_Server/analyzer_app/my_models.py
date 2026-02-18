import torch
import torch.nn as nn, pytorch_lightning as pl
import torchmetrics
from torchmetrics.classification import (
    BinaryAccuracy, BinaryPrecision, BinaryRecall,
    BinaryF1Score, BinaryAUROC
)
from torchmetrics import MetricCollection

from analyzer_app.char_vocab import char2idx, PAD


class CharCNN(pl.LightningModule):
    def __init__(self, emb_dim=128, n_filters=128, dropout=0.3, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        vocab_size = len(char2idx) + 2
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)

        conv_layers = []
        for _ in range(8):
            conv_layers += [
                nn.Conv1d(emb_dim, n_filters, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(n_filters)
            ]
            emb_dim = n_filters
        self.conv = nn.Sequential(*conv_layers)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.Linear(n_filters, 1)
        )

        self.crit = nn.BCEWithLogitsLoss()

        base_metrics = MetricCollection({
            "Acc": BinaryAccuracy(),
            "Prec": BinaryPrecision(),
            "Rec": BinaryRecall(),
            "f1": BinaryF1Score(),
            "AUC": BinaryAUROC()
        })

        self.train_metrics = base_metrics.clone(prefix="train_")
        self.val_metrics = base_metrics.clone(prefix="val_")
        self.test_metrics = base_metrics.clone(prefix="test_")

    def forward(self, x):
        x = self.emb(x).transpose(1, 2)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x).squeeze(-1)

    def _shared_step(self, batch, stage):
        x, y = batch
        logits = self(x)
        loss = self.crit(logits, y)

        preds = torch.sigmoid(logits)
        metric_set = getattr(self, f"{stage}_metrics")

        metrics = metric_set(preds, y.long())
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log_dict(metrics, prog_bar=True, on_epoch=True, on_step=False)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def on_train_epoch_end(self):
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        self.val_metrics.reset()

    def on_test_epoch_end(self):
        self.test_metrics.reset()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,
                                                         mode="min",
                                                         patience=3)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "monitor": "val_loss"
            }
        }

class LitPhishingLSTM(pl.LightningModule):
    def __init__(self, embedding_matrix, hidden_dim=64,
                 num_layers=1, dropout=0.35, pad_idx=0, lr=1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=["embedding_matrix"])

        weights = torch.tensor(embedding_matrix)
        self.embedding = nn.Embedding.from_pretrained(
            weights, freeze=False, padding_idx=pad_idx
        )
        self.emb_drop = nn.Dropout1d(0.2)

        self.lstm = nn.LSTM(
            input_size=weights.shape[1],
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc = nn.Linear(hidden_dim * 2, 1)

        self.criterion = nn.BCEWithLogitsLoss()

        self.train_acc = torchmetrics.Accuracy(threshold=0.5, task="binary")

        self.val_acc = torchmetrics.Accuracy(threshold=0.5, task="binary")
        self.val_precision = torchmetrics.Precision(threshold=0.5, task="binary")
        self.val_rec = torchmetrics.Recall(threshold=0.5, task="binary")
        self.val_f1 = torchmetrics.F1Score(threshold=0.5, task="binary")

        self.test_acc = torchmetrics.Accuracy(threshold=0.5, task="binary")
        self.test_precision = torchmetrics.Precision(threshold=0.5, task="binary")
        self.test_recall = torchmetrics.Recall(threshold=0.5, task="binary")
        self.test_f1 = torchmetrics.F1Score(threshold=0.5, task="binary")

    def forward(self, x, lengths):
        emb = self.emb_drop(self.embedding(x).transpose(1, 2)).transpose(1, 2)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        pooled, _ = torch.max(out, 1)
        return self.fc(pooled).squeeze(1)

    def _common_step(self, batch):
        x, lengths, y = batch
        logits = self(x, lengths)
        loss = self.criterion(logits, y)
        probability = torch.sigmoid(logits)
        return loss, probability, y

    def training_step(self, batch, batch_idx):
        loss, probs, y = self._common_step(batch)
        self.train_acc.update(probs, y)
        self.log("train_loss", loss,
                 prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def on_train_epoch_end(self):
        self.log("train_acc", self.train_acc.compute(),
                 prog_bar=True, on_step=False, on_epoch=True)
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        loss, probs, y = self._common_step(batch)
        self.val_acc.update(probs, y)
        self.val_precision.update(probs, y)
        self.val_rec.update(probs, y)
        self.val_f1.update(probs, y)
        self.log("val_loss", loss, prog_bar=False, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        self.log_dict({
            "val_acc": self.val_acc.compute(),
            "val_precision": self.val_precision.compute(),
            "val_rec": self.val_rec.compute(),
            "val_f1": self.val_f1.compute(),
        }, prog_bar=True)
        self.val_acc.reset()
        self.val_precision.reset()
        self.val_rec.reset()
        self.val_f1.reset()

    def test_step(self, batch, batch_idx):
        loss, probs, y = self._common_step(batch)
        self.test_acc.update(probs, y)
        self.test_precision.update(probs, y)
        self.test_recall.update(probs, y)
        self.test_f1.update(probs, y)
        self.log("test_loss", loss, prog_bar=False)

    def on_test_epoch_end(self):
        self.log_dict({
            "test_acc": self.test_acc.compute(),
            "test_precision": self.test_precision.compute(),
            "test_rec": self.test_recall.compute(),
            "test_f1": self.test_f1.compute(),
        }, prog_bar=True)
        self.test_acc.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_f1.reset()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min",
                                                           factor=0.5, patience=2)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched,
                                                   "monitor": "val_loss"}}

