import torch
from torch.utils.data import Dataset
import torch.nn.utils.rnn as rnn_utils


class UrlDataset(Dataset):
    def __init__(self, encodings, labels, max_length=200):
        self.enc = encodings
        self.lab = labels
        self.max = max_length

    def __len__(self):
        return len(self.lab)

    def __getitem__(self, i):
        seq = torch.tensor(self.enc[i][:self.max], dtype=torch.long)
        y = torch.tensor(self.lab[i], dtype=torch.float32)
        return seq, y


def collate_fn(batch):
    seqs, labels = zip(*batch)
    padded = rnn_utils.pad_sequence(
        seqs, batch_first=True, padding_value=0
    )
    labels = torch.stack(labels)
    return padded, labels
