import torch
from torch.utils.data import Dataset
import torch.nn.utils.rnn as torch_rnn


class PhishingDataset(Dataset):
    def __init__(self, encodings, labels, max_length=200):
        self.enc = encodings
        self.lab = labels
        self.max = max_length
        self.lengths = [min(len(seq), max_length) for seq in encodings]

    def __len__(self):
        return len(self.enc)

    def __getitem__(self, idx_arg):
        seq = self.enc[idx_arg][: self.max]
        y = torch.tensor(self.lab[idx_arg], dtype=torch.float32).unsqueeze(0)
        return torch.tensor(seq, dtype=torch.long), self.lengths[idx_arg], y


def collate_fn(batch):
    seqs, lens, labels = zip(*batch)
    padded = torch_rnn.pad_sequence(seqs, batch_first=True,
                                    padding_value=0)
    lens = torch.tensor(lens, dtype=torch.long).to(padded.device)
    labels = torch.cat(labels)
    return padded, lens, labels
