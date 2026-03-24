import torch
from torch.utils.data import Dataset

FEATURES = [
    "return_1d",
    "return_5d",
    "return_10d",
    "ma_gap_5",
    "ma_gap_10",
    "volatility_10",
    "momentum_5",
    "volume_change",
    "volume_spike",
]

class TimeSeriesDataset(Dataset):
    def __init__(self, df, seq_len):
        self.seq_len = seq_len
        self.features = df[FEATURES].values
        self.targets = df["target"].values
    def __len__(self):
        return len(self.features) - self.seq_len
    
    def __getitem__(self, idx):
        x = self.features[idx : idx + self.seq_len]
        y = self.targets[idx + self.seq_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

