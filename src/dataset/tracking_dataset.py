import numpy as np
import torch
from torch.utils.data import Dataset


class TrackingDataset(Dataset):
    def __init__(self, data_list):
        self.samples = []
        for d in data_list:
            features = d.get('features', [])
            speed = d.get('speed', None)
            length = len(features)
            self.samples.append((torch.tensor(features, dtype=torch.float32),
                                 torch.tensor(speed, dtype=torch.float32),
                                 length))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
