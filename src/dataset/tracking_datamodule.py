import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from torch.utils.data.dataset import random_split
from dataset.tracking_dataset import TrackingDataset
from torch.nn.utils.rnn import pad_sequence


def collate_fn_padding(batch):
    batch.sort(key=lambda x: x[2], reverse=True)
    seqs, speeds, lengths = zip(*batch)
    padded = pad_sequence(seqs, batch_first=True, padding_value=0.0)
    return padded, torch.stack(speeds), torch.tensor(lengths, dtype=torch.long)


class ScaledDataset(Dataset):
    def __init__(self, dataset: Dataset, scaler: StandardScaler):
        self.dataset = dataset
        self.scaler = scaler

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        features, speed, length = self.dataset[idx]
        features_np = features.numpy()
        features_scaled = torch.tensor(
            self.scaler.transform(features_np), dtype=torch.float32
        )
        return features_scaled, speed, length


class TrackingDataModule(LightningDataModule):
    def __init__(self, data_list, batch_size, split_ratio, num_workers=4, scale=True):
        super().__init__()
        self.data_list = data_list
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        self.num_workers = num_workers
        self.scale = scale
        self.scaler = None

    def setup(self, stage=None):
        full_ds = TrackingDataset(self.data_list)
        total = len(full_ds)
        train_n = int(total * self.split_ratio[0])
        val_n   = int(total * self.split_ratio[1])
        test_n  = total - train_n - val_n
        train_ds, val_ds, test_ds = random_split(full_ds, [train_n, val_n, test_n])

        if self.scale:
            features_list = [x.numpy() for x, _, _ in train_ds]
            all_features = np.concatenate(features_list, axis=0)
            self.scaler = StandardScaler().fit(all_features)
            train_ds = ScaledDataset(train_ds, self.scaler)
            val_ds = ScaledDataset(val_ds, self.scaler)
            test_ds = ScaledDataset(test_ds, self.scaler)

        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          batch_size=self.batch_size,
                          shuffle=True,
                          collate_fn=collate_fn_padding,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds,
                          batch_size=self.batch_size,
                          shuffle=False,
                          collate_fn=collate_fn_padding,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds,
                          batch_size=self.batch_size,
                          shuffle=False,
                          collate_fn=collate_fn_padding,
                          num_workers=self.num_workers)
