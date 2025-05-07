from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from dataset.tracking_dataset import TrackingDataset
from torch.nn.utils.rnn import pad_sequence


def collate_fn_padding(batch):
    batch.sort(key=lambda x: x[2], reverse=True)
    seqs, speeds, lengths = zip(*batch)
    padded = pad_sequence(seqs, batch_first=True, padding_value=0.0)
    return padded, torch.stack(speeds), torch.tensor(lengths, dtype=torch.long)


class TrackingDataModule(LightningDataModule):
    def __init__(self, data_list, batch_size, split_ratio, num_workers=4):
        super().__init__()
        self.data_list = data_list
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        self.num_workers = num_workers

    def setup(self, stage=None):
        ds = TrackingDataset(self.data_list)
        total = len(ds)
        train_n = int(total * self.split_ratio[0])
        val_n   = int(total * self.split_ratio[1])
        test_n  = total - train_n - val_n
        self.train_ds, self.val_ds, self.test_ds = random_split(ds, [train_n, val_n, test_n])

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
