import pytorch_lightning as pl
from torch import nn
import torch


class TrackingTransformer(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters()
        # Definir transformer y cabezas
        self.loss_fn = nn.MSELoss()

    def forward(self, x, lengths=None):
        # Implementar forward
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
