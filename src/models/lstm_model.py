import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import torchmetrics
from torchmetrics import Metric
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau


class RelativeRegressionAccuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("acc_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total",   default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds, target):
        preds = preds.view(-1)
        target= target.view(-1)
        for p, t in zip(preds, target):
            score = 1.0 if (t == 0 and p == 0) else max(1 - abs(p - t) / abs(t), 0.0)
            self.acc_sum += score
            self.total   += 1

    def compute(self):
        return self.acc_sum / self.total


class TrackingLSTM(pl.LightningModule):
    def __init__(self, input_size=4, hidden_size=50, num_layers=2,
                 output_size=1, lr=1e-4, threshold=5.0, dropout=0.3, weight_decay=0.0):
        super().__init__()
        self.save_hyperparameters()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden_size, output_size)
        self.loss_fn = nn.MSELoss()

        # Métricas
        self.train_mse  = torchmetrics.MeanSquaredError()
        self.train_mae  = torchmetrics.MeanAbsoluteError()
        self.train_acc  = RelativeRegressionAccuracy()

        self.val_mse    = torchmetrics.MeanSquaredError()
        self.val_mae    = torchmetrics.MeanAbsoluteError()
        self.val_acc    = RelativeRegressionAccuracy()

        self.test_mse   = torchmetrics.MeanSquaredError()
        self.test_mae   = torchmetrics.MeanAbsoluteError()
        self.test_acc   = RelativeRegressionAccuracy()

    def forward(self, x, lengths):
        # Empaquetar secuencias
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        _, (hn, _) = self.lstm(packed)
        out = self.dropout(hn[-1])         # [batch, hidden]
        return self.fc(out).squeeze(-1)    # → [batch]

    def shared_step(self, batch, stage):
        x, y, lengths = batch
        y_hat = self(x, lengths)
        loss  = self.loss_fn(y_hat, y)

        # Actualizar métricas
        getattr(self, f"{stage}_mse")(y_hat, y)
        getattr(self, f"{stage}_mae")(y_hat, y)
        getattr(self, f"{stage}_acc")(y_hat, y)
        # Log
        if stage == 'test':
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_mse",  getattr(self, f"{stage}_mse"), on_epoch=True, )
            self.log(f"{stage}_mae",  getattr(self, f"{stage}_mae"), on_epoch=True, )
            self.log(f"{stage}_acc",  getattr(self, f"{stage}_acc"), on_epoch=True, )
        else:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_mse",  getattr(self, f"{stage}_mse"), on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log(f"{stage}_mae",  getattr(self, f"{stage}_mae"), on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log(f"{stage}_acc", getattr(self, f"{stage}_acc"), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, 'test')

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y, lengths = batch
        y_hat = self(x, lengths)
        return y_hat, y

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.hparams.lr,
                                     weight_decay=self.hparams.weight_decay
                                     )
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]
