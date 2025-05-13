import torch
from torch import nn
import pytorch_lightning as pl
from models.positional_encoding import PositionalEncoding
from torchmetrics import MeanSquaredError, MeanAbsoluteError, Metric


class RelativeRegressionAccuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("acc_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total",   default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds, target):
        preds = preds.view(-1)
        target = target.view(-1)
        for p, t in zip(preds, target):
            score = 1.0 if (t == 0 and p == 0) else max(1 - torch.abs(p - t) / torch.abs(t), 0.0)
            self.acc_sum += score
            self.total   += 1

    def compute(self):
        return self.acc_sum / self.total


class TrackingTransformer(pl.LightningModule):
    def __init__(
        self,
        input_size: int = 4,
        d_model: int = 128,
        nhead: int = 4,
        dim_feedforward: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        lr: float = 1e-3,
        max_len: int = 200
    ):
        super().__init__()
        self.save_hyperparameters()
        hp = self.hparams

        # 1) Learned CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hp.d_model))

        # 2) Input projection + Positional Encoding
        self.input_proj   = nn.Linear(hp.input_size, hp.d_model)
        self.pos_encoder = PositionalEncoding(hp.d_model, dropout, max_len=hp.max_len)

        # 3) Transformer Encoder with final LayerNorm
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hp.d_model,
            nhead=hp.nhead,
            dim_feedforward=hp.dim_feedforward,
            dropout=dropout,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=hp.num_layers,
            norm=nn.LayerNorm(hp.d_model)
        )

        # 4) Dropout + Regressor Head
        self.dropout   = nn.Dropout(dropout)
        self.regressor = nn.Linear(hp.d_model, 1)

        # 5) Loss & Metrics
        self.loss_fn    = nn.MSELoss()
        self.train_mse  = MeanSquaredError()
        self.train_mae  = MeanAbsoluteError()
        self.train_acc  = RelativeRegressionAccuracy()
        self.val_mse    = MeanSquaredError()
        self.val_mae    = MeanAbsoluteError()
        self.val_acc    = RelativeRegressionAccuracy()
        self.test_mse   = MeanSquaredError()
        self.test_mae   = MeanAbsoluteError()
        self.test_acc   = RelativeRegressionAccuracy()

        # 6) Initialize weights
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, lengths):
        B, L, _ = x.size()
        # Mask for padded positions (+ CLS token)
        mask = torch.arange(L + 1, device=x.device).expand(B, L + 1) >= (lengths + 1).unsqueeze(1)

        # Input projection and prepend CLS
        x = self.input_proj(x)                       # (B, L, d_model)
        cls = self.cls_token.expand(B, -1, -1)       # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)               # (B, L+1, d_model)

        # Positional encoding
        x = self.pos_encoder(x)                      # (B, L+1, d_model)
        x = x.permute(1, 0, 2)                       # (L+1, B, d_model)

        # Transformer encoding
        out = self.transformer(x, src_key_padding_mask=mask)  # (L+1, B, d_model)
        h = out[0]                                    # CLS output (B, d_model)

        # Head
        h = self.dropout(h)
        y_hat = self.regressor(h).squeeze(-1)         # (B,)
        return y_hat

    def shared_step(self, batch, stage):
        x, y, lengths = batch
        y_hat = self(x, lengths)
        loss  = self.loss_fn(y_hat, y)
        getattr(self, f"{stage}_mse")(y_hat, y)
        getattr(self, f"{stage}_mae")(y_hat, y)
        getattr(self, f"{stage}_acc")(y_hat, y)
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_mse", getattr(self, f"{stage}_mse"), on_epoch=True)
        self.log(f"{stage}_mae", getattr(self, f"{stage}_mae"), on_epoch=True)
        self.log(f"{stage}_acc", getattr(self, f"{stage}_acc"), on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, 'val')

    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch, 'test')
        self.log('test_mse', self.test_mse, on_epoch=True)
        self.log('test_mae', self.test_mae, on_epoch=True)
        self.log('test_acc', self.test_acc, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y, lengths = batch
        y_hat = self(x, lengths)
        return y_hat, y

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=1e-5
        )
