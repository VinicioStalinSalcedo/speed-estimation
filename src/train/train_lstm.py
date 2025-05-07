import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from dataset.tracking_datamodule import TrackingDataModule
from models.lstm_model import TrackingLSTM


def train_lstm_pipeline(data_list):

    # DataModule
    data_module = TrackingDataModule(
        data_list=data_list,
        batch_size=4,
        split_ratio=(0.8, 0.1, 0.1),
        num_workers=4
    )

    # Modelo
    model = TrackingLSTM(
        input_size=4,
        hidden_size=64,
        num_layers=2,
        output_size=1,
        lr=0.0001,
        dropout=0.0
    )

    # Callbacks y logger
    logger = CSVLogger("logs/", name="tracking_lstm")
    checkpoint_cb = ModelCheckpoint(save_top_k=1, mode='min', monitor='val_loss')
    earlystop_cb = EarlyStopping(monitor='val_loss', patience=7)
    # tqdm_cb = TQDMProgressBar(leave=True)

    trainer = Trainer(
        max_epochs=200,
        callbacks=[
            checkpoint_cb,
            earlystop_cb,
            # tqdm_cb
        ],
        accelerator="auto",
        devices="auto",
        logger=logger,
        deterministic=False,
        log_every_n_steps=10,
    )
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)
