import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from dataset.tracking_datamodule import TrackingDataModule
from models.lstm_model import TrackingLSTM
from tqdm import tqdm


class CustomProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self) -> tqdm:
        bar = super().init_validation_tqdm()
        if bar is not None:
            bar.disable = True
        return bar


def train_lstm_pipeline(data_list, train_idx, val_idx, test_idx, **config):
    # DataModule
    data_module = TrackingDataModule(
        data_list=data_list,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        batch_size=config.get('batch_size', 4),
        split_ratio=config.get('split_ratio', (0.8, 0.1, 0.1)),
        num_workers=config.get('num_workers', 4)
    )

    # Modelo
    model = TrackingLSTM(
        input_size=config.get('input_size', 4),
        hidden_size=config.get('hidden_size', 64),
        num_layers=config.get('num_layers', 2),
        output_size=config.get('output_size', 1),
        lr=config.get('learning_rate', 1e-3),
        dropout=config.get('dropout', 0.1),
        weight_decay=config.get('weight_decay', 0.0)
    )

    # Callbacks y logger
    logger = CSVLogger("logs/", name=config.get('log_path', "tracking_lstm"))
    callbacks = [
        ModelCheckpoint(save_top_k=1, mode='min', monitor='val_loss'),
        EarlyStopping(monitor='val_loss', patience=config.get('patience', 7))
    ]
    if config.get('progress_bar'):
        callbacks.append(CustomProgressBar())

    trainer = Trainer(
        max_epochs=config.get('epochs', 100),
        callbacks=callbacks,
        accelerator="auto",
        devices="auto",
        logger=logger,
        deterministic=False,
        log_every_n_steps=config.get('log_every_n_steps', 10),
        enable_progress_bar=config.get('progress_bar', True),
        enable_model_summary=config.get('model_summary', True)
    )
    trainer.fit(model, datamodule=data_module)
    result = trainer.validate(model, datamodule=data_module)

    return data_module, model, trainer, logger, result
