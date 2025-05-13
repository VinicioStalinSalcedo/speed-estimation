import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from dataset.tracking_datamodule import TrackingDataModule
from models.lstm_model import TrackingLSTM
from tqdm import tqdm


class CustomProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self) -> tqdm:
        """
        Sobrescribe este método para devolver una barra tqdm deshabilitada
        para el bucle de validación.
        """
        bar = super().init_validation_tqdm()
        # Deshabilita la barra; no se mostrará ni actualizará.
        # La barra de épocas y la de entrenamiento (si refresh_rate > 0) seguirán funcionando.
        if bar is not None: # La barra podría no crearse si refresh_rate es 0 en la clase base
            bar.disable = True
        return bar


def train_lstm_pipeline(data_list=None, **config):
    if data_list is None:
        data_list = []
    # DataModule
    data_module = TrackingDataModule(
        data_list=data_list,
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
        dropout=config.get('dropout', 0.1)
    )

    # Callbacks y logger
    logger = CSVLogger("logs/", name=config.get('log_path', "tracking_lstm"))
    checkpoint_cb = ModelCheckpoint(save_top_k=1, mode='min', monitor='val_loss')
    earlystop_cb = EarlyStopping(monitor='val_loss', patience=config.get('patience', 7))
    # tqdm_cb = TQDMProgressBar(leave=True)
    custom_progress_bar_cb = CustomProgressBar()

    trainer = Trainer(
        max_epochs=config.get('epochs', 200),
        callbacks=[
            checkpoint_cb,
            earlystop_cb,
            # tqdm_cb
            custom_progress_bar_cb
        ],
        accelerator="auto",
        devices="auto",
        logger=logger,
        deterministic=False,
        log_every_n_steps=config.get('log_every_n_steps', 10),
    )
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)
