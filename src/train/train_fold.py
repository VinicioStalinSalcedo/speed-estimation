from pytorch_lightning.loggers import CSVLogger
from dataset.tracking_datamodule import TrackingDataModule
from models.lstm_model import TrackingLSTM
import optuna
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from dataset.tracking_datamodule import TrackingDataModule
from dataset.tracking_dataset import load_data_list
from models.lstm_model import TrackingLSTM
from train.train_lstm import CustomProgressBar


def train_fold(data_list, train_idx, val_idx, **config):

    print(f'>>> Start Train fold in rank...')
    # DataModule
    data_module = TrackingDataModule(
        data_list=data_list,
        split_ratio=None,
        train_idx=train_idx,
        val_idx=val_idx,
        batch_size=config.get('batch_size', 4),
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
    logger = CSVLogger("logs/", name=config.get('log_path', "folds"))
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
        # strategy="ddp_spawn",
        logger=logger,
        deterministic=False,
        log_every_n_steps=config.get('log_every_n_steps', 10),
        enable_progress_bar=config.get('progress_bar', True),
        enable_model_summary=config.get('model_summary', True)
    )
    print(f">>> Training fold with config: {config}")

    try:
        trainer.fit(model, datamodule=data_module)
    except Exception as e:
        print("⚠️ Error detectado al entrenar, fallando este fold con loss muy alto")
        return float('inf')

    result = trainer.validate(model, datamodule=data_module, ckpt_path='best')
    print(f'<<< Finish Trial')
    return result[0]['val_loss']
