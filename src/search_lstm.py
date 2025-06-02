from train.train_lstm import train_lstm_pipeline
from dataset.tracking_dataset import load_data_list
import numpy as np
import pandas as pd
import itertools
import json
import os
from pytorch_lightning.callbacks import ModelCheckpoint

ALL_VIDEOS = [
    ('session1', 'center'),
    ('session1', 'left'),
    ('session1', 'right'),
    ('session2', 'center'),
    ('session2', 'left'),
    ('session2', 'right'),
    ('session3', 'center'),
    ('session3', 'left'),
    ('session3', 'right'),
    ('session4', 'center'),
    ('session4', 'left'),
    ('session4', 'right'),
    ('session5', 'center'),
    ('session5', 'left'),
    ('session5', 'right'),
    ('session6', 'center'),
    ('session6', 'left'),
    ('session6', 'right'),
]

# feature_path = 'fbbox'
# feature_path = 'faspect'
# feature_path = 'fbbox_angle'
# feature_path = 'fall'
feature_path = 'all_features'


def search():
    data_list = load_data_list(ALL_VIDEOS, results_folder='results_instance', feature_path='all_features')

    param_grid = {
        'hidden_size': [32, 64, 128],
        'num_layers': [2, 3, 4],
        'dropout': [0.0, 0.1, 0.2],
        'learning_rate': [1e-2, 1e-3, 1e-4],
        'weight_decay': [1e-6, 1e-5, 1e-4],
    }

    input_size = len(data_list[0]['features'][0])
    results = []

    for hs, nl, do, lr, wd in itertools.product(*param_grid.values()):
        print(f'**Grid Search**')
        print(f"Entrenando con hs={hs}, nl={nl}, do={do}, lr={lr}, wd={wd}")
        print(f'Feature path: {feature_path}')
        exp_name = f'{feature_path}/hs{hs}_nl{nl}_do{do}_lr{lr}_wd{wd}'

        data_module, model, trainer, logger, val_metrics, test_metrics = train_lstm_pipeline(
            # Dataset
            data_list,

            # DataModule params
            batch_size=4,
            split_ratio=(0.8, 0.1, 0.1),
            num_workers=4,

            # LSTM params
            input_size=input_size,
            hidden_size=hs,
            num_layers=nl,
            output_size=1,
            learning_rate=lr,
            dropout=do,
            weight_decay=wd,

            # Logger
            log_path=exp_name,

            # Trainer
            patience=5,
            epochs=100,
            log_every_n_steps=10,

            progress_bar=True,
        )

        # Acceder al callback ModelCheckpoint
        model_checkpoint_callback = None
        for callback in trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                model_checkpoint_callback = callback
                break

        best_val_loss = model_checkpoint_callback.best_model_score.item() if model_checkpoint_callback else None

        val_loss = val_metrics["val_loss"].item()
        val_mse = val_metrics["val_mse"].item()
        val_mae = val_metrics["val_mae"].item()
        val_acc = val_metrics["val_acc"].item()
        val_rmse = np.sqrt(val_mse)

        test_loss = test_metrics["test_loss"].item()
        test_mse = test_metrics["test_mse"].item()
        test_mae = test_metrics["test_mae"].item()
        test_acc = test_metrics["test_acc"].item()
        test_rmse = np.sqrt(test_mse)

        results.append({
            "exp": exp_name,
            "hidden_size": hs,
            "num_layers": nl,
            "dropout": do,
            "lr": lr,
            "batch_size": 4,
            "best_val_loss": best_val_loss,
            "val_loss": val_loss,
            "val_mse": val_mse,
            "val_mae": val_mae,
            "val_acc": val_acc,
            "val_rmse": val_rmse,
            "test_loss": test_loss,
            "test_mse": test_mse,
            "test_mae": test_mae,
            "test_acc": test_acc,
            "test_rmse": test_rmse,
        })

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("best_val_loss").reset_index(drop=True)

    results_search_path = f"results_search/{feature_path}"
    os.makedirs(results_search_path, exist_ok=True)

    df_results.to_csv(f"{results_search_path}/search_results.csv", index=False)
    print(f"Resultados del grid search guardados en: {results_search_path}/search_results.csv")


if __name__ == '__main__':
    search()
