from train.train_lstm import train_lstm_pipeline
import numpy as np
import pandas as pd
import itertools
import json
import os
from pytorch_lightning.callbacks import ModelCheckpoint # Importa ModelCheckpoint

ALL_VIDEOS = [
    # ('session0', 'center'),
    # ('session0', 'left'),
    # ('session0', 'right'),
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

feature_path = 'fbbox'
# feature_path = 'faspect'
# feature_path = 'fbbox_angle'
# feature_path = 'fall'


def search_results():
    param_grid = {
        'hidden_size': [32, 64, 128],
        'num_layers': [1, 2, 3],
        'dropout': [0.0, 0.2, 0.5],
        'learning_rate': [1e-3, 1e-4, 1e-5],
    }

    summary_data = []

    for hs, nl, do, lr in itertools.product(*param_grid.values()):
        print(f'**Grid Search**')
        print(f"Entrenando con hs={hs}, nl={nl}, do={do}, lr={lr}")
        print(f'Feature path: {feature_path}')
        exp_name = f'{feature_path}/hs{hs}_nl{nl}_do{do}_lr{lr}'
        metrics_file = f'logs/{exp_name}/metrics.csv'

        if os.path.exists(metrics_file):
            df_metrics = pd.read_csv(metrics_file)

            # Asegurarse de que 'val_loss' existe y no es nula
            if 'val_loss' in df_metrics.columns and not df_metrics['val_loss'].isnull().all():
                # Encontrar la fila con la mínima val_loss (excluyendo NaNs que ocurren en la última época)
                best_val_row = df_metrics.loc[df_metrics['val_loss'].dropna().idxmin()]

                # Puedes agregar más métricas si son relevantes y están en tu CSV
                summary_data.append({
                    'Version': exp_name,
                    'Epoch': int(best_val_row['epoch']),
                    'Best Val Loss': best_val_row['val_loss'],
                    'Train Loss at Best Val': best_val_row['train_loss'] if 'train_loss' in best_val_row else 'N/A',
                    # Agrega aquí otras métricas de validación si las registras (ej. val_accuracy)
                })
            else:
                print(f"Advertencia: '{metrics_file}' no contiene 'val_loss' o está vacía.")
        else:
            print(f"Advertencia: '{metrics_file}' no encontrado.")

    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        return df_summary
    else:
        return pd.DataFrame()


if __name__ == '__main__':
    summary = search_results()
    print(summary)
    display(summary)
    print(summary)
