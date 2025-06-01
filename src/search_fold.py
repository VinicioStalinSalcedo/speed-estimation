import os
import json
import numpy as np
import optuna
from sklearn.model_selection import KFold, train_test_split
import torch
import random
from pytorch_lightning import seed_everything
from dataset.tracking_dataset import load_data_list
from train.train_fold import train_fold
from optuna.importance import get_param_importances
import pandas as pd

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


def objective(trial, train_val_list, seed, input_size):
    hparams = {
        'hidden_size': trial.suggest_categorical('hidden_size', [32, 64, 128]),
        'num_layers':  trial.suggest_int('num_layers', 1, 3),
        'dropout':     trial.suggest_float('dropout', 0.0, 0.3),
        'lr':          trial.suggest_loguniform('lr', 1e-4, 1e-2),
        'weight_decay': trial.suggest_loguniform('weight_decay', 1e-6, 1e-4),
    }

    # K-Fold
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    losses = []

    # Directorio general de resultados del trial
    trial_dir = os.path.join('search_log', f'trial_{trial.number}')

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_list)):
        fold_dir = os.path.join(trial_dir, f'fold_{fold+1}')
        print(f"********* Entrenando fold {fold+1} con hiperparámetros: {hparams}")
        try:
            loss = train_fold(train_val_list,
                              train_idx,
                              val_idx,
                              # DataModule params
                              batch_size=4,
                              num_workers=4,

                              # LSTM params
                              input_size=input_size,
                              hidden_size=hparams['hidden_size'],
                              num_layers=hparams['num_layers'],
                              output_size=1,
                              learning_rate=hparams['lr'],
                              dropout=hparams['dropout'],
                              weight_decay=hparams['weight_decay'],

                              # Logger
                              log_path=fold_dir,

                              # Trainer
                              patience=5,
                              epochs=80,
                              log_every_n_steps=10,

                              progress_bar=True,
                              )
        except Exception as e:
            print(f"Error en el fold {fold+1}: {e}")
            loss = float('inf')
        losses.append(loss)

        # 1) Reporta el loss intermedio de este fold
        trial.report(loss, step=fold)

        # 2) Pregunta si debemos podar
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned(f"Trial pruned at fold {fold}")

    # Métrica agregada
    mean_loss = np.mean(losses)
    return mean_loss


if __name__ == '__main__':
    seed = 47

    seed_everything(seed)
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # Para mejorar el rendimiento en GPU
    random.seed(seed)
    np.random.seed(seed)

    # ── 1) Hold-out TEST ──────────────────────────────────────
    full_list = load_data_list(ALL_VIDEOS, results_folder='results_instance', feature_path='all_features')
    all_indices = np.arange(len(full_list))
    train_val_indices, test_indices = train_test_split(
        all_indices, test_size=0.2, random_state=seed, shuffle=True
    )
    train_val_list = [full_list[i] for i in train_val_indices]
    test_list = [full_list[i] for i in test_indices]
    input_size = len(train_val_list[0]['features'][0])

    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=seed),
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))

    study.optimize(lambda trial: objective(trial, train_val_list, seed, input_size), n_trials=20)

    try:
        best_params = study.best_params
        with open('best_params.json', 'w') as f:
            json.dump(best_params, f, indent=2)
        print("Mejores HParams:", best_params)
    except Exception as e:
        print(f"Error al guardar los mejores hiperparámetros: {e}")

    try:
        best_trial = study.best_trial
        print(f"Mejor trial: {best_trial.number} con loss {best_trial.value}")
        with open('best_trial.json', 'w') as f:
            json.dump({
                'number': best_trial.number,
                'value': best_trial.value,
                'params': best_trial.params
            }, f, indent=2)
        print("Mejor trial guardado en 'best_trial.json'")
    except Exception as e:
        print(f"Error al guardar el mejor trial: {e}")

    # 2) Extraer importancia de cada hparam
    try:
        importances = get_param_importances(study)

        # 3) Volcar a DataFrame y a CSV
        imp_df = pd.DataFrame.from_dict(importances, orient='index', columns=['importance'])
        imp_df = imp_df.sort_values('importance', ascending=False)
        imp_df.to_csv('param_importances.csv')

        print("Importancias de hiperparámetros:")
        print(imp_df)
    except Exception as e:
        print(f"Error al calcular o guardar importancias de hiperparámetros: {e}")

