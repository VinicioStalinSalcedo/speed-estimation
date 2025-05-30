import os
import pandas as pd
import yaml
import numpy as np

def load_trial_results_simplified(base_log_dir, n_trials, n_folds):
    all_trials_data = []
    optuna_hparams_keys = ['hidden_size', 'num_layers', 'dropout', 'lr', 'weight_decay']
    metrics_to_process = ['val_loss', 'val_mse', 'val_mae', 'val_acc']
    version_dir = 'version_0'

    if not os.path.isdir(base_log_dir):
        print(f"Error: Directorio base de logs no encontrado: {base_log_dir}")
        return pd.DataFrame()

    for trial_idx in range(n_trials):
        trial_name = f'trial_{trial_idx}'
        current_trial_hparams = {}

        # Almacena diccionarios de métricas (de la mejor época) para cada fold procesado en este trial
        trial_fold_best_epoch_metrics_list = []

        for fold_k in range(1, n_folds + 1):
            fold_name = f'fold_{fold_k}'
            version_path = os.path.join(base_log_dir, trial_name, fold_name, version_dir)

            if not os.path.isdir(version_path):
                print(f"Advertencia: Directorio no encontrado {version_path}. Saltando fold {fold_k} para trial {trial_idx}.")
                continue

            hparams_file = os.path.join(version_path, 'hparams.yaml')
            with open(hparams_file, 'r') as f:
                hparams_content = yaml.safe_load(f)
                current_trial_hparams = {k: hparams_content[k] for k in optuna_hparams_keys if k in hparams_content}

            metrics_file = os.path.join(version_path, 'metrics.csv')
            if os.path.exists(metrics_file):
                df_metrics = pd.read_csv(metrics_file)
                if 'val_loss' in df_metrics.columns and not df_metrics['val_loss'].dropna().empty:
                    best_epoch_idx = df_metrics['val_loss'].dropna().idxmin()
                    best_epoch_row_metrics = df_metrics.loc[best_epoch_idx]

                    current_fold_metrics = {}
                    for metric_key in metrics_to_process:
                        if metric_key in best_epoch_row_metrics.index and pd.notna(best_epoch_row_metrics[metric_key]):
                            current_fold_metrics[metric_key] = best_epoch_row_metrics[metric_key]

                    if 'val_loss' in current_fold_metrics:
                        trial_fold_best_epoch_metrics_list.append(current_fold_metrics)

        if trial_fold_best_epoch_metrics_list:
            df_aggregated_fold_metrics = pd.DataFrame(trial_fold_best_epoch_metrics_list)
            # Calcular la media de cada métrica a través de los folds
            mean_metrics_for_trial = df_aggregated_fold_metrics.mean().to_dict()

            trial_data_entry = {'trial': trial_idx, **current_trial_hparams}
            for key, value in mean_metrics_for_trial.items():
                trial_data_entry[f'mean_{key}'] = value # Ej: mean_val_loss, mean_val_mse

            trial_data_entry['num_folds_processed'] = len(trial_fold_best_epoch_metrics_list)
            all_trials_data.append(trial_data_entry)

    if not all_trials_data:
        print("No se encontraron datos de trials procesables con la configuración proporcionada.")
        return pd.DataFrame()

    results_df = pd.DataFrame(all_trials_data)

    # Asegurar que todas las columnas esperadas existan
    mean_metric_cols = [f'mean_{m}' for m in metrics_to_process]
    final_columns = ['trial'] + optuna_hparams_keys + mean_metric_cols + ['num_folds_processed']

    for col in final_columns:
        if col not in results_df.columns:
            results_df[col] = np.nan

    results_df = results_df[final_columns]

    if 'mean_val_loss' in results_df.columns:
        results_df = results_df.sort_values(by='mean_val_loss', ascending=True)
    else:
        print("Advertencia: La columna 'mean_val_loss' no está presente. El DataFrame no se ordenará.")

    return results_df


log_directory = 'logs/optuna_trials' # AJUSTA ESTA RUTA

# Número de trials y folds que configuraste en tu estudio Optuna
num_total_trials = 20
num_folds_per_trial = 5

df_simplified_results = load_trial_results_simplified(
    base_log_dir=log_directory,
    n_trials=num_total_trials,
    n_folds=num_folds_per_trial
)

if not df_simplified_results.empty:
    print("Resultados simplificados de los trials (ordenados por mean_val_loss ascendente):")
    print(df_simplified_results.head())
else:
    print("No se pudieron cargar los resultados de los trials con el método simplificado.")
