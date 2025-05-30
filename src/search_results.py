import os
import pandas as pd
import yaml
import numpy as np
import matplotlib.pyplot as plt


def load_trial_results(base_log_dir, n_trials, n_folds):
    all_trials_data = []
    optuna_hparams_keys = ['hidden_size', 'num_layers', 'dropout', 'lr', 'weight_decay']
    metrics_to_process = ['val_loss', 'val_mse', 'val_mae', 'val_acc']
    version_dir = 'version_0'

    if not os.path.isdir(base_log_dir):
        print(f"Error: Directorio base de logs no encontrado: {base_log_dir}")
        return pd.DataFrame()

    for trial_idx in range(n_trials):
        trial_name = f'trial_{trial_idx}'
        fold_metrics = []

        for fold_k in range(1, n_folds + 1):
            fold_name = f'fold_{fold_k}'
            version_path = os.path.join(base_log_dir, trial_name, fold_name, version_dir)

            if not os.path.isdir(version_path):
                print(f"Advertencia: Directorio no encontrado {version_path}. Saltando fold {fold_k} para trial {trial_idx}.")
                continue

            hparams_file = os.path.join(version_path, 'hparams.yaml')
            with open(hparams_file, 'r') as f:
                hparams_content = yaml.safe_load(f)

            metrics_file = os.path.join(version_path, 'metrics.csv')
            df_metrics = pd.read_csv(metrics_file)
            if 'val_loss' in df_metrics.columns and not df_metrics['val_loss'].dropna().empty:
                best_epoch_idx = df_metrics['val_loss'].dropna().idxmin()
                best_epoch_row_metrics = df_metrics.loc[best_epoch_idx]

                current_fold_metrics = {}
                for metric_key in metrics_to_process:
                    if metric_key in best_epoch_row_metrics.index and pd.notna(best_epoch_row_metrics[metric_key]):
                        current_fold_metrics[metric_key] = best_epoch_row_metrics[metric_key]

                if 'val_loss' in current_fold_metrics:
                    fold_metrics.append(current_fold_metrics)

        if fold_metrics:
            df_aggregated_fold_metrics = pd.DataFrame(fold_metrics)
            trial_metrics = df_aggregated_fold_metrics.mean().to_dict()
            trial_metrics['trial'] = trial_idx
            trial_metrics['folds'] = len(fold_metrics)

            metrics_for_std = ['val_mse', 'val_mae', 'val_acc']
            for metric_key_std in metrics_for_std:
                if metric_key_std in df_aggregated_fold_metrics.columns:
                    std_val = df_aggregated_fold_metrics[metric_key_std].std(ddof=0) # ddof=0 para std poblacional de los folds
                    trial_metrics[f'{metric_key_std}_std'] = std_val if pd.notna(std_val) else 0.0
                else:
                    # Si la columna de la métrica no existe en df_aggregated_fold_metrics (ningún fold la tenía), std es NaN.
                    trial_metrics[f'{metric_key_std}_std'] = np.nan

            for hparam_key in optuna_hparams_keys:
                if hparam_key in hparams_content:
                    trial_metrics[hparam_key] = hparams_content[hparam_key]
            all_trials_data.append(trial_metrics)

    if not all_trials_data:
        print("No se encontraron datos de trials procesables con la configuración proporcionada.")
        return pd.DataFrame()

    results_df = pd.DataFrame(all_trials_data)

    # Definir columnas finales incluyendo desviaciones estándar, manteniendo nombres originales donde sea posible
    std_cols = [f'{m}_std' for m in ['val_mse', 'val_mae', 'val_acc']]
    # Estructura original de final_columns: ['trial'] + optuna_hparams_keys + metrics_to_process + ['folds']
    # Nueva estructura:
    final_columns_ordered = ['trial'] + \
                            optuna_hparams_keys + \
                            metrics_to_process + \
                            std_cols + \
                            ['folds']
    results_df = results_df.reindex(columns=final_columns_ordered)

    if 'val_loss' in results_df.columns and not results_df['val_loss'].isnull().all():
        results_df = results_df.sort_values(by='val_loss', ascending=True).reset_index(drop=True)
    else:
        print("Advertencia: La columna 'val_loss' no está presente. El DataFrame no se ordenará.")

    return results_df


def generate_summary_table(results_df):
    """
    Genera una tabla de resumen con hiperparámetros y métricas (media ± std).
    Resalta el mejor trial basado en val_loss.
    """
    if results_df.empty:
        print("El DataFrame de resultados está vacío. No se puede generar la tabla de resumen.")
        return pd.DataFrame()

    summary_data = []
    # Columnas de hiperparámetros a incluir si existen
    hparam_cols = ['hidden_size', 'num_layers', 'dropout', 'lr', 'weight_decay']

    for _, row in results_df.iterrows():
        summary_row = {'Trial': row['trial']}
        for hp_col in hparam_cols:
            if hp_col in row: # Verificar si la columna existe en el DataFrame
                value = row[hp_col]
                if pd.notna(value):
                    if hp_col in ['lr', 'weight_decay']:
                        summary_row[hp_col.replace('_', ' ').title()] = f"{value:.1e}"
                    elif hp_col == 'dropout':
                        summary_row[hp_col.replace('_', ' ').title()] = f"{value:.2f}"
                    else: # hidden_size, num_layers (generalmente int)
                        summary_row[hp_col.replace('_', ' ').title()] = int(value) if pd.notna(value) and isinstance(value, (float, int)) and value == int(value) else value
                else:
                    summary_row[hp_col.replace('_', ' ').title()] = 'N/A'
            else:
                summary_row[hp_col.replace('_', ' ').title()] = 'N/A' # Si el hparam no está en results_df para esta fila

        # Usar .get con default np.nan para métricas y std para manejar datos faltantes con gracia
        mse_mean = row.get('val_mse', np.nan)
        mse_std = row.get('val_mse_std', np.nan)
        mae_mean = row.get('val_mae', np.nan)
        mae_std = row.get('val_mae_std', np.nan)
        acc_mean = row.get('val_acc', np.nan)
        acc_std = row.get('val_acc_std', np.nan)

        summary_row['MSE (Val)'] = f"{mse_mean:.3f} ± {mse_std:.3f}" if pd.notna(mse_mean) and pd.notna(mse_std) else "N/A"
        summary_row['MAE (Val)'] = f"{mae_mean:.3f} ± {mae_std:.3f}" if pd.notna(mae_mean) and pd.notna(mae_std) else "N/A"
        summary_row['RelAcc (Val)'] = f"{acc_mean:.3f} ± {acc_std:.3f}" if pd.notna(acc_mean) and pd.notna(acc_std) else "N/A"

        summary_row['Folds'] = row.get('folds', 0) # Usar 'folds' como en load_trial_results
        summary_row['_val_loss_raw_'] = row.get('val_loss', float('inf')) # Para encontrar el mejor
        summary_data.append(summary_row)

    summary_df = pd.DataFrame(summary_data)

    if not summary_df.empty and '_val_loss_raw_' in summary_df.columns and summary_df['_val_loss_raw_'].notna().any():
        # Asegurarse de que _val_loss_raw_ es numérico antes de idxmin
        summary_df['_val_loss_raw_'] = pd.to_numeric(summary_df['_val_loss_raw_'], errors='coerce')
        if summary_df['_val_loss_raw_'].notna().any(): # Comprobar de nuevo después de la coerción
            best_trial_idx_loc = summary_df['_val_loss_raw_'].idxmin()
            summary_df['Mejor Trial'] = ''
            summary_df.loc[best_trial_idx_loc, 'Mejor Trial'] = '<-- MEJOR'
            print(f"Mejor combinación de hiperparámetros (Trial {summary_df.loc[best_trial_idx_loc, 'Trial']}) basada en val_loss mínimo.")
        else:
            print("Advertencia: No se pudo determinar el mejor trial porque todos los valores de _val_loss_raw_ son NaN.")


    return summary_df.drop(columns=['_val_loss_raw_'], errors='ignore')


def plot_learning_curves(base_log_dir, results_df, n_folds):
    """
    Grafica las curvas de aprendizaje (train_loss, val_loss) promediadas sobre los folds
    para cada trial, con bandas de confianza (±σ).
    """
    if results_df.empty:
        print("El DataFrame de resultados está vacío. No se pueden graficar curvas de aprendizaje.")
        return

    optuna_hparams_keys = ['hidden_size', 'num_layers', 'dropout', 'lr', 'weight_decay']
    version_dir = 'version_0'

    for index, trial_row in results_df.iterrows():
        trial_idx = trial_row['trial'] # Asumiendo que 'trial' es el identificador numérico del trial
        trial_name = f'trial_{trial_idx}'

        hparams_str_parts = []
        for key in optuna_hparams_keys:
            if key in trial_row and pd.notna(trial_row[key]):
                value = trial_row[key]
                if isinstance(value, float):
                    if key in ['lr', 'weight_decay']:
                         hparams_str_parts.append(f"{key}={value:.1e}")
                    elif key == 'dropout':
                         hparams_str_parts.append(f"{key}={value:.2f}")
                    else: # hidden_size, num_layers si fueran float por alguna razón
                         hparams_str_parts.append(f"{key}={value}")
                else: # int o str
                    hparams_str_parts.append(f"{key}={value}")
        hparams_info = ", ".join(hparams_str_parts)

        fold_metrics_dfs = []
        for fold_k in range(1, n_folds + 1):
            fold_name_loop = f'fold_{fold_k}' # Nombre de variable diferente para evitar confusión con 'fold_name' fuera del loop si existiera
            metrics_file = os.path.join(base_log_dir, trial_name, fold_name_loop, version_dir, 'metrics.csv')

            if os.path.exists(metrics_file):
                try:
                    df_metrics_fold = pd.read_csv(metrics_file)
                    required_cols = ['epoch', 'val_loss'] # Columnas mínimas necesarias
                    if all(col in df_metrics_fold.columns for col in required_cols):
                        # Usar filas donde val_loss no es NaN (indica fin de epoch de validación)
                        epoch_metrics = df_metrics_fold[df_metrics_fold['val_loss'].notna()].copy()

                        cols_to_keep = ['epoch', 'val_loss']
                        # Priorizar 'train_loss_epoch' si existe y tiene datos
                        if 'train_loss_epoch' in epoch_metrics.columns and epoch_metrics['train_loss_epoch'].notna().any():
                            cols_to_keep.append('train_loss_epoch')
                        # Si no, usar 'train_loss' si existe y tiene datos (asumiendo que es por epoch)
                        elif 'train_loss' in epoch_metrics.columns and epoch_metrics['train_loss'].notna().any():
                            # Renombrar 'train_loss' a 'train_loss_epoch' para consistencia interna temporal
                            epoch_metrics.rename(columns={'train_loss': 'train_loss_epoch'}, inplace=True)
                            cols_to_keep.append('train_loss_epoch')

                        if not epoch_metrics.empty and 'epoch' in epoch_metrics.columns: # Asegurar que 'epoch' está presente
                             fold_metrics_dfs.append(epoch_metrics[cols_to_keep])
                except pd.errors.EmptyDataError:
                    # print(f"Advertencia: Archivo de métricas vacío {metrics_file}. Saltando.")
                    pass
                except Exception as e:
                    print(f"Error leyendo {metrics_file} para trial {trial_idx}, fold {fold_k}: {e}. Saltando.")
            # else:
                # print(f"Advertencia: Archivo de métricas no encontrado {metrics_file}. Saltando.")


        if not fold_metrics_dfs:
            print(f"No se encontraron datos de métricas para graficar curvas de aprendizaje para el trial {trial_idx}.")
            continue

        combined_metrics_df = pd.concat(fold_metrics_dfs)

        # Renombrar 'train_loss_epoch' a 'train_loss' para el agrupamiento y ploteo final
        if 'train_loss_epoch' in combined_metrics_df.columns:
            combined_metrics_df.rename(columns={'train_loss_epoch': 'train_loss'}, inplace=True)

        # Asegurar que las columnas son numéricas
        combined_metrics_df['epoch'] = pd.to_numeric(combined_metrics_df['epoch'], errors='coerce')
        if 'train_loss' in combined_metrics_df.columns:
            combined_metrics_df['train_loss'] = pd.to_numeric(combined_metrics_df['train_loss'], errors='coerce')
        combined_metrics_df['val_loss'] = pd.to_numeric(combined_metrics_df['val_loss'], errors='coerce')

        combined_metrics_df.dropna(subset=['epoch'], inplace=True) # Eliminar epochs no válidos

        # Agrupar por epoch y calcular media y std (ddof=0 para std de los folds de este trial)
        agg_funcs = {
            'mean_val_loss': ('val_loss', 'mean'),
            'std_val_loss': ('val_loss', lambda x: x.std(ddof=0) if len(x) > 0 else 0.0)
        }
        if 'train_loss' in combined_metrics_df.columns: # Solo agregar si 'train_loss' existe después del preprocesamiento
            agg_funcs['mean_train_loss'] = ('train_loss', 'mean')
            agg_funcs['std_train_loss'] = ('train_loss', lambda x: x.std(ddof=0) if len(x) > 0 else 0.0)

        epoch_summary = combined_metrics_df.groupby('epoch').agg(**agg_funcs).reset_index()

        # Rellenar std con 0.0 si no hay varianza o solo un punto de datos por epoch, o si la columna no existía
        epoch_summary.fillna({'std_train_loss': 0.0, 'std_val_loss': 0.0}, inplace=True)


        plt.figure(figsize=(12, 7))
        plot_legend = False

        if 'mean_train_loss' in epoch_summary.columns and epoch_summary['mean_train_loss'].notna().any():
            plt.plot(epoch_summary['epoch'], epoch_summary['mean_train_loss'], label='Train Loss (promedio)')
            plt.fill_between(epoch_summary['epoch'],
                             epoch_summary['mean_train_loss'] - epoch_summary['std_train_loss'],
                             epoch_summary['mean_train_loss'] + epoch_summary['std_train_loss'],
                             alpha=0.2, label='Train Loss (±σ)')
            plot_legend = True

        if 'mean_val_loss' in epoch_summary.columns and epoch_summary['mean_val_loss'].notna().any():
            plt.plot(epoch_summary['epoch'], epoch_summary['mean_val_loss'], label='Validation Loss (promedio)')
            plt.fill_between(epoch_summary['epoch'],
                             epoch_summary['mean_val_loss'] - epoch_summary['std_val_loss'],
                             epoch_summary['mean_val_loss'] + epoch_summary['std_val_loss'],
                             alpha=0.2, label='Validation Loss (±σ)')
            plot_legend = True

        plt.title(f'Curvas de Aprendizaje - Trial {trial_idx}\n{hparams_info}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if plot_legend:
            plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# Ejemplo de uso (deberás adaptarlo a tu script principal):
# if __name__ == '__main__':
#     BASE_LOG_DIRECTORY = 'lightning_logs/optuna_study_lstm_sc_kfold' # Cambia esto a tu directorio
#     N_TRIALS = 25  # Número total de trials en tu estudio Optuna
#     N_FOLDS = 5    # Número de folds usados en KFold CV
#
#     # Cargar resultados de los trials
#     results = load_trial_results(BASE_LOG_DIRECTORY, N_TRIALS, N_FOLDS)
#
#     if not results.empty:
#         print("Resultados de los Trials Cargados:")
#         # Usar to_string() para mejor visualización de DataFrames anchos en la consola
#         print(results.to_string())
#         print("\n" + "="*50 + "\n")
#
#         # Generar y mostrar tabla de resumen
#         summary_table = generate_summary_table(results)
#         print("Tabla de Resumen Estadístico por Trial:")
#         # Usar to_string() para mejor visualización
#         print(summary_table.to_string())
#         print("\n" + "="*50 + "\n")
#
#         # Graficar curvas de aprendizaje
#         print("Generando Curvas de Aprendizaje...")
#         plot_learning_curves(BASE_LOG_DIRECTORY, results, N_FOLDS)
#         print("Curvas de aprendizaje generadas.")
#     else:
#         print("No se pudieron cargar resultados para generar informes.")