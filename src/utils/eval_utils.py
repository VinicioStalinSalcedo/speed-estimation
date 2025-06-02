import os
import pandas as pd
import yaml
import numpy as np
import matplotlib.pyplot as plt
from models.lstm_model import TrackingLSTM
import glob
import pytorch_lightning as pl


def load_trial_results(base_log_dir, n_trials, n_folds):
    all_trials_data = []
    optuna_hparams_keys = ['hidden_size', 'num_layers', 'dropout', 'lr', 'weight_decay']
    metrics_to_process = ['val_loss', 'val_mse', 'val_mae', 'val_acc']
    version_dir = 'version_0'
    model_class_warning_issued = False

    if not os.path.isdir(base_log_dir):
        print(f"Error: Directorio base de logs no encontrado: {base_log_dir}")
        return pd.DataFrame()

    for trial_idx in range(n_trials):
        trial_name = f'trial_{trial_idx}'
        fold_metrics_list = []
        hparams_content_for_trial = {}
        trainable_params_for_trial = np.nan
        hparams_and_params_loaded_for_trial = False

        for fold_k in range(1, n_folds + 1):
            fold_name = f'fold_{fold_k}'
            version_path = os.path.join(base_log_dir, trial_name, fold_name, version_dir)

            if not os.path.isdir(version_path):
                # print(f"Advertencia: Directorio no encontrado {version_path}. Saltando fold {fold_k} para trial {trial_idx}.")
                continue

            # Cargar hparams y calcular parámetros entrenables UNA VEZ por trial, usando el primer fold exitoso
            if not hparams_and_params_loaded_for_trial:
                hparams_file_temp = os.path.join(version_path, 'hparams.yaml')
                if os.path.exists(hparams_file_temp):
                    with open(hparams_file_temp, 'r') as f:
                        hparams_content_for_trial = yaml.safe_load(f)

                checkpoint_dir = os.path.join(version_path, 'checkpoints')
                if os.path.isdir(checkpoint_dir):
                    # Intentar obtener la época de la mejor val_loss de este fold para el nombre del checkpoint
                    # Esto es solo para la heurística de selección del checkpoint, no afecta las métricas finales del fold.
                    temp_metrics_file_for_epoch = os.path.join(version_path, 'metrics.csv')
                    epoch_num_of_best_val_loss_for_ckpt = 0 # Default a 0 si no se puede leer
                    if os.path.exists(temp_metrics_file_for_epoch):
                        try:
                            temp_df_metrics = pd.read_csv(temp_metrics_file_for_epoch)
                            if 'val_loss' in temp_df_metrics.columns and not temp_df_metrics['val_loss'].dropna().empty and 'epoch' in temp_df_metrics.columns:
                                epoch_num_of_best_val_loss_for_ckpt = int(temp_df_metrics.loc[temp_df_metrics['val_loss'].dropna().idxmin()]['epoch'])
                        except Exception:
                            pass # No es crítico si falla, se usarán otros métodos para encontrar ckpt

                    selected_ckpt_path = None
                    ckpt_pattern_exact_epoch = os.path.join(checkpoint_dir, f"epoch={epoch_num_of_best_val_loss_for_ckpt}-*.ckpt")
                    specific_epoch_ckpts = glob.glob(ckpt_pattern_exact_epoch)
                    if not specific_epoch_ckpts:
                        ckpt_pattern_exact_epoch_no_step = os.path.join(checkpoint_dir, f"epoch={epoch_num_of_best_val_loss_for_ckpt}.ckpt")
                        specific_epoch_ckpts = glob.glob(ckpt_pattern_exact_epoch_no_step)
                    
                    if specific_epoch_ckpts:
                        specific_epoch_ckpts.sort()
                        selected_ckpt_path = specific_epoch_ckpts[-1]
                    else:
                        last_ckpt_path = os.path.join(checkpoint_dir, 'last.ckpt')
                        if os.path.exists(last_ckpt_path):
                            selected_ckpt_path = last_ckpt_path
                        else:
                            all_ckpts = glob.glob(os.path.join(checkpoint_dir, '*.ckpt'))
                            if all_ckpts:
                                all_ckpts.sort()
                                selected_ckpt_path = all_ckpts[-1]
                    
                    if selected_ckpt_path:
                        try:
                            model = TrackingLSTM.load_from_checkpoint(selected_ckpt_path)
                            model.eval()
                            trainable_params_for_trial = sum(p.numel() for p in model.parameters() if p.requires_grad)
                        except NameError:
                            if not model_class_warning_issued:
                                print("Advertencia: La clase 'YourLightningModule' no está definida. No se calcularán los parámetros entrenables.")
                                model_class_warning_issued = True
                        except ImportError:
                             if not model_class_warning_issued:
                                print("Advertencia: PyTorch Lightning o la clase del modelo no pudieron ser importados. No se calcularán los parámetros entrenables.")
                                model_class_warning_issued = True
                        except Exception as e:
                            # print(f"Advertencia: No se pudo cargar el modelo desde {selected_ckpt_path} o calcular params para trial {trial_idx} (usando fold {fold_k}): {e}")
                            pass # Se mantendrá np.nan
                
                if hparams_content_for_trial or not np.isnan(trainable_params_for_trial):
                     hparams_and_params_loaded_for_trial = True


            metrics_file = os.path.join(version_path, 'metrics.csv')
            if not os.path.exists(metrics_file):
                continue
            
            try:
                df_metrics = pd.read_csv(metrics_file)
            except pd.errors.EmptyDataError:
                continue
            except Exception:
                continue

            current_fold_metrics_dict = {}
            if 'val_loss' in df_metrics.columns and not df_metrics['val_loss'].dropna().empty:
                best_epoch_idx = df_metrics['val_loss'].dropna().idxmin()
                best_epoch_row_metrics = df_metrics.loc[best_epoch_idx].copy()

                for metric_key in metrics_to_process:
                    if metric_key in best_epoch_row_metrics.index and pd.notna(best_epoch_row_metrics[metric_key]):
                        current_fold_metrics_dict[metric_key] = best_epoch_row_metrics[metric_key]
                
                if 'val_loss' in current_fold_metrics_dict:
                    fold_metrics_list.append(current_fold_metrics_dict)

        if fold_metrics_list:
            df_aggregated_fold_metrics = pd.DataFrame(fold_metrics_list)
            trial_summary_metrics = df_aggregated_fold_metrics.mean().to_dict() # Media de las métricas de los folds
            
            trial_summary_metrics['trial'] = trial_idx
            trial_summary_metrics['folds'] = len(fold_metrics_list)
            trial_summary_metrics['trainable_params'] = trainable_params_for_trial # Asignar el valor único del trial

            metrics_for_std_calc = ['val_mse', 'val_mae', 'val_acc'] # val_loss ya se promedia, su std no se guarda aquí
            for metric_key_std in metrics_for_std_calc:
                if metric_key_std in df_aggregated_fold_metrics.columns:
                    std_val = df_aggregated_fold_metrics[metric_key_std].std(ddof=0)
                    trial_summary_metrics[f'{metric_key_std}_std'] = std_val if pd.notna(std_val) else 0.0
                else:
                    trial_summary_metrics[f'{metric_key_std}_std'] = np.nan
            
            if hparams_content_for_trial:
                for hparam_key in optuna_hparams_keys:
                    if hparam_key in hparams_content_for_trial:
                        trial_summary_metrics[hparam_key] = hparams_content_for_trial[hparam_key]
            
            all_trials_data.append(trial_summary_metrics)

    if not all_trials_data:
        print("No se encontraron datos de trials procesables con la configuración proporcionada.")
        return pd.DataFrame()

    results_df = pd.DataFrame(all_trials_data)

    std_cols = [f'{m}_std' for m in ['val_mse', 'val_mae', 'val_acc']]
    param_col = ['trainable_params'] if 'trainable_params' in results_df.columns else []

    final_columns_ordered = ['trial'] + \
                            optuna_hparams_keys + \
                            param_col + \
                            metrics_to_process + \
                            std_cols + \
                            ['folds']
    
    results_df = results_df.reindex(columns=final_columns_ordered)

    if 'val_loss' in results_df.columns and not results_df['val_loss'].isnull().all():
        results_df = results_df.sort_values(by='val_loss', ascending=True).reset_index(drop=True)

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
        
        summary_row['Trainable Params'] = f"{row.get('trainable_params', np.nan):,}" if pd.notna(row.get('trainable_params')) else 'N/A'

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


def get_kfold_mean_metrics(base_log_dir, trial_row, n_folds):
    """
    Grafica las curvas de aprendizaje (train_loss, val_loss) promediadas sobre los folds
    para cada trial, con bandas de confianza (±σ).
    """

    optuna_hparams_keys = ['hidden_size', 'num_layers', 'dropout', 'lr', 'weight_decay']
    version_dir = 'version_0'

    trial_idx = int(trial_row['trial'])
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
                df_fold_raw_metrics = pd.read_csv(metrics_file)

                if not ('epoch' in df_fold_raw_metrics.columns and 'val_loss' in df_fold_raw_metrics.columns):
                    # print(f"Advertencia: 'epoch' o 'val_loss' faltan en {metrics_file} para trial {trial_idx}, fold {fold_k}. Saltando fold.")
                    continue

                df_fold_raw_metrics['epoch'] = pd.to_numeric(df_fold_raw_metrics['epoch'], errors='coerce')
                df_fold_raw_metrics.dropna(subset=['epoch'], inplace=True)

                # 1. Procesar val_loss (puntos de época primarios)
                fold_val_loss_data = df_fold_raw_metrics[df_fold_raw_metrics['val_loss'].notna()][['epoch', 'val_loss']].copy()
                if fold_val_loss_data.empty:
                    # print(f"Advertencia: No hay datos de val_loss válidos en {metrics_file} para trial {trial_idx}, fold {fold_k}.")
                    continue

                # 2. Procesar train_loss
                fold_train_loss_data = pd.DataFrame() # DataFrame vacío por defecto
                if 'train_loss_epoch' in df_fold_raw_metrics.columns and df_fold_raw_metrics['train_loss_epoch'].notna().any():
                    # Usar 'train_loss_epoch' si está disponible y tiene datos
                    train_loss_temp = df_fold_raw_metrics[df_fold_raw_metrics['train_loss_epoch'].notna()][['epoch', 'train_loss_epoch']].copy()
                    train_loss_temp.rename(columns={'train_loss_epoch': 'train_loss'}, inplace=True)
                    fold_train_loss_data = train_loss_temp
                elif 'train_loss' in df_fold_raw_metrics.columns and df_fold_raw_metrics['train_loss'].notna().any():
                    # Si 'train_loss' (potencialmente por paso) está disponible, promediar por época
                    df_fold_raw_metrics['train_loss'] = pd.to_numeric(df_fold_raw_metrics['train_loss'], errors='coerce')
                    valid_train_steps = df_fold_raw_metrics.dropna(subset=['epoch', 'train_loss'])
                    if not valid_train_steps.empty:
                        fold_train_loss_data = valid_train_steps.groupby('epoch')['train_loss'].mean().reset_index()

                # 3. Unir datos de val_loss y train_loss para el fold actual
                current_fold_epoch_metrics = fold_val_loss_data
                if not fold_train_loss_data.empty:
                    current_fold_epoch_metrics = pd.merge(fold_val_loss_data, fold_train_loss_data, on='epoch', how='left')
                else:
                    # Asegurar que la columna 'train_loss' exista incluso si no se encontraron datos de entrenamiento
                    current_fold_epoch_metrics['train_loss'] = np.nan

                if not current_fold_epoch_metrics.empty:
                    fold_metrics_dfs.append(current_fold_epoch_metrics)

            except pd.errors.EmptyDataError:
                # print(f"Advertencia: Archivo de métricas vacío {metrics_file}. Saltando.")
                pass
            except Exception as e:
                print(f"Error leyendo {metrics_file} para trial {trial_idx}, fold {fold_k}: {e}. Saltando.")
        # else:
            # print(f"Advertencia: Archivo de métricas no encontrado {metrics_file}. Saltando.")

    if not fold_metrics_dfs:
        print(f"No se encontraron datos de métricas para graficar curvas de aprendizaje para el trial {trial_idx}.")
        return

    combined_metrics_df = pd.concat(fold_metrics_dfs)

    # Asegurar que las columnas son numéricas después de la unión y concatenación
    combined_metrics_df['epoch'] = pd.to_numeric(combined_metrics_df['epoch'], errors='coerce')
    if 'train_loss' in combined_metrics_df.columns:
        combined_metrics_df['train_loss'] = pd.to_numeric(combined_metrics_df['train_loss'], errors='coerce')
    if 'val_loss' in combined_metrics_df.columns: # val_loss debería estar siempre
        combined_metrics_df['val_loss'] = pd.to_numeric(combined_metrics_df['val_loss'], errors='coerce')

    combined_metrics_df.dropna(subset=['epoch'], inplace=True)

    agg_funcs = {
        'mean_val_loss': ('val_loss', 'mean'),
        'std_val_loss': ('val_loss', lambda x: x.std(ddof=0) if len(x) > 0 else 0.0)
    }
    if 'train_loss' in combined_metrics_df.columns:
        agg_funcs['mean_train_loss'] = ('train_loss', 'mean')
        agg_funcs['std_train_loss'] = ('train_loss', lambda x: x.std(ddof=0) if len(x) > 0 else 0.0)

    epoch_summary = combined_metrics_df.groupby('epoch').agg(**agg_funcs).reset_index()
    epoch_summary.fillna({'std_train_loss': 0.0, 'std_val_loss': 0.0,
                            'mean_train_loss': np.nan, 'mean_val_loss': np.nan}, inplace=True)
    
    return epoch_summary


def plot_learning_curves(epoch_summary, title='Curva de Aprendizaje', save_path=None):
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

    plt.title(title)
    plt.xlabel('Época')
    plt.ylabel('MSE')
    if plot_legend:
        plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight')
        print(f"Gráfica guardada en: {save_path}")

    plt.show()

