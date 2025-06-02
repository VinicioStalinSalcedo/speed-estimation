# speed-estimation

Proyecto para la estimación de velocidad de vehículos usando modelos LSTM y PyTorch Lightning sobre el dataset BrnoCompSpeed.

## Estructura del proyecto

speed-estimation/
├── .gitignore
├── README.md
└── src/
    ├── search_lstm.py
    ├── dataset/
    │   ├── tracking_datamodule.py
    │   ├── tracking_dataset.py
    │   └── dataset_info.py
    ├── models/
    │   └── lstm_model.py
    ├── processing/
    │   ├── process_instance.py
    │   ├── process_video.py
    │   ├── tracker.py
    │   └── video_handler.py
    ├── train/
    │   ├── train_fold.py
    │   └── train_lstm.py
    ├── utils/
    │   └── utils.py
    ├── results/
    ├── ims/
    ├── logs/
    └── lightning_logs/

### Descripción de carpetas y archivos

- `.gitignore`: Archivos y carpetas ignorados por git, incluyendo resultados, logs y entornos virtuales.
- `README.md`: Documentación principal del proyecto.
- `src`: Código fuente principal.
  - `search_fold.py`: Script para búsqueda de hiperparámetros del modelo LSTM.
  - `train/`: Scripts de entrenamiento.
    - `train_fold.py`: Entrenamiento por fold (validación cruzada).
    - `train_lstm.py`: Entrenamiento general del modelo LSTM.
  - `dataset/`: Manejo y carga de datos del dataset BrnoCompSpeed.
    - `tracking_datamodule.py`: DataModule `TrackingDataModule` para PyTorch Lightning.
    - `tracking_dataset.py`: Definición y utilidades del dataset `TrackingDataset`. Extracción de características y etiquetas. Carga de datos para entrenamiento y validación.
    - `dataset_info.py`: Información de sesiones, rutas y utilidades del dataset.
    - `session_recording.py`: Registro de sesiones y utilidades para el dataset. Datos del ground truth en la clase `SessionRecording`.
  - `models/`: Definición de modelos.
    - `lstm_model.py`: Implementación del modelo `TrackingLSTM` para estimación de velocidad.
  - `processing/`: Scripts de procesamiento de video y tracking.
    - `process_instance.py`: Procesamiento de video con YOLO+Tracking y segmentación de instancias. Funciones para procesar el video y guardar videos procesados y datos json. 
    - `tracker.py`: Implementación del sistema de tracking de vehículos con YOLO y procesamiento de los resultados retornados como bbox y máscaras.
    - `video_handler.py`: Utilidades para manejo y procesamiento de archivos de video. Función process para realizar el proceso de seguimiento, segmentación de instancias, visualización y validación de la detección.
  - `utils/`: Funciones utilitarias y de apoyo.
    - `utils.py`: Funciones auxiliares usadas en distintos módulos del proyecto.
  - `results/`: Resultados de experimentos y validaciones.
  - `ims/`: Imágenes y visualizaciones generadas.
  - `logs/`: Logs de entrenamiento.

## Dependencias principales

- Python 3.10.12
- PyTorch
- PyTorch Lightning
- Optuna

Instala las dependencias con:

pip install -r requirements.txt

## Uso

1. Configura las rutas del dataset en `src/dataset/dataset_info.py` si es necesario.
2. Ejecuta los scripts de entrenamiento o búsqueda de hiperparámetros desde la carpeta `src/`:
   - Para entrenar:
     python train/train_lstm.py
   - Para búsqueda de hiperparámetros:
     python search_lstm.py
3. Los resultados y logs se almacenarán en las carpetas correspondientes dentro de `src/`.

## Dataset

El proyecto utiliza el dataset BrnoCompSpeed. Para obtener el dataset, comunicarse con Jakub Špaňhel - ispanhel@fit.vutbr.cz.

Posteriormente configurar correctamente las rutas en `dataset_info.py`.

## Créditos

Desarrollado por Vinicio Salcedo.
