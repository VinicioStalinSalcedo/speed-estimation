from train.train_lstm import train_lstm_pipeline
import numpy as np
import pandas as pd
import itertools
import json
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


def search():
    data_list = []
    for session, recording in ALL_VIDEOS:
        with open(f'results_instance/{session}_{recording}/tracked_cars_data.json', 'r') as f:
            cars = json.load(f)['cars']
            valid_cars = list(filter(lambda car: car.get('gt_data', {}).get('speed', None) is not None, cars))
            data_list.extend([extract_features(car) for car in valid_cars])


    param_grid = {
        'hidden_size': [32, 64, 128],
        'num_layers': [1, 2, 3],
        'dropout': [0.0, 0.2, 0.5],
        'learning_rate': [1e-3, 1e-4, 1e-5],
    }

    input_size = len(data_list[0]['features'][0])
    results = []

    for hs, nl, do, lr in itertools.product(*param_grid.values()):
        exp_name = f'hs{hs}_nl{nl}_do{do}_lr{lr}'

        data_module, model, trainer, logger = train_lstm_pipeline(
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

            # Logger
            log_path=exp_name,

            # Trainer
            patience=7,
            epochs=200,
            log_every_n_steps=10,

            progress_bar=True,
        )

        metrics = trainer.callback_metrics
        val_loss = metrics["val_loss"].item()
        val_mse = metrics["val_mse"].item()
        val_mae = metrics["val_mae"].item()
        val_rmse = np.sqrt(val_mse)

        results.append({
            "exp": exp_name,
            "hidden_size": hs,
            "num_layers": nl,
            "dropout": do,
            "lr": lr,
            "batch_size": 4,
            "val_loss": val_loss,
            "val_mae": val_mae,
            "val_rmse": val_rmse
        })

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("val_loss").reset_index(drop=True)
    df_results.to_csv("results_search/search_results.csv", index=False)


def extract_features(car):
    speed = car.get('gt_data', {}).get('speed', None)
    features = []
    fdatas = car['fdata']
    for fdata in fdatas:
        [x1, y1], [x2, y2] = fdata['bbox']['pos']
        bbox_center = fdata['bbox']['center']
        bbox_width, bbox_height = fdata['bbox']['dims']
        bbox_area, bbox_aspect_ratio = fdata['bbox']['area'], fdata['bbox']['aspect_ratio'] # may change aspect_ratio for (width / height)
        [rx1, ry1], [rx2, ry2], [rx3, ry3], [rx4, ry4] = fdata['rotated_bbox']['pos']
        rbbox_center = fdata['rotated_bbox']['center']
        rbbox_width, rbbox_height = fdata['rotated_bbox']['dims']
        rbbox_area, rbbox_angle, rbbox_aspect_ratio = fdata['rotated_bbox']['area'], fdata['rotated_bbox']['angle'], fdata['rotated_bbox']['aspect_ratio']
        bottom_x, bottom_y = fdata['bottom']
        mask_area = fdata['mask_area']

        features.append([
            # x1, y1, x2, y2,
            # bbox_center[0], bbox_center[1],
            # bbox_width, bbox_height,
            bottom_x, bottom_y,
            bbox_aspect_ratio,
            # rx1, ry1, rx2, ry2, rx3, ry3, rx4, ry4,
            # rbbox_center[0], rbbox_center[1],
            # rbbox_width, rbbox_height,
            rbbox_angle,
            rbbox_aspect_ratio,
            mask_area,
        ])
    return {
        'features': features,
        'speed': speed
    }


if __name__ == '__main__':
    search()
