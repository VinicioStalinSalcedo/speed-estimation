from train.train_lstm import train_lstm_pipeline
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


def train():
    data_list = []
    for session, recording in ALL_VIDEOS:
        with open(f'results_instance/{session}_{recording}/tracked_cars_data.json', 'r') as f:
            cars = json.load(f)['cars']
            valid_cars = list(filter(lambda car: car.get('gt_data', {}).get('speed', None) is not None, cars))
            data_list.extend([extract_features(car) for car in valid_cars])

    train_lstm_pipeline(
        # Dataset
        data_list,

        # DataModule params
        batch_size=4,
        split_ratio=(0.7, 0.15, 0.15),
        num_workers=4,

        # LSTM params
        input_size=4,
        hidden_size=64,
        num_layers=2,
        output_size=1,
        learning_rate=1e-3,
        dropout=0.1,

        # Logger
        log_path='tracking_lstm_sc',

        # Trainer
        patience=7,
        epochs=200,
        log_every_n_steps=10,
    )


def extract_features(car):
    features = []
    fdatas = car['fdata']
    for fdata in fdatas:
        [x1, y1], [x2, y2] = fdata['bbox']['pos']
        features.append([
            x1, y1, x2, y2
        ])
    return {
        'features': features,
        'speed': car.get('gt_data', {}).get('speed', None)
    }


if __name__ == '__main__':
    train()
