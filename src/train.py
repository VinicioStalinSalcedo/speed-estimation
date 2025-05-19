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

    input_size = len(data_list[0]['features'][0])

    data_module, model, trainer, logger = train_lstm_pipeline(
        # Dataset
        data_list,

        # DataModule params
        batch_size=4,
        split_ratio=(0.8, 0.1, 0.1),
        num_workers=4,

        # LSTM params
        input_size=input_size,
        hidden_size=64,
        num_layers=2,
        output_size=1,
        learning_rate=1e-3,
        dropout=0.0,

        # Logger
        log_path='tracking_lstm_sc',

        # Trainer
        patience=7,
        epochs=200,
        log_every_n_steps=10,

        progress_bar=True,
    )

    trainer.test(model, datamodule=data_module)


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
    train()

