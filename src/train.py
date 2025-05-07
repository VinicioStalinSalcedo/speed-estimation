from train.train_lstm import train_lstm_pipeline
import json
ALL_VIDEOS = [
    # ('session0', 'center'),
    # ('session0', 'left'),
    # ('session0', 'right'),
    ('session1', 'center'),
    # ('session1', 'left'),
    # ('session1', 'right'),
    ('session2', 'center'),
    # ('session2', 'left'),
    # ('session2', 'right'),
    ('session3', 'center'),
    # ('session3', 'left'),
    # ('session3', 'right'),
    ('session4', 'center'),
    # ('session4', 'left'),
    # ('session4', 'right'),
    ('session5', 'center'),
    # ('session5', 'left'),
    # ('session5', 'right'),
    # ('session6', 'center'),
    # ('session6', 'left'),
    # ('session6', 'right'),
]


def train():
    data_list = []
    for session, recording in ALL_VIDEOS:
        with open(f'results/{session}_{recording}/tracked_cars_data.json', 'r') as f:
            cars = json.load(f)['cars']
            valid_cars = list(filter(lambda car: car['gt_speed'] is not None, cars))
            data_list.extend(valid_cars)

    train_lstm_pipeline(data_list)


if __name__ == '__main__':
    train()
