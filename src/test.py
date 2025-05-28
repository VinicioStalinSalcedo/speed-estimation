from dataset.dataset_info import getPathForRecording
import json
import os
import pickle

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


def test():
    real = 0
    tracked = 0
    for session, recording in ALL_VIDEOS:
        try:
            base_path = os.path.join(getPathForRecording(session, recording))
            gt_path = os.path.join(base_path, "gt_data.pkl")
            with open(gt_path, "rb") as f:
                gt_cars = list(filter(lambda c: c['valid'], pickle.load(f, encoding="latin1")['cars']))

            results_path = f'results_instance/{session}_{recording}/tracked_cars_data.json'
            with open(results_path, 'r') as f:
                cars = list(filter(lambda car: car.get('gt_data', {}).get('speed', None) is not None, json.load(f)['cars']))

            print(f"Real {len(gt_cars)}, tracked {len(cars)}")
            real += len(gt_cars)
            tracked += len(cars)
        except Exception as e:
            print(f"Error in {session}_{recording}: {e}")
            continue

    print(f'Total real: {real}, tracked: {tracked},,,')



if __name__ == '__main__':
    test()




