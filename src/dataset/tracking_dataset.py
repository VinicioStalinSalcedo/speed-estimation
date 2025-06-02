import numpy as np
import torch
from torch.utils.data import Dataset
import json


class TrackingDataset(Dataset):
    def __init__(self, data_list):
        self.samples = []
        for d in data_list:
            features = d.get('features', [])
            speed = d.get('speed', None)
            length = len(features)
            self.samples.append((torch.tensor(features, dtype=torch.float32),
                                 torch.tensor(speed, dtype=torch.float32),
                                 length))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def extract_features(car, feature_path='all_features'):
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

        if feature_path == 'all_features':
            features.append([
                x1, y1, x2, y2,
                bbox_center[0], bbox_center[1],
                bbox_width, bbox_height,

                bottom_x, bottom_y,
                bbox_aspect_ratio,

                rx1, ry1, rx2, ry2, rx3, ry3, rx4, ry4,
                rbbox_center[0], rbbox_center[1],
                rbbox_width, rbbox_height,

                rbbox_angle,
                rbbox_aspect_ratio,
                mask_area,
            ])

        elif feature_path == 'fbbox':
            features.append([
                x1, y1, x2, y2
            ])

        elif feature_path == 'faspect':
            features.append([
                bottom_x, bottom_y,
                bbox_aspect_ratio,
                rbbox_angle,
                rbbox_aspect_ratio,
                mask_area,
            ])
        elif feature_path == 'fbbox_angle':
            features.append([
                x1, y1, x2, y2,
                rbbox_angle,
                mask_area,
            ])
        elif feature_path == 'fall':
            features.append([
                x1, y1, x2, y2,

                bottom_x, bottom_y,
                bbox_aspect_ratio,

                rx1, ry1, rx2, ry2, rx3, ry3, rx4, ry4,

                rbbox_angle,
                rbbox_aspect_ratio,
                mask_area,
            ])

    return {
        'features': features,
        'speed': speed
    }


def load_data_list(videos, results_folder='results', feature_path='all_features'):
    data_list = []
    for session, recording in videos:
        with open(f'{results_folder}/{session}_{recording}/tracked_cars_data.json', 'r') as f:
            cars = json.load(f)['cars']
            valid_cars = list(filter(lambda car: car.get('gt_data', {}).get('speed', None) is not None, cars))
            data_list.extend([extract_features(car, feature_path) for car in valid_cars])

    return data_list


