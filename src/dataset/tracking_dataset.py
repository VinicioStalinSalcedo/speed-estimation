import torch
from torch.utils.data import Dataset


class TrackingDataset(Dataset):
    def __init__(self, data):
        self.samples = []
        for d in data:
            bboxes = [(x1, y1, x2, y2) for (fid, x1, y1, x2, y2) in d['pos']]
            speed = d['gt_speed']
            if bboxes and speed is not None and len(bboxes) >= 50:
                self.samples.append((
                    torch.tensor(bboxes, dtype=torch.float32),
                    torch.tensor(speed, dtype=torch.float32),
                    len(bboxes)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
