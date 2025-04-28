import torch
import cv2
import numpy as np
from ultralytics import YOLO


class Tracker:
    def __init__(self, model_path, device="cuda"):
        self.model = YOLO(model_path).to(device)

    def track(self, frame):
        results = self.model.track(frame, persist=True, verbose=False)[0]

        outputs = []
        if results.boxes.id is not None and results.boxes.cls is not None:
            ids = results.boxes.id.int().tolist()
            classes = results.boxes.cls.int().tolist()
            boxes = results.boxes.xyxy.tolist()

            for track_id, class_id, (x1, y1, x2, y2) in zip(ids, classes, boxes):
                outputs.append((track_id, class_id, (x1, y1, x2, y2)))

        return outputs
