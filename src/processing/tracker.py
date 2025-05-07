import torch
import cv2
import numpy as np
from ultralytics import YOLO
from utils.utils import get_device


class Tracker:
    def __init__(self, model_path):
        device = get_device()
        self.model = YOLO(model_path).to(device)

    def track(self, frame, class_ids=None):
        try:
            results = self.model.track(frame, persist=True, verbose=False)[0]
        except Exception as e:
            print(f"***!!! Error al ejecutar YOLO: {e} !!!***")
            return []

        outputs = []
        if results.boxes.id is not None and results.boxes.cls is not None:
            ids = results.boxes.id.int().tolist()
            classes = results.boxes.cls.int().tolist()
            boxes = results.boxes.xyxy.tolist()

            for track_id, class_id, (x1, y1, x2, y2) in zip(ids, classes, boxes):
                if class_ids is not None and class_id not in class_ids:
                    continue
                outputs.append((track_id, class_id, (x1, y1, x2, y2)))

        return outputs
