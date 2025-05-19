import torch
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils import ops
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

    def instance_track(self, frame, class_ids=None):
        try:
            results = self.model.track(frame, persist=True, verbose=False)[0]
        except Exception as e:
            print(f"***!!! Error al ejecutar YOLO: {e} !!!***")
            return []

        outputs = []
        if results.boxes.id is not None and results.boxes.cls is not None:
            self.smooth_masks(results)

            ids = results.boxes.id.int().tolist()
            classes = results.boxes.cls.int().tolist()
            boxes = results.boxes.xyxy.tolist()
            masks = results.masks.xy

            for track_id, class_id, (x1, y1, x2, y2), mask in zip(ids, classes, boxes, masks):
                if class_ids is not None and class_id not in class_ids:
                    continue

                outputs.append((track_id, class_id, (x1, y1, x2, y2), mask))

        return outputs

    def smooth_masks(self, results, kernel_size=20, blur_kernel_size=25, threshold_value=127):
        if results.masks is not None and results.masks.data is not None:
            original_mask_tensor = results.masks.data  # Tensor de PyTorch (N, H, W), booleano

            mask_data_np_uint8 = original_mask_tensor.cpu().numpy().astype(np.uint8) * 255

            kernel = np.ones((kernel_size, kernel_size), np.uint8)

            smoothed_masks_list_np_uint8 = []
            for i in range(mask_data_np_uint8.shape[0]):
                current_mask_uint8 = mask_data_np_uint8[i]

                # 1. Aplicar Apertura Morfológica
                morph_mask = cv2.morphologyEx(current_mask_uint8, cv2.MORPH_OPEN, kernel)

                # 2. Aplicar Cierre Morfológico sobre la máscara resultante de la apertura
                morph_mask = cv2.morphologyEx(morph_mask, cv2.MORPH_CLOSE, kernel)

                # Opcional: Añadir Gaussian Blur y Threshold si aún se necesita suavizado de contorno
                blurred_mask = cv2.GaussianBlur(morph_mask, (blur_kernel_size, blur_kernel_size), 0)
                _, final_smoothed_mask_binary = cv2.threshold(blurred_mask, threshold_value, 255, cv2.THRESH_BINARY)

                smoothed_masks_list_np_uint8.append(final_smoothed_mask_binary)

            if smoothed_masks_list_np_uint8:
                smoothed_masks_np_uint8 = np.array(smoothed_masks_list_np_uint8, dtype=np.uint8)
                smoothed_masks_np_bool = smoothed_masks_np_uint8.astype(bool)
                new_mask_tensor = torch.from_numpy(smoothed_masks_np_bool).to(original_mask_tensor.device)
                results.masks.data = new_mask_tensor
