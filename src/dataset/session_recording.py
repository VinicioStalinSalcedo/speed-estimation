from dataset.dataset_info import getPathForRecording
import os
import cv2
import pickle


class SessionRecording:
    def __init__(self, session, recording):
        self.session = session
        self.recording = recording
        self.base_path = os.path.join(getPathForRecording(session, recording))

        self.video_path = os.path.join(self.base_path, "video.avi")
        self.mask_path = os.path.join(self.base_path, "video_mask.png")
        self.gt_path = os.path.join(self.base_path, "gt_data.pkl")

        self.mask = None
        self.mask_bin = None
        self.contours = None
        self.gt_data = None
        self.fps = None
        self.lane_lines = []
        self.measurement_lines = []
        self.cars = []

    def load_data(self):
        self.mask = cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE)
        with open(self.gt_path, "rb") as f:
            self.gt_data = pickle.load(f, encoding="latin1")
            self.lane_lines = self.gt_data["laneDivLines"]
            self.measurement_lines = self.gt_data["measurementLines"]
            self.fps = self.gt_data["fps"]
            self.cars = self.gt_data["cars"]
        self._extract_contours()

    def _extract_contours(self):
        _, self.mask_bin = cv2.threshold(self.mask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(self.mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.contours = contours
