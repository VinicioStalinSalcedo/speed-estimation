import os
import cv2
import pandas as pd
import numpy as np
from processing.tracker import Tracker


def process_video(session_recording):
    print(f"▶️ Procesando {session_recording.session} {session_recording.recording}...")

    # === Preparar carpetas de salida ===
    save_dir = os.path.join("results", f"{session_recording.session}_{session_recording.recording}")
    os.makedirs(save_dir, exist_ok=True)
    csv_out_path = os.path.join(save_dir, "tracks.csv")

    # === Preparar video ===
    cap = cv2.VideoCapture(session_recording.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # === Cargar YOLO ===
    tracker = Tracker(model_path="../models/yolo11s.pt", device="cuda")

    # === Measurement Line para cruce ===
    measurement_line = np.array(eval(session_recording.gt_data["measurementLines"][0]))

    # === Inicializar almacenamiento ===
    tracks_info = {}
    final_detections = []

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = tracker.track(frame)

        for track_id, class_id, (x1, y1, x2, y2) in detections:
            if class_id not in [2, 5]:  # solo car y bus
                continue

            # Punto de referencia: esquina inferior izquierda
            xb, yb = int(x1), int(y2)

            # Verificar si está dentro de la máscara
            if cv2.pointPolygonTest(session_recording.contours[0], (xb, yb), False) < 0:
                continue

            # Calcular el carril
            lane_id = determine_lane((xb, yb), session_recording.gt_data["laneDivLines"])
            if lane_id is None:
                continue

            # Guardar trayectoria
            if track_id not in tracks_info:
                tracks_info[track_id] = {"points": [], "lane_id": lane_id}
            tracks_info[track_id]["points"].append((frame_num, xb, yb))

        frame_num += 1

    cap.release()

    # === Ahora detectar cruce de measurementLine para cada track ===
    for track_id, info in tracks_info.items():
        points = info["points"]
        prev_val = None
        cross_frame = None

        for fnum, x, y in points:
            val = np.dot(measurement_line, np.array([x, y, 1]))

            if prev_val is not None and val * prev_val < 0:
                cross_frame = fnum
                break

            prev_val = val

        if cross_frame is not None:
            video_time = cross_frame / fps
            final_detections.append({
                "track_id": track_id,
                "lane_id": info["lane_id"],
                "frame_cross": cross_frame,
                "video_time_cross": video_time
            })

    # === Guardar a CSV ===
    df = pd.DataFrame(final_detections)
    df.to_csv(csv_out_path, index=False)
    print(f"✅ Guardado CSV: {csv_out_path}")


# Función auxiliar para determinar el lane
def determine_lane(point, lane_lines):
    x, y = point
    p = np.array([x, y, 1])
    parsed_lines = [np.array(eval(l)) for l in lane_lines]

    for i in range(len(parsed_lines) - 1):
        side1 = np.sign(np.dot(parsed_lines[i], p))
        side2 = np.sign(np.dot(parsed_lines[i + 1], p))
        if side1 != side2:
            return i
    return None
