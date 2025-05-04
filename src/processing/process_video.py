import os
import cv2
import json
import numpy as np
from processing.tracker import Tracker
from processing.video_handler import VideoProcessor
from utils.utils import is_inside_mask, get_lane_index, draw_lines, draw_detection, point_to_line_distance

TIME_MATCH_TOLERANCE = 0.1
MAX_DISTANCE_THRESHOLD = 20.0


def find_gt_data(lane_id, time_cross, cars):
    gt_data = {}
    filtered = list(filter(lambda car: 'laneIndex' in car and lane_id in car['laneIndex'], cars))
    filtered.sort(key=lambda car: abs(car['timeIntersectionLastShifted'] - time_cross))
    if len(filtered) > 0 and abs(filtered[0]['timeIntersectionLastShifted'] - time_cross) < TIME_MATCH_TOLERANCE:
        gt_data = filtered[0]
    return gt_data


def process_video(session_recording):
    print(f"▶️ Procesando {session_recording.session} {session_recording.recording}...")

    # === Preparar carpetas y nombres de archivo de salida ===
    save_dir = os.path.join("results", f"{session_recording.session}_{session_recording.recording}")
    os.makedirs(save_dir, exist_ok=True)
    output_json_path = os.path.join(save_dir, "tracked_cars_data.json")
    # Cambiar la extensión a .mp4 para que coincida con el códec 'mp4v'
    video_out_path = os.path.join(save_dir, "video_output.mp4")

    # === Cargar datos necesarios de la sesión ===
    contours = session_recording.contours
    lane_lines = session_recording.lane_lines
    measurement_lines = session_recording.measurement_lines
    cars = session_recording.cars
    gt_data = session_recording.gt_data

    try:
        video_processor = VideoProcessor(
            input_path=session_recording.video_path,
            output_path=video_out_path,
            fps=session_recording.fps,
            window_name=f"Tracking - {session_recording.session} {session_recording.recording}",
            show_output=False,
            save_output=True,
        )
    except IOError as e:
        print(f"Error al inicializar VideoProcessor: {e}")
        return

    video_props = video_processor.get_properties()
    width, height = video_props["width"], video_props["height"]
    fps = video_props["fps"]

    # === Cargar Tracker (YOLO) ===
    tracker = Tracker(model_path="../models/yolo11s.pt")

    # === Inicializar almacenamiento de trayectorias ===
    tracks_info = {}
    frame_num = 0
    i = 0

    # === Bucle principal de procesamiento ===
    while True:
        ret, frame = video_processor.read_frame()
        if not ret:
            break

        if frame_num > 1000:
            break

        # i += 1
        # if i < 5000:
        #     frame_num += 1
        #     continue

        detections = tracker.track(frame, class_ids=[2])

        cv2.polylines(frame, contours, isClosed=True, color=(255, 0, 0), thickness=2)
        draw_lines(frame, lane_lines, width, color=(0, 255, 255))
        draw_lines(frame, measurement_lines, width, color=(255, 0, 255))

        for track_id, class_id, (x1, y1, x2, y2) in detections:
            ref_x, ref_y = int(x1), int(y2)

            # if is_inside_mask(ref_x, ref_y, mask_bin):
            lane_id = get_lane_index((ref_x, ref_y, 1), lane_lines)
            if lane_id is None:
                continue

            draw_detection(frame, x1, y1, x2, y2, lane_id, track_id)

            if track_id not in tracks_info:
                tracks_info[track_id] = {"lane_id": lane_id, "positions": []}
            tracks_info[track_id]["positions"].append((frame_num, x1, y1, x2, y2))

        if not video_processor.show_frame(frame):
            break
        video_processor.write_frame(frame)
        frame_num += 1

    video_processor.release()

    print(f"\n🏁 Procesamiento de video terminado. Buscando punto más cercano (umbral: {MAX_DISTANCE_THRESHOLD}px) y GT...")
    output_data = {'cars': []}
    ignored_tracks_count = 0

    last_measurement_line = measurement_lines[0]
    print(f"Usando la última línea de medición: {last_measurement_line}")

    for track_id, info in tracks_info.items():
        lane_id = info['lane_id']
        positions_data = info.get('positions', [])
        if not positions_data:
            continue

        min_dist = float('inf')
        closest_frame = -1
        closest_pos = None

        for frame_idx, x1, y1, x2, y2 in positions_data:
            pos_x, pos_y = int(x1), int(y2)
            dist = point_to_line_distance(np.array([pos_x, pos_y, 1]), last_measurement_line)
            if dist < min_dist:
                min_dist = dist
                closest_frame = frame_idx
                closest_pos = (pos_x, pos_y)

        if closest_frame != -1 and min_dist < MAX_DISTANCE_THRESHOLD:
            video_time_approx = closest_frame / fps
            gt_match = find_gt_data(lane_id, video_time_approx, cars)
            output_data['cars'].append({
                'id': track_id,
                'pos': positions_data,
                'gt_speed': gt_match.get('speed', None) or None,
                'cross_time_approx_s': video_time_approx,
                'cross_frame_approx': closest_frame,
                'min_distance_to_line': min_dist,
                'closest_pos_to_line': closest_pos
            })
        else:
            # El coche nunca se acercó lo suficiente o no tenía posiciones
            ignored_tracks_count += 1
            # Opcional: Imprimir un mensaje si quieres saber qué tracks se ignoran
            # print(f"   Track {track_id} ignorado (distancia mínima: {min_dist:.2f} >= {MAX_DISTANCE_THRESHOLD} o sin puntos)")


    # === Guardar los resultados en formato JSON ===
    try:
        with open(output_json_path, 'w') as f:
            json.dump(output_data, f, indent=4)
        print(f"\n✅ Datos procesados guardados en: {output_json_path}")
        num_cars_saved = len(output_data.get('cars', []))
        num_cars_with_gt = sum(1 for car in output_data.get('cars', []) if car.get('gt_speed') is not None)
        num_cars_crossed_approx = len(output_data.get('cars', [])) # Todos los guardados tienen tiempo aproximado
        print(f"   Se guardaron datos para {num_cars_saved} coches (de {len(tracks_info)} tracks iniciales).")
        if ignored_tracks_count > 0:
            print(f"   Se ignoraron {ignored_tracks_count} tracks por no acercarse lo suficiente a la línea.")
        print(f"   Se estimó un tiempo de cruce aproximado para {num_cars_crossed_approx} coches.")
        print(f"   Se encontró velocidad GT coincidente para {num_cars_with_gt} coches.")

    except TypeError as e:
        print(f"\n❌ Error de tipo al intentar guardar en JSON: {e}. Considera usar un conversor 'default' en json.dump si usas tipos numpy.")
    except Exception as e:
        print(f"\n❌ Error inesperado al guardar el archivo JSON: {e}")
