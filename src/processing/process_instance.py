import os
import cv2
import json
import numpy as np
import torch # Importar torch para manejar tensores de YOLO si es necesario
from processing.tracker import Tracker
from processing.video_handler import VideoProcessor
from utils.utils import point_to_line_distance

TIME_MATCH_TOLERANCE = 0.25
MAX_DISTANCE_THRESHOLD = 35.0


def find_gt_data(lane_id, time_cross, cars):
    gt_data = {}
    filtered = list(filter(lambda car: 'laneIndex' in car and lane_id in car['laneIndex'] and 'valid' in car and car['valid'], cars))
    if filtered:
        filtered.sort(key=lambda car: abs(car.get('timeIntersectionLastShifted', float('inf')) - time_cross)) # Usar .get para seguridad
        if len(filtered) > 0 and abs(filtered[0].get('timeIntersectionLastShifted', float('inf')) - time_cross) < TIME_MATCH_TOLERANCE:
            gt_data = filtered[0]
    return gt_data

def match_tracks_info(tracks_info, cars, last_measurement_line, input_fps):
    output_data = {'cars': []}
    ignored_tracks_count = 0

    for track_id, info in tracks_info.items():
        lane_id = info.get('lane_id', None)
        fdata = info.get('fdata', [])
        if not fdata:
            continue

        min_dist = float('inf')
        closest_frame = -1
        closest_pos = None

        for frame_features in fdata:
            frame_idx = frame_features['frame']
            pos_x, pos_y = frame_features['bottom']
            dist = point_to_line_distance(np.array([pos_x, pos_y, 1]), last_measurement_line)
            if dist < min_dist:
                min_dist = dist
                closest_frame = frame_idx
                closest_pos = (pos_x, pos_y)

        if closest_frame != -1 and min_dist < MAX_DISTANCE_THRESHOLD:
            # Usamos el FPS ORIGINAL (input_fps) para calcular el tiempo correctamente
            video_time_approx = closest_frame / input_fps
            gt_match = find_gt_data(lane_id, video_time_approx, cars)
            output_data['cars'].append({
                'id': track_id,
                'lane_id': lane_id,
                'fdata': fdata,
                'gt_data': {
                    'acceleration': gt_match.get('acceleration', None),
                    'id': gt_match.get('carId', None),
                    'speed': gt_match.get('speed', None),
                    'time_intersection': gt_match.get('timeIntersectionLastShifted', None),
                    'valid': gt_match.get('valid', None),
                },
                'cross': {
                    'time': video_time_approx,
                    'frame': closest_frame,
                    'pos': closest_pos,
                    'dist': min_dist,
                }
            })
        else:
            ignored_tracks_count += 1
    return output_data, ignored_tracks_count


def process_instance_video(session_recording):
    print(f"▶️ Procesando {session_recording.session} {session_recording.recording}...")

    # === Preparar carpetas y nombres de archivo de salida ===
    save_dir = os.path.join("results", f"{session_recording.session}_{session_recording.recording}")
    os.makedirs(save_dir, exist_ok=True)
    output_json_path = os.path.join(save_dir, "tracked_cars_data.json")
    video_out_path = os.path.join(save_dir, "video_output_match.mp4")

    try:
        video_processor = VideoProcessor(
            input_path=session_recording.video_path,
            output_path=video_out_path,
            fps=session_recording.fps,
            window_name=f"Tracking - {session_recording.session} {session_recording.recording}",
            show_output=True,
            save_output=True,
        )
    except IOError as e:
        print(f"Error al inicializar VideoProcessor: {e}")
        return

    tracks_info = video_processor.process(session_recording)
    print(f"\n🏁 Procesamiento de video terminado. Buscando punto más cercano (umbral: {MAX_DISTANCE_THRESHOLD}px) y GT...")

    output_data, ignored_tracks_count = match_tracks_info(tracks_info,
                                                          session_recording.cars,
                                                          session_recording.measurement_lines[0],
                                                          video_processor.input_fps)

    # === Guardar los resultados en formato JSON ===
    try:
        with open(output_json_path, 'w') as f:
            json.dump(output_data, f, indent=4)
        print(f"\n✅ Datos procesados guardados en: {output_json_path}")
        num_cars_saved = len(output_data.get('cars', []))
        num_cars_with_gt = sum(1 for car in output_data.get('cars', []) if car.get('gt_data', {}).get('speed') is not None)
        num_cars_crossed_approx = len(output_data.get('cars', []))
        print(f"   Se guardaron datos para {num_cars_saved} coches (de {len(tracks_info)} tracks iniciales).")
        if ignored_tracks_count > 0:
            print(f"   Se ignoraron {ignored_tracks_count} tracks por no acercarse lo suficiente a la línea.")
        print(f"   Se estimó un tiempo de cruce aproximado para {num_cars_crossed_approx} coches.")
        print(f"   Se encontró velocidad GT coincidente para {num_cars_with_gt} coches.")

    except TypeError as e:
        print(f"\n❌ Error de tipo al intentar guardar en JSON: {e}. Considera usar un conversor 'default' en json.dump si usas tipos numpy.")
    except Exception as e:
        print(f"\n❌ Error inesperado al guardar el archivo JSON: {e}")
