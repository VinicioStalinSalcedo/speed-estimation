import os
import cv2
import json
import numpy as np
import torch # Importar torch para manejar tensores de YOLO si es necesario
from processing.tracker import Tracker
from processing.video_handler import VideoProcessor
from utils.utils import is_inside_mask, get_lane_index, draw_lines, draw_detection, point_to_line_distance

from ultralytics.utils.plotting import Annotator, colors

TIME_MATCH_TOLERANCE = 0.25
MAX_DISTANCE_THRESHOLD = 35.0

# --- Parámetros de suavizado de máscara ---
GAUSSIAN_SMOOTH_KERNEL_SIZE = (5, 5) # Tamaño del kernel Gaussiano. Ajustar según sea necesario. Debe ser impar (ej. (3,3), (5,5), (7,7))
GAUSSIAN_SMOOTH_SIGMA_X = 0 # SigmaX (desviación estándar en X). 0 significa calcularlo a partir del kernel.
SMOOTH_THRESHOLD_VALUE = 127 # Umbral para re-binarizar después del blur (para uint8)
# --- Fin Parámetros de suavizado ---


def find_gt_data(lane_id, time_cross, cars):
    gt_data = {}
    filtered = list(filter(lambda car: 'laneIndex' in car and lane_id in car['laneIndex'] and 'valid' in car and car['valid'], cars))
    if filtered:
        filtered.sort(key=lambda car: abs(car.get('timeIntersectionLastShifted', float('inf')) - time_cross)) # Usar .get para seguridad
        if len(filtered) > 0 and abs(filtered[0].get('timeIntersectionLastShifted', float('inf')) - time_cross) < TIME_MATCH_TOLERANCE:
            gt_data = filtered[0]
    return gt_data

def get_bottom_point_stable(mask_xy, quantile=0.1):
    """
    Toma el 10% inferior de puntos en Y y promedia en X para reducir ruido.
    """
    pts = np.array(mask_xy)
    y_thresh = np.quantile(pts[:, 1], 1 - quantile)
    bottom_pts = pts[pts[:, 1] >= y_thresh]
    x_avg = np.mean(bottom_pts[:, 0])
    y_max = np.max(bottom_pts[:, 1])
    return np.array([x_avg, y_max])


def process_instance_video(session_recording):
    print(f"▶️ Procesando {session_recording.session} {session_recording.recording}...")

    # === Preparar carpetas y nombres de archivo de salida ===
    save_dir = os.path.join("results_instance", f"{session_recording.session}_{session_recording.recording}")
    os.makedirs(save_dir, exist_ok=True)
    output_json_path = os.path.join(save_dir, "tracked_cars_data.json")
    video_out_path = os.path.join(save_dir, "video_output.mp4")

    # === Cargar datos necesarios de la sesión ===
    contours = session_recording.contours
    lane_lines = session_recording.lane_lines
    measurement_lines = session_recording.measurement_lines
    cars = session_recording.cars
    mask_bin = session_recording.mask_bin

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

    video_props = video_processor.get_properties()
    width, height = video_props["width"], video_props["height"]
    input_fps = video_props["input_fps"] # FPS original para cálculos de tiempo
    output_fps = video_props["output_fps"] # FPS deseado para salida

    # === Calcular ratio para salto de fotogramas ===
    if output_fps > 0 and input_fps > output_fps:
        frame_skip_ratio = round(input_fps / output_fps)
        print(f"   Saltando fotogramas: Escribiendo 1 de cada {frame_skip_ratio} fotogramas (Input: {input_fps:.2f} FPS, Output: {output_fps:.2f} FPS).")
    else:
        frame_skip_ratio = 1 # No saltar si output_fps >= input_fps o es 0
        print(f"   No se saltarán fotogramas (Input: {input_fps:.2f} FPS, Output: {output_fps:.2f} FPS).")
    # === Fin cálculo ratio ===

    # === Cargar Tracker (YOLO) ===
    tracker = Tracker(model_path="../models/yolo11s-seg.pt")

    # === Inicializar almacenamiento de trayectorias ===
    tracks_info = {}
    frame_num = 0
    i = 0

    # === Bucle principal de procesamiento ===
    while True:
        ret, frame = video_processor.read_frame()
        if not ret:
            break

        i += 1
        if i < 100:
            frame_num += 1
            continue

        if i > 525:
            break

        annotator = Annotator(frame, line_width=2, )
        
        detections = tracker.instance_track(frame, class_ids=[2])

        cv2.polylines(frame, contours, isClosed=True, color=(255, 0, 0), thickness=2)
        draw_lines(frame, lane_lines, width, color=(0, 255, 255))
        draw_lines(frame, measurement_lines, width, color=(255, 0, 255))

        for track_id, class_id, (x1, y1, x2, y2), mask_points in detections:
            if len(mask_points) == 0:
                continue

            try:
                 rotated_rect = cv2.minAreaRect(mask_points)
            except Exception as e:
                 print(f"Error calculating minAreaRect for track {track_id} at frame {frame_num}: {e}")
                 continue # Skip this detection if calculation fails

            (r_center_x, r_center_y), (r_width, r_height), r_angle = rotated_rect
            rotated_bbox = cv2.boxPoints(rotated_rect).tolist()

            # Calcular área de la máscara y del recuadro rotado
            mask_area = float(cv2.contourArea(mask_points))
            rotated_bbox_area = float(r_width * r_height)

            # Calcular Aspect Ratio del recuadro rotado (evitar división por cero)
            rotated_bbox_aspect_ratio = float(max(r_width, r_height) / (min(r_width, r_height) + 1e-6))

            # Centro del bbox alineado original
            aligned_center_x, aligned_center_y = float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)
            aligned_bbox_width, aligned_bbox_height = float(x2 - x1), float(y2 - y1)
            aligned_bbox_area = aligned_bbox_width * aligned_bbox_height
            aligned_bbox_aspect_ratio = float(max(aligned_bbox_width, aligned_bbox_height) / (min(aligned_bbox_width, aligned_bbox_height) + 1e-6))

            bottom = mask_points[np.argmax(mask_points[:, 1])].tolist()
            # bottom = get_bottom_point_stable(mask_points, quantile=0.1)
            # bottom = [float((x1 + x2) / 2.0), float(y2)]  # [center_x, bottom_y]

            annotator.seg_bbox(mask=mask_points, mask_color=colors(track_id, True))

            if is_inside_mask(bottom[0], bottom[1], mask_bin):
                lane_id = get_lane_index((bottom[0], bottom[1], 1), lane_lines)
                if lane_id is None:
                    continue


                if track_id not in tracks_info:
                    tracks_info[track_id] = {"lane_id": lane_id, "fdata": []}

                frame_features = {
                    'frame': frame_num,
                    'bbox': {
                        'pos': [[x1, y1], [x2, y2]],
                        'center': [aligned_center_x, aligned_center_y],
                        "dims": [aligned_bbox_width, aligned_bbox_height],
                        "area": aligned_bbox_area,
                        "aspect_ratio": aligned_bbox_aspect_ratio,
                    },
                    'rotated_bbox': {
                        'pos': rotated_bbox,
                        'center': [float(r_center_x), float(r_center_y)],
                        'dims': [float(r_width), float(r_height)],
                        'area': rotated_bbox_area,
                        "angle": float(r_angle),
                        'aspect_ratio': rotated_bbox_aspect_ratio
                    },
                    'bottom': bottom,
                    'mask_area': mask_area,
                }

                tracks_info[track_id]["fdata"].append(frame_features)

                draw_detection(frame, x1, y1, x2, y2, lane_id, track_id)

                # 5. Dibujar la información en el frame de visualización
                # Obtener los 4 vértices del recuadro rotado para dibujarlo
                box_points_np = cv2.boxPoints(rotated_rect) # Numpy array de shape (4, 2)
                # Convertir a enteros para cv2.drawContours
                box_points_int = np.int0(box_points_np)

                # Dibujar el contorno del recuadro rotado (ej. en verde brillante)
                cv2.drawContours(frame, [box_points_int], 0, (255, 0, 255), 2) # Color verde brillante

                # Dibujar el punto más bajo de la máscara (ej. círculo rojo)
                if bottom[0] != -1.0:
                    cv2.circle(frame, (int(bottom[0]), int(bottom[1])), 5, (0, 0, 255), -1)

                # Añadir la etiqueta (ID y carril) - usar el centro del recuadro rotado para posicionar
                label = f'ID: {track_id} - L: {lane_id}'
                # Calcular una posición para el texto, un poco por encima del recuadro rotado
                # Usar el centro del recuadro rotado como referencia
                label_pos_x = int(r_center_x)
                label_pos_y = int(r_center_y - max(r_width, r_height)/2 - 5) # 5 píxeles arriba del borde superior (usando la dimensión mayor)

                # Asegurarse de que la posición del texto no esté fuera de la imagen
                label_pos_x = max(0, label_pos_x)
                label_pos_y = max(15, label_pos_y) # Mínimo 15 para no salir por arriba

                cv2.putText(frame, label, (label_pos_x, label_pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        if not video_processor.show_frame(frame):
            break

        # === Escribir frame SOLO si cumple el ratio de salto ===
        if frame_num % frame_skip_ratio == 0:
            video_processor.write_frame(frame) # write_frame redimensiona internamente
        # === Fin escritura selectiva ===

        frame_num += 1

    video_processor.release()

    print(f"\n🏁 Procesamiento de video terminado. Buscando punto más cercano (umbral: {MAX_DISTANCE_THRESHOLD}px) y GT...")
    output_data = {'cars': []}
    ignored_tracks_count = 0

    last_measurement_line = measurement_lines[0]
    print(f"Usando la última línea de medición: {last_measurement_line}")

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
