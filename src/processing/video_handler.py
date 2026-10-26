import cv2
import os
from tqdm import tqdm
import numpy as np
from processing.tracker import Tracker
from utils.utils import is_inside_mask, get_lane_index, draw_lines, draw_detection, draw_instance, draw_bottom
from ultralytics.utils.plotting import Annotator, colors
# color_b = (102, 102, 255) # rosado durazno
# color_b = (255, 153, 51) # celeste oscuro
# color_b = (102, 255, 102) # verde claro
# color_b = (102, 178, 255) # tomate claro

bbox_color = (102, 255, 102)
# mask_color = (255, 153, 51)
mask_color = colors(3, True)
text_color = (50, 50, 50)
crosshair_color = (0, 255, 255)
aspect_ratio_rect_color = (255, 165, 0)


class VideoProcessor:
    def __init__(self, input_path, output_path, fps, window_name="Processing", show_output=True, save_output=True):
        self.input_path = input_path
        self.output_path = output_path
        self.window_name = window_name
        self.show_output = show_output
        self.save_output = save_output

        self.cap = cv2.VideoCapture(self.input_path)
        if not self.cap.isOpened():
            raise IOError(f"No se pudo abrir el video de entrada: {self.input_path}")

        # Propiedades del video de ENTRADA
        self.input_fps = fps or self.cap.get(cv2.CAP_PROP_FPS)
        self.input_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.input_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # === Configuración para la SALIDA comprimida ===
        self.output_fps = 50.0
        self.output_width = 1280 # 960 # 854 # 1280 # 1920
        self.output_height = 720 # 540 # 480 # 720 # 1080
        self.output_size = (self.output_width, self.output_height)
        # === Fin Configuración para la SALIDA ===

        self.out_video = None
        if self.save_output:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            # Usar 'mp4v' para archivos .mp4, es más eficiente que XVID
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # Inicializar VideoWriter con los NUEVOS parámetros de salida
            self.out_video = cv2.VideoWriter(self.output_path, fourcc, self.output_fps, self.output_size)
            if not self.out_video.isOpened():
                 print(f"⚠️ Advertencia: No se pudo abrir el archivo de video de salida: {self.output_path}")
                 self.save_output = False # Desactivar guardado si falla la apertura
                 self.out_video = None

        if self.show_output:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            # Podrías querer ajustar el tamaño de la ventana de visualización también
            # cv2.resizeWindow(self.window_name, self.output_width // 2, self.output_height // 2)

        print(f"📼 Procesando {self.total_frames} frames desde [{self.input_width}x{self.input_height}] @ {self.input_fps:.2f} FPS...")
        if self.save_output and self.out_video:
             print(f"💾 Guardando video comprimido en [{self.output_width}x{self.output_height}] @ {self.output_fps:.2f} FPS...")
        else:
            print("💾 Guardado de video desactivado.")

        print(f"   Visualización: {'Activada' if self.show_output else 'Desactivada'}")

        self.pbar = tqdm(total=self.total_frames, desc="Procesando video")

    def read_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.pbar.update(1)
        return ret, frame

    def write_frame(self, frame):
        if self.save_output and self.out_video:
            # --- Redimensionar el frame ANTES de guardarlo ---
            frame_resized = cv2.resize(frame, self.output_size, interpolation=cv2.INTER_AREA)
            self.out_video.write(frame_resized)

    def show_frame(self, frame):
        # La visualización sigue mostrando el frame original (no redimensionado)
        should_continue = True
        if self.show_output:
            cv2.imshow(self.window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                should_continue = False
        return should_continue

    def release(self):
        print("\nLiberando recursos...")
        self.pbar.close()
        self.cap.release()

        if self.save_output and self.out_video:
            self.out_video.release()
            print(f"✅ Video procesado y comprimido guardado en: {self.output_path}")

        if self.show_output:
            cv2.destroyAllWindows()

    def get_properties(self):
        return {
            "input_fps": self.input_fps,
            "output_fps": self.output_fps,
            "width": self.input_width,
            "height": self.input_height,
            "total_frames": self.total_frames
        }

    def get_bottom_point_stable(self, mask_xy, quantile=0.1):
        """
        Toma el 10% inferior de puntos en Y y promedia en X para reducir ruido.
        """
        pts = np.array(mask_xy)
        y_thresh = np.quantile(pts[:, 1], 1 - quantile)
        bottom_pts = pts[pts[:, 1] >= y_thresh]
        x_avg = np.mean(bottom_pts[:, 0])
        y_max = np.max(bottom_pts[:, 1])
        return np.array([x_avg, y_max])

    def calculate_skip_ratio(self):
        if self.output_fps > 0 and self.input_fps > self.output_fps:
            frame_skip_ratio = round(self.input_fps / self.output_fps)
            print(
                f"   Saltando fotogramas: Escribiendo 1 de cada {frame_skip_ratio} fotogramas (Input: {self.input_fps:.2f} FPS, Output: {self.output_fps:.2f} FPS).")
        else:
            frame_skip_ratio = 1  # No saltar si output_fps >= input_fps o es 0
            print(f"   No se saltarán fotogramas (Input: {self.input_fps:.2f} FPS, Output: {self.output_fps:.2f} FPS).")
        return frame_skip_ratio

    def get_frame_features(self, x1, y1, x2, y2, rotated_rect, mask_points, frame_num, bottom):
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
        return frame_features

    def process(self, session_recording):
        # === Cargar datos necesarios de la sesión ===
        contours = session_recording.contours
        lane_lines = session_recording.lane_lines
        measurement_lines = session_recording.measurement_lines
        mask_bin = session_recording.mask_bin

        frame_skip_ratio = self.calculate_skip_ratio()

        # === Cargar Tracker (YOLO) ===
        tracker = Tracker(model_path="../models/yolo11s-seg.pt")

        # === Inicializar almacenamiento de trayectorias ===
        tracks_info = {}
        frame_num = 0

        i = 0

        # === Bucle principal de procesamiento ===
        while True:
            ret, frame = self.read_frame()
            if not ret:
                break

            i += 1
            if i < 2725:
                frame_num += 1
                continue

            annotator = Annotator(frame,
                                  line_width=3,
                                  font='ims/OpenSans-Regular.ttf', # no está funcionando la fuente
                                  )

            detections = tracker.instance_track(frame, class_ids=[2])

            # cv2.polylines(frame, contours, isClosed=True, color=(255, 0, 0), thickness=2)
            # draw_lines(frame, lane_lines, self.input_width, color=(0, 255, 255))
            draw_lines(frame, measurement_lines, self.input_width, color=(255, 0, 255))

            for track_id, class_id, (x1, y1, x2, y2), mask_points in detections:
                if len(mask_points) == 0:
                    continue

                try:
                    rotated_rect = cv2.minAreaRect(mask_points)
                except Exception as e:
                    print(f"Error calculating minAreaRect for track {track_id} at frame {frame_num}: {e}")
                    continue  # Skip this detection if calculation fails

                bottom = mask_points[np.argmax(mask_points[:, 1])].tolist()

                # annotator.seg_bbox(mask=mask_points,
                #                    label=f"Real {71.11}, Predicted {69.93} km/h",
                #                    mask_color=mask_color,
                #                    txt_color=text_color
                #                    )

                if is_inside_mask(bottom[0], bottom[1], mask_bin):
                    lane_id = get_lane_index((bottom[0], bottom[1], 1), lane_lines)
                    if lane_id is None:
                        continue

                    if track_id not in tracks_info:
                        tracks_info[track_id] = {"lane_id": lane_id, "fdata": []}

                    frame_features = self.get_frame_features(x1, y1, x2, y2, rotated_rect, mask_points, frame_num, bottom)
                    tracks_info[track_id]["fdata"].append(frame_features)

                    tracks_info[track_id]['real_speed'] = '71.11'
                    tracks_info[track_id]['speed'] = '-'

                    # annotator.box_label([x1, y1, x2, y2],
                    #                    label=f"Carril {lane_id}, ID {track_id}",
                    #                    color=bbox_color,
                    #                    txt_color=text_color
                    #                    )
                    annotator.seg_bbox(mask=mask_points,
                                       label=f"Carril {lane_id}, ID {track_id}",
                                       mask_color=mask_color,
                                       txt_color=text_color
                                       )
                # elif track_id in tracks_info and 'speed' in tracks_info[track_id]:
                #     tracks_info[track_id]['speed'] = '69.93'
                #     annotator.seg_bbox(mask=mask_points,
                #                        label=f"Real {tracks_info[track_id]['real_speed']}, Pred. {tracks_info[track_id]['speed']} km/h",
                #                        mask_color=mask_color,
                #                        txt_color=text_color
                #                        )


                    # draw_detection(frame, x1, y1, x2, y2, lane_id, track_id)
                    # draw_instance(frame, rotated_rect)
                    draw_bottom(frame, bottom)

                    # overlay = frame.copy()
                    # # Usa un color específico para el relleno o uno basado en track_id
                    # # current_fill_color = colors(int(track_id % 10), True) # Ejemplo de color por track_id
                    # current_fill_color = (int(mask_color[0]), int(mask_color[1]),
                    #                       int(mask_color[2]))  # Usar el mask_color global
                    # cv2.fillPoly(overlay, [mask_points.astype(np.int32)], current_fill_color)
                    # opacity = 0.3  # Ajusta la opacidad
                    # cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)

                    # # --- Inicio: Dibujar cruz de orientación estilizada ---
                    # (center_x, center_y), (rect_width, rect_height), angle_degrees = rotated_rect
                    # center_x, center_y = float(center_x), float(center_y)
                    #
                    # angle_rad = np.deg2rad(angle_degrees)
                    # cos_a = np.cos(angle_rad)
                    # sin_a = np.sin(angle_rad)
                    #
                    # axis_scale_factor = 0.4
                    # arrow_length = (2*rect_width/3) * axis_scale_factor
                    # perp_line_segment_length = (rect_height / 2.0) * axis_scale_factor
                    #
                    # # --- Eje principal (con flecha estilizada) ---
                    # # Punto de inicio de la flecha (centro del objeto)
                    # arrow_start_x = int(center_x)
                    # arrow_start_y = int(center_y)
                    #
                    # # Punto final de la flecha (punta)
                    # arrow_tip_x = int(center_x + arrow_length * cos_a)
                    # arrow_tip_y = int(center_y + arrow_length * sin_a)
                    #
                    # # Dibujar el cuerpo principal de la flecha
                    # cv2.line(frame,
                    #          (arrow_start_x, arrow_start_y),
                    #          (arrow_tip_x, arrow_tip_y),
                    #          crosshair_color, thickness=4)
                    #
                    # # Parámetros para la cabeza de la flecha (ajusta estos valores para el estilo deseado)
                    # head_wing_length_ratio = 0.35  # Longitud de las alas de la punta como fracción de arrow_length
                    # head_wing_angle_deg = 30  # Ángulo de cada ala con respecto al eje de la flecha
                    #
                    # actual_head_wing_length = arrow_length * head_wing_length_ratio
                    # head_wing_angle_rad = np.deg2rad(head_wing_angle_deg)
                    #
                    # # Calcular y dibujar las alas de la cabeza de la flecha
                    # # Ala 1
                    # # El ángulo del ala es el ángulo de la flecha (angle_rad) + pi (para ir hacia atrás) - el ángulo del ala
                    # angle_w1 = angle_rad + np.pi - head_wing_angle_rad
                    # wing1_x = arrow_tip_x + actual_head_wing_length * np.cos(angle_w1)
                    # wing1_y = arrow_tip_y + actual_head_wing_length * np.sin(angle_w1)
                    # cv2.line(frame,
                    #          (arrow_tip_x, arrow_tip_y),  # Desde la punta de la flecha
                    #          (int(wing1_x), int(wing1_y)),
                    #          crosshair_color, thickness=4)
                    #
                    # # Ala 2
                    # # El ángulo del ala es el ángulo de la flecha (angle_rad) + pi (para ir hacia atrás) + el ángulo del ala
                    # angle_w2 = angle_rad + np.pi + head_wing_angle_rad
                    # wing2_x = arrow_tip_x + actual_head_wing_length * np.cos(angle_w2)
                    # wing2_y = arrow_tip_y + actual_head_wing_length * np.sin(angle_w2)
                    # cv2.line(frame,
                    #          (arrow_tip_x, arrow_tip_y),  # Desde la punta de la flecha
                    #          (int(wing2_x), int(wing2_y)),
                    #          crosshair_color, thickness=4)
                    # # --- Fin Eje principal ---
                    # # --- Fin: Dibujar cruz de orientación estilizada ---

            # if i == 167:
            #     cv2.imwrite("ims/brno_info_road.png", frame)
            #     break
            # if i == 7086:
            #     cv2.imwrite("ims/yolo_track.png", frame)
            #     break
            # if i == 208:
            #     # pass
            #     # cv2.imwrite('ims/yolo_bbox.png', frame)
            #     # cv2.imwrite('ims/yolo_mask.png', frame)
            #     # cv2.imwrite('ims/yolo_area.png', frame)
            #     # cv2.imwrite('ims/yolo_angle.png', frame)
            #     # cv2.imwrite('ims/yolo_rotated.png', frame)
            #     break
            # if i == 208:
            #     cv2.imwrite('ims/yolo_mask_cross.png', frame)
            #     break
            # if i == 2810:
            #     break
            if i == 2820:
                break

            if not self.show_frame(frame):
                break

            if frame_num % frame_skip_ratio == 0:
                self.write_frame(frame)  # write_frame redimensiona internamente

            frame_num += 1

        self.release()

        return tracks_info
