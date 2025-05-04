import cv2
import os
from tqdm import tqdm


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
        self.output_fps = 25.0  # Reducir FPS para compresión
        self.output_width = 1280 # Reducir resolución a 720p
        self.output_height = 720
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
        # Devolvemos las propiedades del video ORIGINAL, ya que se usan para
        # el procesamiento y dibujo antes del redimensionamiento final al guardar.
        return {
            "fps": self.input_fps,
            "width": self.input_width,
            "height": self.input_height,
            "total_frames": self.total_frames
        }
