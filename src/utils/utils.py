import torch
import numpy as np
import cv2


# === GPU automática ===
def get_best_gpu():
    import subprocess
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'])
        mem_free = [int(x) for x in output.decode('utf-8').strip().split('\n')]
        best_gpu = int(torch.tensor(mem_free).argmax())
        print(f"🚀 Usando GPU {best_gpu} con {mem_free[best_gpu]} MB libres")
        return best_gpu
    except:
        print("⚠️ No se pudo detectar GPU. Usando CPU.")
        return None

def get_device():
    gpu_id = get_best_gpu()
    return torch.device(f'cuda:{gpu_id}' if gpu_id is not None else 'cpu')


def is_inside_mask(xc, yc, mask):
    h, w = mask.shape
    if 0 <= int(xc) < w and 0 <= int(yc) < h:
        # Asegurarse que los índices son enteros
        return mask[int(yc), int(xc)] == 255
    return False

def is_between_lines(p, l1, l2):
    return np.dot(p,l1)*np.dot(p,l2)*np.dot(l1[0:2],l2[0:2]) <= 0

def get_lane_index(point, lines):
    for i in range(len(lines)-1):
        if is_between_lines(point, lines[i], lines[i+1]):
            return i
    return None

def draw_lines(frame, lines, width, color=(0, 255, 0) ):
    for line in lines:
        a, b, c = line
        if abs(b) > 1e-6:
            x_vals = np.array([0, width])
            y_vals = -(a * x_vals + c) / b
            pts = np.int32(np.column_stack((x_vals, y_vals)))
            cv2.line(frame, tuple(pts[0]), tuple(pts[1]), color, 2)

def draw_detection(frame, x1, y1, x2, y2, lane_id, track_id):
    # Convertir coordenadas a enteros
    pt1 = (int(x1), int(y1))
    pt2 = (int(x2), int(y2))

    # Dibujar el rectángulo con coordenadas enteras
    cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)

    # Asegurarse que la posición del texto también use coordenadas enteras si es necesario
    text_pos = (int(x1), int(y1) - 10)
    cv2.putText(frame, f"ID {track_id} - L{lane_id}", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def point_to_line_distance(p, l):
    return abs(np.dot(l, p / p[2])) / np.linalg.norm(l[0:2])


