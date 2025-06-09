import cv2
import time
import argparse
from ultralytics import YOLO
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Real-time Object Detection')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Model path')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--save', action='store_true', help='Save video')
    return parser.parse_args()

def main():
    args = parse_args()

    # Cargar modelo
    model = YOLO(args.model)

    # Configurar cámara
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir la cámara {args.camera}")
        return

    # Configurar grabación de video si se solicita
    if args.save:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

    # Variables para FPS
    prev_time = 0
    curr_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Calcular FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time

        # Infiere con YOLO
        results = model(frame, conf=args.conf)

        # Dibuja las detecciones en el frame
        annotated_frame = results[0].plot()

        # Añadir FPS al frame
        cv2.putText(annotated_frame, f'FPS: {fps:.1f}', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Guardar frame si se solicita
        if args.save:
            out.write(annotated_frame)

        # Mostrar el frame
        cv2.imshow("YOLO Real-Time", annotated_frame)

        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar recursos
    cap.release()
    if args.save:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
