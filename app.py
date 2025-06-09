import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time
import pandas as pd
from datetime import datetime
import os

# Configuración de la página
st.set_page_config(
    page_title="YOLO Real-Time Dashboard",
    page_icon="🎥",
    layout="wide"
)

# Título y descripción
st.title("YOLO Real-Time Dashboard")
st.markdown("""
    Este dashboard permite monitorear la detección de objetos en tiempo real usando YOLOv8.
    Características:
    - Detección en tiempo real
    - Múltiples cámaras
    - Estadísticas de detección
    - Grabación de video
    - Configuración de eventos
""")

# Sidebar para configuración
st.sidebar.header("Configuración")

# Selección de modelo
model_type = st.sidebar.selectbox(
    "Seleccionar modelo YOLO",
    ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]
)

# Umbral de confianza
confidence = st.sidebar.slider("Umbral de confianza", 0.0, 1.0, 0.5)

# Selección de cámara
camera_index = st.sidebar.number_input("Índice de cámara", 0, 10, 0)

# Opciones de grabación
record_video = st.sidebar.checkbox("Grabar video")
if record_video:
    st.sidebar.info("El video se guardará en la carpeta 'recordings'")

# Cargar modelo
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

model = load_model(model_type)

# Crear columnas para la visualización
col1, col2 = st.columns(2)

# Inicializar variables
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'start_time' not in st.session_state:
    st.session_state.start_time = time.time()
if 'detections' not in st.session_state:
    st.session_state.detections = []

# Función para procesar el frame
def process_frame(frame):
    results = model(frame, conf=confidence)
    annotated_frame = results[0].plot()

    # Actualizar estadísticas
    st.session_state.frame_count += 1
    for r in results:
        for box in r.boxes:
            st.session_state.detections.append({
                'class': model.names[int(box.cls[0])],
                'confidence': float(box.conf[0]),
                'timestamp': datetime.now()
            })

    return annotated_frame, results

# Configurar captura de video
cap = cv2.VideoCapture(camera_index)

# Configurar grabación si está activada
if record_video and not os.path.exists('recordings'):
    os.makedirs('recordings')

if record_video:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        f'recordings/recording_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4',
        fourcc, fps, (width, height)
    )

# Placeholder para el video
video_placeholder = col1.empty()
stats_placeholder = col2.empty()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Error al acceder a la cámara")
            break

        # Procesar frame
        processed_frame, results = process_frame(frame)

        # Convertir frame para Streamlit
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

        # Mostrar frame
        video_placeholder.image(processed_frame, channels="RGB", use_column_width=True)

        # Guardar frame si está activada la grabación
        if record_video:
            out.write(cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))

        # Mostrar estadísticas
        elapsed_time = time.time() - st.session_state.start_time
        fps = st.session_state.frame_count / elapsed_time if elapsed_time > 0 else 0

        # Crear DataFrame con las últimas detecciones
        if st.session_state.detections:
            df = pd.DataFrame(st.session_state.detections[-100:])  # Últimas 100 detecciones
            stats_placeholder.dataframe(df)

        # Mostrar FPS
        st.sidebar.metric("FPS", f"{fps:.1f}")

        # Limpiar detecciones antiguas
        if len(st.session_state.detections) > 1000:
            st.session_state.detections = st.session_state.detections[-1000:]

except Exception as e:
    st.error(f"Error: {str(e)}")

finally:
    cap.release()
    if record_video:
        out.release()
    cv2.destroyAllWindows()
