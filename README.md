# 🚀 YOLO Computer Vision Apps

## 📋 Overview

**YOLO Computer Vision Apps** is an educational and experimentation platform that enables you to work with state-of-the-art computer vision models, including web and desktop interfaces, as well as tools for custom training and image annotation.

## 🎯 Project Goals

### Primary Goals

- **Democratize access** to advanced computer vision technologies
- **Simplify experimentation** with YOLO models without complex setup
- **Provide educational tools** for AI and Computer Vision courses
- **Facilitate rapid prototyping** of computer vision applications
- **Offer practical examples** of containerized deployment

### Secondary Goals

- Compare performance between different YOLO versions
- Demonstrate MLOps and containerization best practices
- Provide REST APIs for integration with other systems
- Create comprehensive educational documentation

## 🏗️ System Architecture

### General Architecture

```
┌─────────────────────────────────────────────────────┐
│                    LOAD BALANCER                    │
│                   (Nginx/Traefik)                   │
└─────────────────┬───────────────────────────────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
┌───▼───┐    ┌───▼───┐    ┌───▼───┐
│ App 1 │    │ App 2 │    │ App N │
│Gradio │    │Stream │    │FastAPI│
│:7860  │    │:8501  │    │:8000  │
└───┬───┘    └───┬───┘    └───┬───┘
    │            │            │
    └────────────┼────────────┘
                 │
    ┌────────────▼────────────┐
    │     SHARED STORAGE      │
    │  Models | Data | Logs   │
    └─────────────────────────┘
```

### Main Components

#### 1. Frontend Applications

- **Gradio Interface** (Port 7860): Interactive web interface with drag-and-drop
- **Streamlit Dashboard** (Port 8501): Analytical dashboard with multiple tabs
- **FastAPI Service** (Port 8000): REST API with automatic documentation
- **Jupyter Environment** (Port 8888): Notebooks for experimentation

#### 2. Backend Services

- **Model Management**: YOLO model loading and management
- **Image Processing**: Image processing pipeline
- **Results Storage**: Results and metrics storage
- **Logging System**: Centralized logging system

#### 3. Infrastructure

- **Docker Containers**: Each application in its own container
- **Shared Volumes**: Shared volumes for models and data
- **Network Bridge**: Internal network for service communication
- **Health Checks**: Service health monitoring

## 🔧 Technology Stack

### Core Technologies

- **Python 3.8+**: Main programming language
- **Ultralytics YOLO**: Model framework (YOLOv8, YOLO11)
- **PyTorch**: Deep learning framework
- **OpenCV**: Image processing
- **NumPy**: Numerical operations

### Web Frameworks

- **Gradio 4.0+**: Interactive ML interfaces
- **Streamlit 1.28+**: Dashboards and visualizations
- **FastAPI 0.104+**: Modern REST APIs
- **Uvicorn**: ASGI server for FastAPI

### Containerization

- **Docker**: Application containerization
- **Docker Compose**: Multi-container orchestration
- **Nginx**: Load balancer and reverse proxy (optional)

### ML Dependencies

```python
ultralytics>=8.0.0      # YOLO models
torch>=2.0.0           # Deep learning framework
torchvision>=0.15.0    # Computer vision utilities
opencv-python>=4.8.0   # Image processing
pillow>=10.0.0         # Image manipulation
numpy>=1.24.0          # Numerical operations
matplotlib>=3.7.0      # Plotting and visualization
seaborn>=0.12.0        # Statistical visualization
```

## 🚀 Quick Start

### Option 1: Complete Deployment (Recommended for Courses)

```bash
# 1. Clone repository
git clone https://github.com/your-username/yolo-vision-apps.git
cd yolo-vision-apps

# 2. Configure environment variables
cp .env.example .env
# Edit .env with specific configurations

# 3. Build and launch all services
docker-compose up --build -d

# 4. Verify services
./scripts/docker/health_check.sh

# 5. Access applications
echo "Gradio: http://localhost:7860"
echo "Streamlit: http://localhost:8501"
echo "FastAPI: http://localhost:8000/docs"
echo "Jupyter: http://localhost:8888"
```

### Option 2: Selective Deployment

```bash
# Only Gradio and FastAPI
docker-compose up gradio-app fastapi-app -d

# Development only
docker-compose -f docker-compose.dev.yml up -d

# Production only
docker-compose -f docker-compose.prod.yml up -d
```

## 📚 Educational Resources

### Interactive Notebooks

1. **YOLO_Introduction.ipynb**: Basic concepts and first steps
2. **Model_Comparison.ipynb**: YOLOv8 vs YOLO11 comparison
3. **Real_Time_Detection.ipynb**: Real-time detection with webcam
4. **Custom_Training.ipynb**: Training with custom datasets
5. **Performance_Optimization.ipynb**: Optimization and acceleration

### Example Datasets

- **COCO Sample**: 100 images from COCO dataset
- **Custom Objects**: Dataset for specific object detection
- **Security Demo**: Images for security demos
- **Traffic Analysis**: Videos for traffic analysis

### Practical Exercises

- Face mask detection
- People counting in spaces
- Safety equipment analysis
- Product classification

## 🔧 Troubleshooting

### Common Issues

#### 1. Insufficient Memory

```bash
# Solution: Increase Docker memory
# In Docker Desktop: Settings > Resources > Memory: 4GB+

# Or use smaller model
docker-compose exec gradio-app python -c "
from ultralytics import YOLO
model = YOLO('yolo11n.pt')  # Nano model
"
```

#### 2. Port Already in Use

```bash
# Check ports in use
netstat -tulpn | grep :7860

# Change port in docker-compose.yml
ports:
  - "7861:7860"  # Different host port
```

#### 3. Model Download Issues

```bash
# Manual download
docker-compose exec gradio-app python -c "
from ultralytics import YOLO
YOLO('yolo11n.pt')  # Force download
"
```

## 📊 Monitoring and Observability

### Key Metrics

- **Inference Latency**: Response time per prediction
- **Throughput**: Predictions per second
- **Memory Usage**: RAM consumption per container
- **CPU/GPU Usage**: Resource utilization
- **Application Errors**: HTTP error rates

## 🔒 Security and Best Practices

### Security Configuration

```yaml
# docker-compose.security.yml
version: "3.8"
services:
  gradio-app:
    security_opt:
      - no-new-privileges:true
    user: "1000:1000"
    read_only: true
    tmpfs:
      - /tmp:rw,noexec,nosuid,size=512m
    cap_drop:
      - ALL
    cap_add:
      - CHOWN
      - SETGID
      - SETUID
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## 📫 Support

For support, please open an issue in the GitHub repository or contact the maintainers.

## 🙏 Acknowledgments

- Ultralytics for the YOLO framework
- The open-source community for their contributions
- All the educators and researchers who inspired this project

# YOLO Real-Time Dashboard

Este proyecto proporciona una interfaz web para la detección de objetos en tiempo real usando YOLOv8, con características avanzadas de monitoreo y análisis.

## Características

- 🎥 Detección de objetos en tiempo real
- 📊 Estadísticas de detección en vivo
- 📹 Grabación de video
- 🔄 Soporte para múltiples cámaras
- ⚙️ Configuración flexible
- 📈 Monitoreo de FPS
- 📝 Registro de eventos

## Requisitos

- Python 3.12 o superior
- Cámara web o dispositivo de captura
- Dependencias listadas en `requirements.txt`

## Instalación

1. Clonar el repositorio:

```bash
git clone <url-del-repositorio>
cd yolo
```

2. Instalar dependencias:

```bash
py -3.12 -m pip install -r requirements.txt
```

## Uso

1. Iniciar la aplicación:

```bash
streamlit run app.py
```

2. Abrir el navegador en la URL mostrada (generalmente http://localhost:8501)

3. Configurar en el panel lateral:
   - Seleccionar modelo YOLO
   - Ajustar umbral de confianza
   - Seleccionar cámara
   - Activar/desactivar grabación

## Controles

- **Panel lateral**: Configuración de la aplicación
- **Vista principal**:
  - Izquierda: Video en tiempo real con detecciones
  - Derecha: Estadísticas de detección

## Características avanzadas

### Grabación de video

- Los videos se guardan en la carpeta `recordings`
- Formato: MP4
- Nombre: `recording_YYYYMMDD_HHMMSS.mp4`

### Estadísticas

- FPS en tiempo real
- Historial de detecciones
- Clase y confianza de cada detección

### Múltiples cámaras

- Soporte para hasta 10 cámaras
- Cambio dinámico entre cámaras

## Solución de problemas

1. **Error de cámara**:

   - Verificar que la cámara esté conectada
   - Comprobar el índice de cámara en la configuración

2. **Bajo rendimiento**:

   - Usar un modelo más ligero (yolov8n.pt)
   - Reducir la resolución de la cámara
   - Aumentar el umbral de confianza

3. **Error de memoria**:
   - Reducir el número de detecciones guardadas
   - Limpiar la caché del navegador

## Contribuir

Las contribuciones son bienvenidas. Por favor, sigue estos pasos:

1. Fork el repositorio
2. Crea una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abre un Pull Request

## Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 🖥️ Aplicaciones incluidas

### 1. App de escritorio (PyQt6)

- Archivo: `yolo_desktop.py`
- Permite seleccionar la cámara, definir reglas de eventos personalizados (notificaciones, comandos, emails) y monitorizar detecciones en tiempo real.
- **Uso:**
  ```bash
  py -3.12 yolo_desktop.py
  ```

### 2. Dashboard web en tiempo real (Streamlit)

- Archivo: `app.py`
- Interfaz web moderna para detección en tiempo real, estadísticas, grabación y selección de cámara.
- **Uso:**
  ```bash
  streamlit run app.py
  ```

### 3. Herramienta de anotación LabelImg

- Instalación incluida vía pip.
- Permite anotar imágenes localmente en formato YOLO.
- **Uso:**
  ```bash
  labelImg
  ```

### 4. Entrenamiento personalizado con Roboflow y YOLOv8

- Sube tus imágenes a [Roboflow](https://roboflow.com/), anótalas y exporta en formato YOLOv8.
- Descarga el dataset y entrena tu modelo con:
  ```bash
  yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640
  ```
- Cambia el modelo en las apps para usar tu modelo entrenado (`best.pt`).

### 5. Etiquetado masivo de imágenes desde ZIP (Bulk Labeler)

- Archivo: `bulk_labeler.py`
- Permite seleccionar un archivo ZIP con imágenes, elegir la clase/etiqueta y el destino (train/val/test).
- Extrae las imágenes, genera las anotaciones YOLO (caja cubriendo toda la imagen) y las guarda en la estructura correcta (`data/images/SPLIT` y `data/labels/SPLIT`).
- Actualiza o crea automáticamente el archivo `data/data.yaml`.
- **Uso:**
  ```bash
  py -3.12 bulk_labeler.py
  ```
- Ideal para datasets homogéneos donde todas las imágenes son de la misma clase.

## 📦 Estructura del proyecto

- `yolo_desktop.py` — App de escritorio PyQt6 para reglas y eventos
- `app.py` — Dashboard web Streamlit
- `requirements.txt` — Dependencias
- `README.md` — Documentación
- `data/`, `models/`, `recordings/` — Datos, modelos y grabaciones

## 📝 Notas rápidas

- Puedes usar LabelImg o Roboflow para anotar imágenes.
- El entrenamiento personalizado se realiza con Ultralytics YOLOv8.
- El proyecto soporta tanto uso educativo como prototipado profesional.

## 🔗 Recursos útiles

- [Roboflow](https://roboflow.com/) — Anotación y gestión de datasets
- [LabelImg](https://github.com/tzutalin/labelImg) — Anotación local
- [Ultralytics YOLO Docs](https://docs.ultralytics.com/) — Documentación oficial

## 🏁 ¡Comienza a experimentar!

1. Prueba la app de escritorio para automatizaciones y eventos.
2. Usa el dashboard web para análisis y monitoreo.
3. Anota y entrena tus propios modelos fácilmente.
