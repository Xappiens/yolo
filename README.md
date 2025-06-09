# 🚀 YOLO Computer Vision Apps

**Developed and maintained by [Xappiens.com](https://xappiens.com)**

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
- **Developed and maintained by [Xappiens.com](https://xappiens.com)**

# 🖥️ Included Applications

### 1. Desktop App (PyQt6)

- File: `yolo_desktop.py`
- Allows you to select the camera, define custom event rules (notifications, commands, emails), and monitor detections in real time.
- **Usage:**
  ```bash
  py -3.12 yolo_desktop.py
  ```

### 2. Real-Time Web Dashboard (Streamlit)

- File: `app.py`
- Modern web interface for real-time detection, statistics, recording, and camera selection.
- **Usage:**
  ```bash
  streamlit run app.py
  ```

### 3. Image Annotation Tool: LabelImg

- Installed via pip.
- Allows you to annotate images locally in YOLO format.
- **Usage:**
  ```bash
  labelImg
  ```

### 4. Custom Training with Roboflow and YOLOv8

- Upload your images to [Roboflow](https://roboflow.com/), annotate them, and export in YOLOv8 format.
- Download the dataset and train your model with:
  ```bash
  yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640
  ```
- Change the model in the apps to use your trained model (`best.pt`).

### 5. Bulk Image Labeling from ZIP (Bulk Labeler)

- File: `bulk_labeler.py`
- Lets you select a ZIP file with images, choose the class/label and the destination (train/val/test).
- Extracts images, generates YOLO annotations (box covering the whole image), and saves them in the correct structure (`data/images/SPLIT` and `data/labels/SPLIT`).
- Automatically updates or creates the `data/data.yaml` file.
- **Usage:**
  ```bash
  py -3.12 bulk_labeler.py
  ```
- Ideal for homogeneous datasets where all images belong to the same class.

## 📦 Project Structure

- `yolo_desktop.py` — PyQt6 desktop app for rules and events
- `app.py` — Streamlit web dashboard
- `bulk_labeler.py` — Bulk labeling desktop app
- `requirements.txt` — Dependencies
- `README.md` — Documentation
- `data/`, `models/`, `recordings/` — Data, models, and recordings

## 📝 Quick Notes

- You can use LabelImg or Roboflow to annotate images.
- Custom training is done with Ultralytics YOLOv8.
- The project supports both educational and professional prototyping use cases.

## 🔗 Useful Resources

- [Roboflow](https://roboflow.com/) — Annotation and dataset management
- [LabelImg](https://github.com/tzutalin/labelImg) — Local annotation
- [Ultralytics YOLO Docs](https://docs.ultralytics.com/) — Official documentation

## 🏁 Get Started!

1. Try the desktop app for automations and event rules.
2. Use the web dashboard for analysis and monitoring.
3. Annotate and train your own models easily.
