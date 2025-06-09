import os
from pathlib import Path
from ultralytics import YOLO

def download_models():
    """Download required YOLO models."""
    models_dir = Path("/app/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    # List of models to download
    models = [
        "yolov8n.pt",    # Nano model
        "yolov8s.pt",    # Small model
        "yolov8m.pt",    # Medium model
        "yolov8l.pt",    # Large model
        "yolov8x.pt",    # XLarge model
    ]

    print("Downloading YOLO models...")
    for model_name in models:
        model_path = models_dir / model_name
        if not model_path.exists():
            print(f"Downloading {model_name}...")
            try:
                model = YOLO(model_name)
                print(f"Successfully downloaded {model_name}")
            except Exception as e:
                print(f"Error downloading {model_name}: {e}")
        else:
            print(f"{model_name} already exists")

if __name__ == "__main__":
    download_models()
