import os
import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLO model
model_path = os.getenv("MODEL_PATH", "yolov8n.pt")
model = YOLO(model_path)

def process_image(image):
    """Process image with YOLO model."""
    if image is None:
        return None, "No image provided"

    # Run inference
    results = model(image)

    # Get the first result
    result = results[0]

    # Draw boxes on the image
    annotated_img = result.plot()

    # Get detection information
    detections = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = result.names[class_id]
        detections.append(f"{class_name}: {confidence:.2f}")

    return annotated_img, "\n".join(detections)

# Create Gradio interface
iface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="numpy"),
    outputs=[
        gr.Image(label="Detected Objects"),
        gr.Textbox(label="Detection Results")
    ],
    title="YOLO Object Detection",
    description="Upload an image to detect objects using YOLO"
)

# Launch the app
if __name__ == "__main__":
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    iface.launch(server_name=server_name, server_port=7860, share=True)
