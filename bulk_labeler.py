"""
YOLO Bulk Labeler - PyQt6 Desktop App

This tool allows you to select a ZIP file of images, specify a class label, and a split (train/val/test).
It extracts the images, generates YOLO annotation files (bounding box covering the whole image), and saves them in the correct structure for YOLO training.
"""
import sys
import os
import zipfile
import shutil
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton,
    QFileDialog, QLineEdit, QComboBox, QMessageBox
)
from PyQt6.QtCore import Qt
from PIL import Image

class BulkLabeler(QMainWindow):
    """
    Main window for the YOLO Bulk Labeler desktop app.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Bulk Labeler")
        self.setGeometry(200, 200, 400, 250)
        self.init_ui()

    def init_ui(self):
        """Initialize the UI components."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.zip_label = QLabel("Select a ZIP file with images:")
        layout.addWidget(self.zip_label)
        self.zip_button = QPushButton("Choose ZIP")
        self.zip_button.clicked.connect(self.select_zip)
        layout.addWidget(self.zip_button)
        self.zip_path = QLineEdit()
        self.zip_path.setReadOnly(True)
        layout.addWidget(self.zip_path)

        self.class_label = QLabel("Class label for all images:")
        layout.addWidget(self.class_label)
        self.class_input = QLineEdit()
        layout.addWidget(self.class_input)

        self.split_label = QLabel("Destination split (train/val/test):")
        layout.addWidget(self.split_label)
        self.split_combo = QComboBox()
        self.split_combo.addItems(["train", "val", "test"])
        layout.addWidget(self.split_combo)

        self.start_button = QPushButton("Process and label")
        self.start_button.clicked.connect(self.process_zip)
        layout.addWidget(self.start_button)

    def select_zip(self):
        """Open a file dialog to select a ZIP file."""
        file, _ = QFileDialog.getOpenFileName(self, "Select ZIP", "", "ZIP Files (*.zip)")
        if file:
            self.zip_path.setText(file)

    def process_zip(self):
        """Extract images, generate YOLO labels, and save in the correct structure."""
        zip_file = self.zip_path.text()
        class_name = self.class_input.text().strip()
        split = self.split_combo.currentText()
        if not zip_file or not os.path.isfile(zip_file):
            QMessageBox.warning(self, "Error", "Please select a valid ZIP file.")
            return
        if not class_name:
            QMessageBox.warning(self, "Error", "Please enter a class label.")
            return

        # Create destination folders
        img_dir = os.path.join("data", "images", split)
        label_dir = os.path.join("data", "labels", split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)

        # Extract zip to a temporary folder
        temp_dir = "_bulk_labeler_temp"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # Find image files
        image_files = []
        for root, _, files in os.walk(temp_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                    image_files.append(os.path.join(root, f))

        # Assign class id (if data.yaml exists, use the correct index)
        class_id = 0
        yaml_path = os.path.join("data", "data.yaml")
        if os.path.exists(yaml_path):
            import yaml
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            if 'names' in data and class_name in data['names']:
                class_id = data['names'].index(class_name)
            else:
                # Add the class to yaml
                data['names'].append(class_name)
                data['nc'] = len(data['names'])
                with open(yaml_path, 'w') as f:
                    yaml.dump(data, f)
                class_id = len(data['names']) - 1
        else:
            # Create a new data.yaml
            with open(yaml_path, 'w') as f:
                f.write(f"train: ./images/train\nval: ./images/val\nnc: 1\nnames: ['{class_name}']\n")
            class_id = 0

        # Process images and create YOLO labels
        for img_path in image_files:
            img_name = os.path.basename(img_path)
            dest_img = os.path.join(img_dir, img_name)
            shutil.copy(img_path, dest_img)
            # Get image size
            with Image.open(img_path) as im:
                w, h = im.size
            # Box covering the whole image
            x_center, y_center, width, height = 0.5, 0.5, 1.0, 1.0
            label_txt = f"{class_id} {x_center} {y_center} {width} {height}\n"
            label_name = os.path.splitext(img_name)[0] + ".txt"
            dest_label = os.path.join(label_dir, label_name)
            with open(dest_label, 'w') as f:
                f.write(label_txt)

        shutil.rmtree(temp_dir)
        QMessageBox.information(self, "Done!", f"{len(image_files)} images processed and labeled in {img_dir} and {label_dir}.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = BulkLabeler()
    window.show()
    sys.exit(app.exec())
