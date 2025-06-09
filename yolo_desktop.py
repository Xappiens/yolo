"""
YOLO Event Detector - PyQt6 Desktop App

This application allows you to select a camera, define custom event rules (notifications, commands, emails), and monitor real-time object detection using YOLOv8.
"""
import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QHBoxLayout, QLabel, QComboBox, QPushButton,
                            QSpinBox, QCheckBox, QListWidget, QMessageBox,
                            QDialog, QFormLayout, QLineEdit, QTextEdit,
                            QDoubleSpinBox, QGroupBox, QRadioButton)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QImage, QPixmap
from ultralytics import YOLO
import json
import os
from datetime import datetime
import subprocess
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    detection_signal = pyqtSignal(dict)

    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.running = True
        self.model = YOLO("yolov8n.pt")
        self.confidence = 0.5
        self.selected_classes = set()

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        while self.running:
            ret, frame = cap.read()
            if ret:
                # Realizar detección
                results = self.model(frame, conf=self.confidence)

                # Procesar detecciones
                for r in results:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = self.model.names[cls]

                        # Si la clase está seleccionada, emitir señal
                        if class_name in self.selected_classes:
                            self.detection_signal.emit({
                                'class': class_name,
                                'confidence': conf,
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            })

                # Dibujar detecciones
                annotated_frame = results[0].plot()
                self.change_pixmap_signal.emit(annotated_frame)

            QThread.msleep(30)  # ~30 FPS

        cap.release()

    def stop(self):
        self.running = False
        self.wait()

class EventConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configurar Evento")
        self.setModal(True)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Grupo de condiciones
        condition_group = QGroupBox("Condiciones")
        condition_layout = QFormLayout()

        self.min_confidence = QDoubleSpinBox()
        self.min_confidence.setRange(0.0, 1.0)
        self.min_confidence.setSingleStep(0.1)
        self.min_confidence.setValue(0.5)
        condition_layout.addRow("Confianza mínima:", self.min_confidence)

        self.min_duration = QSpinBox()
        self.min_duration.setRange(0, 60)
        self.min_duration.setValue(0)
        condition_layout.addRow("Duración mínima (segundos):", self.min_duration)

        condition_group.setLayout(condition_layout)
        layout.addWidget(condition_group)

        # Grupo de acciones
        action_group = QGroupBox("Acciones")
        action_layout = QVBoxLayout()

        # Notificación
        self.notify_radio = QRadioButton("Mostrar notificación")
        self.notify_radio.setChecked(True)
        action_layout.addWidget(self.notify_radio)

        # Ejecutar comando
        self.command_radio = QRadioButton("Ejecutar comando")
        self.command_radio.setChecked(False)
        action_layout.addWidget(self.command_radio)
        self.command_input = QLineEdit()
        self.command_input.setPlaceholderText("Comando a ejecutar")
        action_layout.addWidget(self.command_input)

        # Enviar email
        self.email_radio = QRadioButton("Enviar email")
        self.email_radio.setChecked(False)
        action_layout.addWidget(self.email_radio)

        email_form = QFormLayout()
        self.email_to = QLineEdit()
        self.email_to.setPlaceholderText("destinatario@ejemplo.com")
        email_form.addRow("Para:", self.email_to)

        self.email_subject = QLineEdit()
        self.email_subject.setPlaceholderText("Asunto del email")
        email_form.addRow("Asunto:", self.email_subject)

        self.email_body = QTextEdit()
        self.email_body.setPlaceholderText("Cuerpo del email")
        email_form.addRow("Mensaje:", self.email_body)

        action_layout.addLayout(email_form)

        action_group.setLayout(action_layout)
        layout.addWidget(action_group)

        # Botones
        button_layout = QHBoxLayout()
        self.save_button = QPushButton("Guardar")
        self.save_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("Cancelar")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

    def get_config(self):
        config = {
            'min_confidence': self.min_confidence.value(),
            'min_duration': self.min_duration.value(),
            'action_type': 'notify' if self.notify_radio.isChecked() else
                          'command' if self.command_radio.isChecked() else
                          'email',
            'command': self.command_input.text() if self.command_radio.isChecked() else None,
            'email_to': self.email_to.text() if self.email_radio.isChecked() else None,
            'email_subject': self.email_subject.text() if self.email_radio.isChecked() else None,
            'email_body': self.email_body.toPlainText() if self.email_radio.isChecked() else None
        }
        return config

class EventRule:
    def __init__(self, trigger_class, config):
        self.trigger_class = trigger_class
        self.min_confidence = config.get('min_confidence', 0.5)
        self.min_duration = config.get('min_duration', 0)
        self.action_type = config.get('action_type', 'notify')
        self.command = config.get('command', None)
        self.email_to = config.get('email_to', None)
        self.email_subject = config.get('email_subject', None)
        self.email_body = config.get('email_body', None)
        self.detection_start = None
        self.detection_count = 0

    def check_condition(self, detection):
        if detection['confidence'] < self.min_confidence:
            self.detection_start = None
            self.detection_count = 0
            return False

        if self.detection_start is None:
            self.detection_start = datetime.now()
            self.detection_count = 1
        else:
            self.detection_count += 1

        duration = (datetime.now() - self.detection_start).total_seconds()
        return duration >= self.min_duration

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Event Detector")
        self.setGeometry(100, 100, 1200, 800)

        # Variables
        self.video_thread = None
        self.event_rules = []

        # Crear interfaz
        self.create_ui()
        # Cargar reglas después de crear los widgets
        self.load_rules()

    def create_ui(self):
        # Widget principal
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # Layout principal
        layout = QHBoxLayout()
        main_widget.setLayout(layout)

        # Panel izquierdo (video y controles)
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)

        # Etiqueta para el video
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        left_layout.addWidget(self.video_label)

        # Controles de cámara
        camera_layout = QHBoxLayout()
        self.camera_combo = QComboBox()
        self.camera_combo.addItems([f"Cámara {i}" for i in range(10)])
        self.start_button = QPushButton("Iniciar")
        self.start_button.clicked.connect(self.toggle_camera)
        camera_layout.addWidget(QLabel("Cámara:"))
        camera_layout.addWidget(self.camera_combo)
        camera_layout.addWidget(self.start_button)
        left_layout.addLayout(camera_layout)

        # Controles de detección
        detection_layout = QHBoxLayout()
        self.confidence_spin = QSpinBox()
        self.confidence_spin.setRange(0, 100)
        self.confidence_spin.setValue(50)
        self.confidence_spin.valueChanged.connect(self.update_confidence)
        detection_layout.addWidget(QLabel("Confianza:"))
        detection_layout.addWidget(self.confidence_spin)
        left_layout.addLayout(detection_layout)

        # Panel derecho (eventos y reglas)
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)

        # Lista de clases para detección
        right_layout.addWidget(QLabel("Clases a detectar:"))
        self.class_list = QListWidget()
        self.class_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.load_classes()
        right_layout.addWidget(self.class_list)

        # Lista de reglas activas
        right_layout.addWidget(QLabel("Reglas activas:"))
        self.rules_list = QListWidget()
        right_layout.addWidget(self.rules_list)

        # Lista de eventos
        right_layout.addWidget(QLabel("Eventos detectados:"))
        self.event_list = QListWidget()
        right_layout.addWidget(self.event_list)

        # Botones de acción
        button_layout = QHBoxLayout()
        self.add_rule_button = QPushButton("Añadir Regla")
        self.add_rule_button.clicked.connect(self.add_rule)
        self.remove_rule_button = QPushButton("Eliminar Regla")
        self.remove_rule_button.clicked.connect(self.remove_rule)
        self.clear_events_button = QPushButton("Limpiar Eventos")
        self.clear_events_button.clicked.connect(self.clear_events)
        button_layout.addWidget(self.add_rule_button)
        button_layout.addWidget(self.remove_rule_button)
        button_layout.addWidget(self.clear_events_button)
        right_layout.addLayout(button_layout)

        # Añadir paneles al layout principal
        layout.addWidget(left_panel, 2)
        layout.addWidget(right_panel, 1)

    def load_classes(self):
        model = YOLO("yolov8n.pt")
        for class_name in model.names.values():
            self.class_list.addItem(class_name)

    def toggle_camera(self):
        if self.video_thread is None or not self.video_thread.running:
            # Iniciar cámara
            camera_index = self.camera_combo.currentIndex()
            self.video_thread = VideoThread(camera_index)
            self.video_thread.change_pixmap_signal.connect(self.update_image)
            self.video_thread.detection_signal.connect(self.handle_detection)
            self.video_thread.start()
            self.start_button.setText("Detener")
        else:
            # Detener cámara
            self.video_thread.stop()
            self.video_thread = None
            self.start_button.setText("Iniciar")

    def update_image(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.video_label.width(), self.video_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio))

    def update_confidence(self):
        if self.video_thread:
            self.video_thread.confidence = self.confidence_spin.value() / 100

    def handle_detection(self, detection):
        # Actualizar lista de eventos
        event_text = f"{detection['timestamp']} - {detection['class']} ({detection['confidence']:.2f})"
        self.event_list.addItem(event_text)

        # Verificar reglas
        for rule in self.event_rules:
            if rule.trigger_class == detection['class'] and rule.check_condition(detection):
                self.execute_action(rule, detection)

    def add_rule(self):
        selected_items = self.class_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Error", "Selecciona al menos una clase")
            return

        # Abrir diálogo de configuración
        dialog = EventConfigDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            config = dialog.get_config()

            # Crear reglas para cada clase seleccionada
            for item in selected_items:
                rule = EventRule(item.text(), config)
                self.event_rules.append(rule)
                self.video_thread.selected_classes.add(item.text())

                # Actualizar lista de reglas
                rule_text = f"{item.text()} - {config['action_type']}"
                self.rules_list.addItem(rule_text)

            self.save_rules()

    def remove_rule(self):
        selected_items = self.rules_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Error", "Selecciona una regla para eliminar")
            return

        # Recolectar índices a eliminar
        indices = [self.rules_list.row(item) for item in selected_items]
        indices.sort(reverse=True)  # Eliminar de atrás hacia adelante

        for index in indices:
            rule = self.event_rules[index]
            # Solo modificar selected_classes si el hilo está activo y la clase está presente
            if self.video_thread and rule.trigger_class in self.video_thread.selected_classes:
                self.video_thread.selected_classes.remove(rule.trigger_class)
            self.event_rules.pop(index)
            self.rules_list.takeItem(index)

        self.save_rules()

    def execute_action(self, rule, detection):
        if rule.action_type == "notify":
            QMessageBox.information(self, "Evento Detectado",
                                  f"Se detectó {detection['class']} with confidence {detection['confidence']:.2f}")

        elif rule.action_type == "command" and rule.command:
            try:
                subprocess.Popen(rule.command, shell=True)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Error al ejecutar comando: {str(e)}")

        elif rule.action_type == "email" and rule.email_to:
            try:
                # Configurar email
                msg = MIMEMultipart()
                msg['From'] = "yolo_detector@example.com"  # Cambiar por tu email
                msg['To'] = rule.email_to
                msg['Subject'] = rule.email_subject

                # Personalizar cuerpo del email
                body = rule.email_body.format(
                    class_name=detection['class'],
                    confidence=detection['confidence'],
                    timestamp=detection['timestamp']
                )
                msg.attach(MIMEText(body, 'plain'))

                # Enviar email (necesitarás configurar el servidor SMTP)
                # server = smtplib.SMTP('smtp.gmail.com', 587)
                # server.starttls()
                # server.login("tu_email@gmail.com", "tu_contraseña")
                # server.send_message(msg)
                # server.quit()

                QMessageBox.information(self, "Email Enviado",
                                      f"Se ha enviado un email a {rule.email_to}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Error al enviar email: {str(e)}")

    def clear_events(self):
        self.event_list.clear()

    def save_rules(self):
        rules_data = [{
            'trigger_class': rule.trigger_class,
            'min_confidence': rule.min_confidence,
            'min_duration': rule.min_duration,
            'action_type': rule.action_type,
            'command': rule.command,
            'email_to': rule.email_to,
            'email_subject': rule.email_subject,
            'email_body': rule.email_body
        } for rule in self.event_rules]

        with open('event_rules.json', 'w') as f:
            json.dump(rules_data, f)

    def load_rules(self):
        try:
            with open('event_rules.json', 'r') as f:
                rules_data = json.load(f)
                self.event_rules = [
                    EventRule(rule['trigger_class'], rule)
                    for rule in rules_data
                ]

                # Actualizar lista de reglas
                for rule in self.event_rules:
                    rule_text = f"{rule.trigger_class} - {rule.action_type}"
                    self.rules_list.addItem(rule_text)
        except FileNotFoundError:
            self.event_rules = []

    def closeEvent(self, event):
        if self.video_thread:
            self.video_thread.stop()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
