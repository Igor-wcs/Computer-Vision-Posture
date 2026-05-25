import cv2
import mediapipe as mp
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk
import threading
import time
import csv
import psutil
import os

# Configurações iniciais do CustomTkinter
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class PostureDetectorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Detector de Postura - Home Office")
        self.geometry("900x750")

        # Inicializa MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Variáveis de Calibração
        self.calibrated = False
        self.base_head_dist = 0.15 # Valor default
        self.base_shoulder_width = 0.2
        self.base_z_depth = 0.0

        # Configuração da UI
        self.setup_ui()

        # Variáveis de controle
        self.cap = cv2.VideoCapture(0)
        self.running = True
        
        # Logging para IC
        self.log_file = "log_mediapipe.csv"
        self.init_log_file()
        self.frame_count = 0
        self.start_time = time.time()
        
        # Thread para processamento de vídeo
        self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
        self.video_thread.start()

    def init_log_file(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'FPS', 'CPU_Usage_Percent', 'RAM_Usage_MB', 'Posture_Status'])

    def log_data(self, status):
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed > 1.0:
            fps = self.frame_count / elapsed
            cpu = psutil.cpu_percent()
            ram = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
            
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), round(fps, 2), cpu, round(ram, 2), status])
            
            self.frame_count = 0
            self.start_time = time.time()

    def setup_ui(self):
        # Grid layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Header
        self.header_label = ctk.CTkLabel(self, text="Monitoramento de Postura em Tempo Real", font=ctk.CTkFont(size=24, weight="bold"))
        self.header_label.grid(row=0, column=0, padx=20, pady=10)

        # Video Frame
        self.video_label = ctk.CTkLabel(self, text="")
        self.video_label.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")

        # Control Panel
        self.control_panel = ctk.CTkFrame(self)
        self.control_panel.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        self.control_panel.grid_columnconfigure((0, 1, 2), weight=1)

        self.status_val_label = ctk.CTkLabel(self.control_panel, text="Aguardando Calibração...", font=ctk.CTkFont(size=20, weight="bold"), text_color="yellow")
        self.status_val_label.grid(row=0, column=0, columnspan=2, padx=20, pady=10)

        self.calibrate_button = ctk.CTkButton(self.control_panel, text="Calibrar Postura Ideal", command=self.start_calibration, fg_color="green", hover_color="darkgreen")
        self.calibrate_button.grid(row=0, column=2, padx=20, pady=10)

        # Info Labels
        self.info_label = ctk.CTkLabel(self, text="Sente-se ereto e clique em 'Calibrar' para começar.", font=ctk.CTkFont(size=12))
        self.info_label.grid(row=3, column=0, pady=5)

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def start_calibration(self):
        self.calibrating = True
        self.info_label.configure(text="Calibrando... Mantenha a postura por 2 segundos.")
        # A calibração real acontece no process_posture quando calibrating é True
        threading.Timer(2.0, self.finish_calibration).start()

    def finish_calibration(self):
        self.calibrated = True
        self.info_label.configure(text="Calibração Concluída!")
        self.calibrate_button.configure(text="Recalibrar")

    def video_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = self.process_posture(frame)

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = img.resize((800, 450), Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(image=img)

            if self.running:
                self.video_label.configure(image=img_tk)
                self.video_label._image = img_tk

        self.cap.release()

    def process_posture(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        posture_status = "Aguardando Calibração"
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Landmarks 3D (X, Y, Z)
            nose = (landmarks[self.mp_pose.PoseLandmark.NOSE.value].x, 
                    landmarks[self.mp_pose.PoseLandmark.NOSE.value].y,
                    landmarks[self.mp_pose.PoseLandmark.NOSE.value].z)
            
            l_shoulder = (landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                          landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                          landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].z)
            
            r_shoulder = (landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, 
                          landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                          landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z)

            mid_shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2
            current_head_dist = abs(nose[1] - mid_shoulder_y)
            current_shoulder_width = abs(l_shoulder[0] - r_shoulder[0])
            current_z = (l_shoulder[2] + r_shoulder[2]) / 2

            # Fase de Calibração
            if hasattr(self, 'calibrating') and self.calibrating:
                self.base_head_dist = current_head_dist
                self.base_shoulder_width = current_shoulder_width
                self.base_z_depth = current_z
                self.calibrating = False

            if self.calibrated:
                posture_status = self.classify_enhanced_posture(
                    nose, l_shoulder, r_shoulder, 
                    current_head_dist, current_shoulder_width, current_z
                )

            # Desenha landmarks
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

        # Atualiza UI
        color_map = {"Boa Postura": "green", "Aguardando Calibração": "yellow"}
        color_name = color_map.get(posture_status, "red")
        self.status_val_label.configure(text=posture_status, text_color=color_name)

        # Log para IC
        self.log_data(posture_status)

        return frame

    def classify_enhanced_posture(self, nose, l_shoulder, r_shoulder, head_dist, shoulder_width, z_depth):
        # 1. Ombros Desalinhados (Y diff)
        if abs(l_shoulder[1] - r_shoulder[1]) > 0.05:
            return "Ombros Desalinhados!"
        
        # 2. Cabeça Baixa (Comparado à calibração)
        if head_dist < (self.base_head_dist * 0.75):
            return "Cabeça Muito Baixa!"

        # 3. Projetado para Frente (Z-depth ou Largura dos Ombros)
        # Se os ombros aumentarem muito de largura, você se aproximou da câmera
        if shoulder_width > (self.base_shoulder_width * 1.25):
            return "Muito Perto da Tela!"
        
        # Se o Z dos ombros diminuir (ficar mais negativo), você se inclinou para frente
        if z_depth < (self.base_z_depth - 0.15):
            return "Inclinado para Frente!"

        return "Boa Postura"

    def on_closing(self):
        self.running = False
        self.pose.close()
        self.destroy()

if __name__ == "__main__":
    app = PostureDetectorApp()
    app.mainloop()
