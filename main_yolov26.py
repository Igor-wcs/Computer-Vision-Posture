import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk
import threading
from ultralytics import YOLO
import time
import csv
import psutil
import os

# Configurações iniciais do CustomTkinter
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class PostureDetectorAppYOLO(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Detector de Postura - YOLOv26")
        self.geometry("900x750")

        # Inicializa YOLOv26-pose
        self.model = YOLO('yolo26n-pose.pt')

        # Variáveis de Calibração
        self.calibrated = False
        self.base_head_dist = 0.15 
        self.base_shoulder_width = 0.2
        
        # Configuração da UI
        self.setup_ui()

        # Variáveis de controle
        self.cap = cv2.VideoCapture(0)
        self.running = True
        
        # Logging para IC
        self.log_file = "log_yolo.csv"
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
                writer.writerow(['Timestamp', 'FPS', 'CPU_Usage_Percent', 'RAM_Usage_MB', 'Posture_Status', 'Nose_X', 'Nose_Y', 'Head_Dist'])

    def log_data(self, status, nose_x=0, nose_y=0, head_dist=0):
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed > 1.0:
            fps = self.frame_count / elapsed
            cpu = psutil.cpu_percent()
            ram = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
            
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S"), 
                    round(fps, 2), 
                    cpu, 
                    round(ram, 2), 
                    status,
                    round(nose_x, 4),
                    round(nose_y, 4),
                    round(head_dist, 4)
                ])
            
            self.frame_count = 0
            self.start_time = time.time()

    def setup_ui(self):
        # Grid layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Header
        self.header_label = ctk.CTkLabel(self, text="Monitoramento de Postura - YOLOv26", font=ctk.CTkFont(size=24, weight="bold"))
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
        results = self.model(frame, verbose=False)[0]
        posture_status = "Aguardando Calibração"
        curr_nose_x, curr_nose_y, curr_head_dist = 0, 0, 0
        
        if results.keypoints is not None and len(results.keypoints.xyn) > 0:
            kpts = results.keypoints.xyn[0].cpu().numpy()
            if len(kpts) > 6:
                nose = kpts[0]
                l_shoulder = kpts[5]
                r_shoulder = kpts[6]

                if not (np.all(nose == 0) or np.all(l_shoulder == 0) or np.all(r_shoulder == 0)):
                    mid_shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2
                    curr_head_dist = abs(nose[1] - mid_shoulder_y)
                    current_shoulder_width = abs(l_shoulder[0] - r_shoulder[0])
                    
                    curr_nose_x, curr_nose_y = nose[0], nose[1]

                    if hasattr(self, 'calibrating') and self.calibrating:
                        self.base_head_dist = curr_head_dist
                        self.base_shoulder_width = current_shoulder_width
                        self.calibrating = False

                    if self.calibrated:
                        posture_status = self.classify_posture_yolo(
                            nose, l_shoulder, r_shoulder, 
                            curr_head_dist, current_shoulder_width
                        )
            frame = results.plot()

        color_map = {"Boa Postura": "green", "Aguardando Calibração": "yellow"}
        color_name = color_map.get(posture_status, "red")
        self.status_val_label.configure(text=posture_status, text_color=color_name)

        # Log para IC (incluindo coordenadas para Jitter)
        self.log_data(posture_status, curr_nose_x, curr_nose_y, curr_head_dist)

        return frame

    def classify_posture_yolo(self, nose, l_shoulder, r_shoulder, head_dist, shoulder_width):
        # 1. Ombros Desalinhados (Y diff > 5%)
        if abs(l_shoulder[1] - r_shoulder[1]) > 0.05:
            return "Ombros Desalinhados!"
        
        # 2. Cabeça Baixa (Desvio > 25%)
        if head_dist < (self.base_head_dist * 0.75):
            return "Cabeça Muito Baixa!"
        
        # 3. Inclinado para Frente (Expansão > 20%)
        if shoulder_width > (self.base_shoulder_width * 1.20):
            return "Inclinado para Frente!"
        
        # 4. Muito Longe (Retração > 20%)
        if shoulder_width < (self.base_shoulder_width * 0.80):
            return "Muito Longe da Tela!"
            
        return "Boa Postura"

    def on_closing(self):
        self.running = False
        self.destroy()

if __name__ == "__main__":
    app = PostureDetectorAppYOLO()
    app.mainloop()
