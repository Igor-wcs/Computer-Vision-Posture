"""Interface unica para executar o benchmark padronizado.

Uso recomendado: ``python launcher.py`` com Python 3.12.
"""

from __future__ import annotations

import queue
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import customtkinter as ctk
import cv2
import numpy as np
import psutil
from PIL import Image

from benchmark_core import (
    GuidedProtocol,
    SessionLogger,
    archive_generated_files,
    classify_posture,
)
from onnx_pose import decode_pose_output, draw_detection, preprocess
from openvino_pose import OpenVinoRunner, check_availability as check_openvino_availability
from reporting import generate_report, missing_logs


# No executavel unico, o PyInstaller extrai os modelos em uma pasta temporaria.
# Os resultados permanecem ao lado do .exe para que a versao seja portatil.
RESOURCE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
PROJECT_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else RESOURCE_DIR
MODEL_CONFIG = {
    "mediapipe": {"title": "MediaPipe", "log": "log_mediapipe.csv"},
    "yolo": {"title": "YOLOv26 (PyTorch)", "log": "log_yolo.csv"},
    "yolo_onnx": {"title": "YOLOv26 (ONNX Runtime)", "log": "log_yolo_onnx.csv"},
    "yolo_openvino": {"title": "YOLOv26 (OpenVINO FP16)", "log": "log_yolo_openvino.csv"},
}

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


@dataclass
class PoseObservation:
    frame: np.ndarray
    status: str
    nose_x: float = 0.0
    nose_y: float = 0.0
    head_distance: float = 0.0
    fps: float = 0.0
    cpu: float = 0.0
    ram_mb: float = 0.0
    error: str | None = None


class BaseBackend:
    name = ""

    def __init__(self) -> None:
        self.calibrated = False
        self.calibrating = False
        self.calibration_started = 0.0
        self.calibration_samples: list[tuple[float, float]] = []
        self.base_head_distance = 0.15
        self.base_shoulder_width = 0.20

    def request_calibration(self) -> None:
        self.calibrated = False
        self.calibrating = True
        self.calibration_started = time.monotonic()
        self.calibration_samples.clear()

    def _update_calibration(self, head_distance: float, shoulder_width: float) -> None:
        if not self.calibrating:
            return
        if head_distance > 0 and shoulder_width > 0:
            self.calibration_samples.append((head_distance, shoulder_width))
        if time.monotonic() - self.calibration_started >= 2.0 and len(self.calibration_samples) >= 5:
            self.base_head_distance = statistics.median(item[0] for item in self.calibration_samples)
            self.base_shoulder_width = statistics.median(item[1] for item in self.calibration_samples)
            self.calibrating = False
            self.calibrated = True

    def close(self) -> None:
        pass


class MediaPipeBackend(BaseBackend):
    name = "MediaPipe"

    def __init__(self) -> None:
        super().__init__()
        import mediapipe as mp

        self.mp = mp
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, frame: np.ndarray) -> PoseObservation:
        result = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        status = "Aguardando deteccao"
        nose_x = nose_y = head_distance = 0.0
        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            pose = self.mp.solutions.pose.PoseLandmark
            nose = landmarks[pose.NOSE.value]
            left = landmarks[pose.LEFT_SHOULDER.value]
            right = landmarks[pose.RIGHT_SHOULDER.value]
            nose_x, nose_y = nose.x, nose.y
            left_xy = (left.x, left.y)
            right_xy = (right.x, right.y)
            shoulder_mid_y = (left.y + right.y) / 2
            head_distance = abs(nose.y - shoulder_mid_y)
            shoulder_width = abs(left.x - right.x)
            self._update_calibration(head_distance, shoulder_width)
            if self.calibrated:
                status = classify_posture(
                    left_xy,
                    right_xy,
                    head_distance,
                    shoulder_width,
                    self.base_head_distance,
                    self.base_shoulder_width,
                )
            elif self.calibrating:
                status = "Calibrando... mantenha a postura correta"
            else:
                status = "Aguardando calibracao"
            self.mp.solutions.drawing_utils.draw_landmarks(
                frame,
                result.pose_landmarks,
                self.mp.solutions.pose.POSE_CONNECTIONS,
            )
        return PoseObservation(frame, status, nose_x, nose_y, head_distance)

    def close(self) -> None:
        self.pose.close()


class YoloBackend(BaseBackend):
    name = "YOLOv26 (PyTorch)"

    def __init__(self) -> None:
        super().__init__()
        from ultralytics import YOLO

        model_path = RESOURCE_DIR / "yolo26n-pose.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo nao encontrado: {model_path.name}")
        self.model = YOLO(str(model_path))

    def process(self, frame: np.ndarray) -> PoseObservation:
        result = self.model(frame, verbose=False)[0]
        status = "Aguardando deteccao"
        nose_x = nose_y = head_distance = 0.0
        if result.keypoints is not None and len(result.keypoints.xyn) > 0:
            points = result.keypoints.xyn[0].cpu().numpy()
            if len(points) > 6:
                nose, left, right = points[0], points[5], points[6]
                if not (np.all(nose == 0) or np.all(left == 0) or np.all(right == 0)):
                    nose_x, nose_y = float(nose[0]), float(nose[1])
                    head_distance = abs(nose_y - float((left[1] + right[1]) / 2))
                    shoulder_width = abs(float(left[0] - right[0]))
                    self._update_calibration(head_distance, shoulder_width)
                    if self.calibrated:
                        status = classify_posture(
                            (float(left[0]), float(left[1])),
                            (float(right[0]), float(right[1])),
                            head_distance,
                            shoulder_width,
                            self.base_head_distance,
                            self.base_shoulder_width,
                        )
                    elif self.calibrating:
                        status = "Calibrando... mantenha a postura correta"
                    else:
                        status = "Aguardando calibracao"
        return PoseObservation(result.plot(), status, nose_x, nose_y, head_distance)


class OnnxRuntimeBackend(BaseBackend):
    """YOLO exportado para ONNX e executado diretamente pelo ONNX Runtime."""

    name = "YOLOv26 (ONNX Runtime)"

    def __init__(self) -> None:
        super().__init__()
        import onnxruntime as ort

        model_path = RESOURCE_DIR / "yolo26n-pose.onnx"
        if not model_path.exists():
            raise FileNotFoundError(
                "Modelo ONNX nao encontrado: yolo26n-pose.onnx. "
                "Mantenha o arquivo exportado junto do launcher."
            )
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def process(self, frame: np.ndarray) -> PoseObservation:
        tensor, scale, pad_left, pad_top = preprocess(frame)
        output = self.session.run(None, {self.input_name: tensor})[0]
        detection = decode_pose_output(output, frame.shape[:2], scale, pad_left, pad_top)
        status = "Aguardando deteccao"
        nose_x = nose_y = head_distance = 0.0
        if detection is not None:
            points = detection.keypoints
            nose, left, right = points[0], points[5], points[6]
            if min(nose[2], left[2], right[2]) >= 0.25:
                nose_x, nose_y = float(nose[0]), float(nose[1])
                head_distance = abs(nose_y - float((left[1] + right[1]) / 2))
                shoulder_width = abs(float(left[0] - right[0]))
                self._update_calibration(head_distance, shoulder_width)
                if self.calibrated:
                    status = classify_posture(
                        (float(left[0]), float(left[1])),
                        (float(right[0]), float(right[1])),
                        head_distance,
                        shoulder_width,
                        self.base_head_distance,
                        self.base_shoulder_width,
                    )
                elif self.calibrating:
                    status = "Calibrando... mantenha a postura correta"
                else:
                    status = "Aguardando calibracao"
            frame = draw_detection(frame, detection)
        return PoseObservation(frame, status, nose_x, nose_y, head_distance)


class OpenVinoBackend(BaseBackend):
    """YOLO IR otimizado em FP16, executado diretamente pelo OpenVINO."""

    name = "YOLOv26 (OpenVINO FP16)"

    def __init__(self, device: str) -> None:
        super().__init__()
        model_path = RESOURCE_DIR / "yolo26n-pose_openvino_model" / "yolo26n-pose.xml"
        self.runner = OpenVinoRunner(model_path, device)

    def process(self, frame: np.ndarray) -> PoseObservation:
        tensor, scale, pad_left, pad_top = preprocess(frame)
        output = self.runner.infer(tensor)
        detection = decode_pose_output(output, frame.shape[:2], scale, pad_left, pad_top)
        status = "Aguardando deteccao"
        nose_x = nose_y = head_distance = 0.0
        if detection is not None:
            points = detection.keypoints
            nose, left, right = points[0], points[5], points[6]
            if min(nose[2], left[2], right[2]) >= 0.25:
                nose_x, nose_y = float(nose[0]), float(nose[1])
                head_distance = abs(nose_y - float((left[1] + right[1]) / 2))
                shoulder_width = abs(float(left[0] - right[0]))
                self._update_calibration(head_distance, shoulder_width)
                if self.calibrated:
                    status = classify_posture(
                        (float(left[0]), float(left[1])),
                        (float(right[0]), float(right[1])),
                        head_distance,
                        shoulder_width,
                        self.base_head_distance,
                        self.base_shoulder_width,
                    )
                elif self.calibrating:
                    status = "Calibrando... mantenha a postura correta"
                else:
                    status = "Aguardando calibracao"
            frame = draw_detection(frame, detection)
        return PoseObservation(frame, status, nose_x, nose_y, head_distance)


class VideoWorker(threading.Thread):
    def __init__(self, backend: BaseBackend, camera_index: int) -> None:
        super().__init__(daemon=True)
        self.backend = backend
        self.camera_index = camera_index
        self.output: queue.Queue[PoseObservation] = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()
        self.camera_fps_limit: float | None = None

    def _publish(self, observation: PoseObservation) -> None:
        while self.output.full():
            try:
                self.output.get_nowait()
            except queue.Empty:
                break
        self.output.put_nowait(observation)

    def _measure_camera_fps(self, capture: cv2.VideoCapture) -> float | None:
        """Mede a cadencia de captura quando o driver nao informa o FPS."""

        started = time.monotonic()
        deadline = started + 3.0
        frames = 0
        while time.monotonic() < deadline and not self.stop_event.is_set():
            ok, _ = capture.read()
            if not ok:
                break
            frames += 1
        elapsed = time.monotonic() - started
        measured = frames / elapsed if elapsed > 0 else 0.0
        return measured if measured >= 1 else None

    def run(self) -> None:
        capture = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(self.camera_index)
        if not capture.isOpened():
            empty = np.zeros((450, 800, 3), dtype=np.uint8)
            self._publish(PoseObservation(empty, "Camera indisponivel", error=f"Nao foi possivel abrir a camera {self.camera_index}."))
            return
        reported_fps = capture.get(cv2.CAP_PROP_FPS)
        if 1 <= reported_fps <= 240:
            # Informacao declarada pelo driver/dispositivo: e a fonte preferida.
            self.camera_fps_limit = reported_fps
        else:
            # Alguns drivers retornam 0; nesse caso, mede-se somente a captura,
            # antes de incluir o tempo de inferencia no benchmark.
            self.camera_fps_limit = self._measure_camera_fps(capture)
        process = psutil.Process()
        frames = 0
        fps = 0.0
        counter_started = time.monotonic()
        process.cpu_percent(None)
        try:
            while not self.stop_event.is_set():
                ok, frame = capture.read()
                if not ok:
                    self._publish(PoseObservation(np.zeros((450, 800, 3), dtype=np.uint8), "Falha na camera", error="A webcam deixou de fornecer imagens."))
                    return
                observation = self.backend.process(frame)
                frames += 1
                elapsed = time.monotonic() - counter_started
                if elapsed >= 1.0:
                    fps = frames / elapsed
                    frames = 0
                    counter_started = time.monotonic()
                observation.fps = fps
                observation.cpu = process.cpu_percent(None) / max(1, psutil.cpu_count())
                observation.ram_mb = process.memory_info().rss / (1024 * 1024)
                self._publish(observation)
        except Exception as exc:
            empty = np.zeros((450, 800, 3), dtype=np.uint8)
            self._publish(PoseObservation(empty, "Erro no processamento", error=str(exc)))
        finally:
            capture.release()
            self.backend.close()

    def stop(self) -> None:
        self.stop_event.set()


class BenchmarkApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Benchmark guiado de postura")
        self.geometry("1220x860")
        self.minsize(1040, 760)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.current_model: str | None = None
        self.backend: BaseBackend | None = None
        self.worker: VideoWorker | None = None
        self.protocol_run: GuidedProtocol | None = None
        self.logger: SessionLogger | None = None
        self.latest: PoseObservation | None = None
        self.last_summary: dict[str, dict[str, float]] = {}
        self.calibration_ready_announced = False
        self._context_action = None
        self.sound_enabled = ctk.BooleanVar(value=True)
        self.camera_options: dict[str, int] = {}
        self.camera_names: dict[str, str] = {}
        self.test_camera_fps_limits: set[float] = set()
        self.camera_var = ctk.StringVar(value="Camera 0")
        openvino_model = RESOURCE_DIR / "yolo26n-pose_openvino_model" / "yolo26n-pose.xml"
        self.openvino_available, self.openvino_status, self.openvino_device = check_openvino_availability(openvino_model)

        self._build_ui()
        self.bind_all("<space>", self._on_space)
        self._refresh_log_status()
        self.after(50, self._poll_video)

    def _build_ui(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=22, pady=(18, 8))
        ctk.CTkLabel(header, text="Benchmark de postura", font=ctk.CTkFont(size=28, weight="bold")).pack(side="left")
        flow = "MediaPipe  →  YOLO/PyTorch  →  YOLO/ONNX Runtime"
        if self.openvino_available:
            flow += "  →  YOLO/OpenVINO"
        ctk.CTkLabel(header, text=flow + "  →  Relatorio", text_color="#9aa4b2").pack(side="right")

        sidebar = ctk.CTkFrame(self, width=315)
        sidebar.grid(row=1, column=0, sticky="nsew", padx=(22, 10), pady=(8, 20))
        sidebar.grid_propagate(False)

        ctk.CTkLabel(sidebar, text="Clique em um modelo para iniciar.", font=ctk.CTkFont(size=14), text_color="#b4bdc9").pack(anchor="w", padx=18, pady=(18, 8))
        ctk.CTkLabel(sidebar, text="1. Escolha o modelo", font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w", padx=18, pady=(2, 8))
        self.mp_button = ctk.CTkButton(sidebar, text="Executar MediaPipe", command=lambda: self.select_model("mediapipe"))
        self.mp_button.pack(fill="x", padx=18, pady=5)
        self.yolo_button = ctk.CTkButton(sidebar, text="Executar YOLOv26 (PyTorch)", command=lambda: self.select_model("yolo"))
        self.yolo_button.pack(fill="x", padx=18, pady=5)
        self.onnx_button = ctk.CTkButton(sidebar, text="Executar YOLOv26 (ONNX Runtime)", command=lambda: self.select_model("yolo_onnx"))
        self.onnx_button.pack(fill="x", padx=18, pady=(5, 14))
        if self.openvino_available:
            self.openvino_button = ctk.CTkButton(sidebar, text="Executar YOLOv26 (OpenVINO FP16)", command=lambda: self.select_model("yolo_openvino"))
            self.openvino_button.pack(fill="x", padx=18, pady=(0, 8))
            ctk.CTkLabel(sidebar, text=self.openvino_status, text_color="#8b95a5", font=ctk.CTkFont(size=12)).pack(anchor="w", padx=18, pady=(0, 8))
        else:
            self.openvino_button = None
            ctk.CTkLabel(sidebar, text=f"OpenVINO indisponivel: {self.openvino_status}", text_color="#8b95a5", wraplength=275).pack(anchor="w", padx=18, pady=(0, 8))

        ctk.CTkLabel(sidebar, text="Camera", font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=18, pady=(6, 4))
        self.camera_menu = ctk.CTkOptionMenu(sidebar, variable=self.camera_var, values=["Camera 0"], command=self._on_camera_change)
        self.camera_menu.pack(fill="x", padx=18, pady=4)
        self.refresh_cameras_button = ctk.CTkButton(sidebar, text="Atualizar cameras", command=self._refresh_camera_list, height=28, fg_color="#566173")
        self.refresh_cameras_button.pack(fill="x", padx=18, pady=(2, 10))

        ctk.CTkLabel(sidebar, text="2. Calibre e colete", font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w", padx=18, pady=(8, 8))
        self.calibrate_button = ctk.CTkButton(sidebar, text="Calibrar", command=self.calibrate, state="disabled")
        self.calibrate_button.pack(fill="x", padx=18, pady=5)
        self.start_button = ctk.CTkButton(sidebar, text="Iniciar teste guiado", command=self.start_test, state="disabled", fg_color="#1f9d55")
        self.start_button.pack(fill="x", padx=18, pady=5)
        self.retry_button = ctk.CTkButton(sidebar, text="Testar novamente", command=self.start_test, state="disabled", fg_color="#566173")
        self.retry_button.pack(fill="x", padx=18, pady=5)
        self.next_button = ctk.CTkButton(sidebar, text="Proximo passo", command=self.next_step, state="disabled")
        self.next_button.pack(fill="x", padx=18, pady=5)
        self.sound_checkbox = ctk.CTkCheckBox(
            sidebar,
            text="Som de avanço",
            variable=self.sound_enabled,
            font=ctk.CTkFont(size=13),
            checkbox_width=18,
            checkbox_height=18,
        )
        self.sound_checkbox.pack(anchor="w", padx=20, pady=(5, 10))

        ctk.CTkLabel(sidebar, text="3. Resultados", font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w", padx=18, pady=(10, 8))
        self.report_button = ctk.CTkButton(sidebar, text="Gerar relatorio final", command=self.show_report)
        self.report_button.pack(fill="x", padx=18, pady=5)
        self.archive_button = ctk.CTkButton(sidebar, text="Arquivar logs/resultados antigos", command=self.archive_results, fg_color="#8b5e34")
        self.archive_button.pack(fill="x", padx=18, pady=5)
        self.log_status = ctk.CTkLabel(sidebar, text="", justify="left", wraplength=275, text_color="#9aa4b2")
        self.log_status.pack(anchor="w", padx=18, pady=(10, 10))

        content = ctk.CTkFrame(self)
        content.grid(row=1, column=1, sticky="nsew", padx=(10, 22), pady=(8, 20))
        content.grid_columnconfigure(0, weight=1)
        content.grid_rowconfigure(1, weight=1)

        self.guide = ctk.CTkFrame(content, fg_color="#282d35")
        self.guide.grid(row=0, column=0, sticky="ew", padx=14, pady=14)
        self.guide.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(self.guide, text="ORIENTACAO ATUAL", font=ctk.CTkFont(size=14, weight="bold"), text_color="#b4bdc9").grid(row=0, column=0, columnspan=2, sticky="w", padx=16, pady=(12, 0))
        self.model_label = ctk.CTkLabel(self.guide, text="Escolha um modelo para iniciar", font=ctk.CTkFont(size=22, weight="bold"))
        self.model_label.grid(row=1, column=0, sticky="w", padx=16, pady=(2, 2))
        self.timer_label = ctk.CTkLabel(self.guide, text="00:00", font=ctk.CTkFont(size=22, weight="bold"), text_color="#e6eaf0")
        self.timer_label.grid(row=1, column=1, padx=16, pady=(2, 2))
        self.instruction_label = ctk.CTkLabel(self.guide, text="No painel esquerdo, escolha um modelo para comecar.", justify="left", wraplength=740, font=ctk.CTkFont(size=17, weight="bold"), text_color="#ffffff")
        self.instruction_label.grid(row=2, column=0, columnspan=2, sticky="w", padx=16, pady=(4, 8))
        self.context_action_button = ctk.CTkButton(self.guide, text="Escolha um modelo no painel esquerdo", height=38, state="disabled")
        self.context_action_button.grid(row=3, column=0, columnspan=2, sticky="ew", padx=16, pady=(0, 8))
        self.progress = ctk.CTkProgressBar(self.guide)
        self.progress.grid(row=4, column=0, columnspan=2, sticky="ew", padx=16, pady=(2, 12))
        self.progress.set(0)

        self.video_label = ctk.CTkLabel(content, text="A visualizacao da webcam aparecera aqui", fg_color="#090d16", corner_radius=10)
        self.video_label.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 10))
        self.status_label = ctk.CTkLabel(content, text="Aguardando", font=ctk.CTkFont(size=20, weight="bold"), text_color="#f0c75e")
        self.status_label.grid(row=2, column=0, pady=(2, 4))
        self.metrics_label = ctk.CTkLabel(content, text="FPS —  |  CPU —  |  RAM —", text_color="#9aa4b2")
        self.metrics_label.grid(row=3, column=0, pady=(0, 12))
        self._refresh_camera_list()

    @staticmethod
    def _windows_camera_names() -> list[str]:
        """Obtem os nomes amigaveis que o Windows expoe para dispositivos de camera.

        O OpenCV abre cameras por indice e nao fornece o nome do dispositivo. A
        correspondencia e apresentada junto do indice, que continua sendo usado
        para a abertura da camera.
        """

        command = (
            "Get-CimInstance Win32_PnPEntity -ErrorAction SilentlyContinue | "
            "Where-Object { $_.PNPClass -eq 'Camera' -or $_.PNPClass -eq 'Image' } | "
            "Select-Object -ExpandProperty Name"
        )
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", command],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return []
        names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return list(dict.fromkeys(names))

    def _detect_cameras(self) -> tuple[dict[str, int], dict[str, str]]:
        cameras: dict[str, int] = {}
        names_by_choice: dict[str, str] = {}
        windows_names = self._windows_camera_names()
        detected_position = 0
        for index in range(6):
            capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if not capture.isOpened():
                capture.release()
                capture = cv2.VideoCapture(index)
            if capture.isOpened():
                friendly_name = (
                    windows_names[detected_position]
                    if detected_position < len(windows_names)
                    else f"Camera {index}"
                )
                choice = f"{index} — {friendly_name}"
                cameras[choice] = index
                names_by_choice[choice] = friendly_name
                detected_position += 1
            capture.release()
        if not cameras:
            cameras = {"0 — Camera 0": 0}
            names_by_choice = {"0 — Camera 0": "Camera 0"}
        return cameras, names_by_choice

    def _refresh_camera_list(self) -> None:
        self.camera_options, self.camera_names = self._detect_cameras()
        choices = list(self.camera_options)
        self.camera_menu.configure(values=choices)
        if self.camera_var.get() not in self.camera_options:
            self.camera_var.set(choices[0])

    def _on_camera_change(self, _choice: str) -> None:
        if self.protocol_run and not self.protocol_run.completed:
            self.instruction_label.configure(text="A camera nao pode ser trocada durante um teste. Conclua ou reinicie a sessao.")
            return
        if self.current_model:
            self.select_model(self.current_model)

    def _selected_camera_index(self) -> int:
        return self.camera_options.get(self.camera_var.get(), 0)

    def _selected_camera_name(self) -> str:
        return self.camera_names.get(self.camera_var.get(), "Camera nao identificada")

    def select_model(self, model: str) -> None:
        if self.protocol_run and not self.protocol_run.completed:
            self.instruction_label.configure(text="Conclua o teste atual antes de trocar de modelo.")
            return
        self._stop_worker()
        self.current_model = model
        self.protocol_run = None
        self.logger = None
        self.calibration_ready_announced = False
        self._hide_context_action("Carregando modelo e webcam...")
        self.progress.set(0)
        self.timer_label.configure(text="00:00")
        self.start_button.configure(state="disabled")
        self.retry_button.configure(state="disabled")
        self.next_button.configure(text="Proximo passo", state="disabled")
        self.calibrate_button.configure(state="disabled")
        title = MODEL_CONFIG[model]["title"]
        self.model_label.configure(text=f"Carregando {title}...")
        self.instruction_label.configure(text="Aguarde o carregamento do modelo e da webcam.")
        self.update_idletasks()
        try:
            if model == "mediapipe":
                self.backend = MediaPipeBackend()
            elif model == "yolo":
                self.backend = YoloBackend()
            elif model == "yolo_onnx":
                self.backend = OnnxRuntimeBackend()
            elif self.openvino_available and self.openvino_device:
                self.backend = OpenVinoBackend(self.openvino_device)
            else:
                raise RuntimeError("OpenVINO nao esta disponivel neste computador.")
            self.worker = VideoWorker(self.backend, self._selected_camera_index())
            self.worker.start()
        except Exception as exc:
            self.backend = None
            self.worker = None
            self.model_label.configure(text=f"Falha ao carregar {title}")
            self.instruction_label.configure(text=str(exc))
            self._hide_context_action("Modelo indisponivel")
            return
        self.model_label.configure(text=title)
        self.instruction_label.configure(text="Sente-se reto, fique visivel na camera e clique em Calibrar.")
        self.calibrate_button.configure(state="normal")
        self._show_context_action("Calibrar", self.calibrate)
        self._flash_guide()

    def calibrate(self) -> None:
        if not self.backend:
            return
        self.backend.request_calibration()
        self.calibration_ready_announced = False
        self.calibrate_button.configure(state="disabled")
        self.start_button.configure(state="disabled")
        self.instruction_label.configure(text="Calibrando por 2 segundos: mantenha a postura correta e fique parado.")
        self._hide_context_action("Calibrando...")
        self._flash_guide()

    def start_test(self) -> None:
        if not self.backend or not self.backend.calibrated or not self.current_model:
            self.instruction_label.configure(text="Calibre o modelo antes de iniciar o teste.")
            return
        self.protocol_run = GuidedProtocol()
        log_path = PROJECT_DIR / MODEL_CONFIG[self.current_model]["log"]
        self.logger = SessionLogger(log_path)
        self.logger.start()
        if self.worker and self.worker.camera_fps_limit:
            self.test_camera_fps_limits.add(self.worker.camera_fps_limit)
        self.last_summary = {}
        self._hide_context_action("Teste em andamento...")
        self.start_button.configure(state="disabled")
        self.retry_button.configure(state="disabled")
        self.next_button.configure(state="disabled")
        self.calibrate_button.configure(state="disabled")
        self._update_protocol_labels()
        self._flash_guide()

    def _update_protocol_labels(self) -> None:
        if not self.protocol_run:
            return
        elapsed = int(self.protocol_run.elapsed)
        self.timer_label.configure(text=f"{elapsed // 60:02d}:{elapsed % 60:02d}")
        self.progress.set(self.protocol_run.progress)
        if self.protocol_run.completed:
            self.instruction_label.configure(text="Teste concluido. Os logs foram interrompidos automaticamente.")
        else:
            step = self.protocol_run.current
            self.instruction_label.configure(
                text=f"Etapa {self.protocol_run.index + 1}/{len(self.protocol_run.steps)} — {step.instruction}\nMantenha a posicao ate o programa avancar."
            )

    def _complete_test(self) -> None:
        if not self.logger:
            return
        self.last_summary = self.logger.stop()
        self.logger = None
        self.progress.set(1)
        lines = ["Teste concluido. Resumo desta sessao:"]
        units = {"FPS": "", "CPU": "%", "RAM": " MB", "Jitter": ""}
        for metric, values in self.last_summary.items():
            unit = units[metric]
            lines.append(
                f"{metric}: min {values['min']:.2f}{unit} | media {values['mean']:.2f}{unit} | max {values['max']:.2f}{unit}"
            )
        self.instruction_label.configure(text="\n".join(lines))
        self.retry_button.configure(state="normal")
        if self.current_model == "mediapipe":
            self.next_button.configure(text="Ir para YOLOv26", state="normal")
            self._show_context_action("Continuar para teste YOLOv26  →", self.next_step)
        elif self.current_model == "yolo":
            self.next_button.configure(text="Ir para ONNX Runtime", state="normal")
            self._show_context_action("Continuar para teste ONNX Runtime  →", self.next_step)
        elif self.current_model == "yolo_onnx" and self.openvino_available:
            self.next_button.configure(text="Ir para OpenVINO", state="normal")
            self._show_context_action("Continuar para teste OpenVINO  →", self.next_step)
        else:
            self.next_button.configure(text="Gerar relatorio", state="normal")
            self._show_context_action("Gerar relatorio final  →", self.show_report)
        self.calibrate_button.configure(state="normal")
        self._refresh_log_status()
        self._flash_guide()

    def _show_context_action(self, text: str, command) -> None:
        self._context_action = command
        self.context_action_button.configure(text=f"{text} (Aperte espaço)", command=command, state="normal")

    def _hide_context_action(self, text: str = "Aguarde...") -> None:
        self._context_action = None
        self.context_action_button.configure(text=text, command=lambda: None, state="disabled")

    def _on_space(self, _event):
        if self._context_action is not None and self.context_action_button.cget("state") == "normal":
            self._context_action()
        return "break"

    def _flash_guide(self) -> None:
        self.guide.configure(fg_color="#3a4654")
        self.after(140, lambda: self.guide.configure(fg_color="#282d35"))

    def _play_step_sound(self) -> None:
        if not self.sound_enabled.get():
            return

        def play() -> None:
            try:
                import winsound

                # Tom explicito: nao depende de um alias de som do Windows estar configurado.
                winsound.Beep(880, 120)
                time.sleep(0.06)
                winsound.Beep(1175, 160)
            except (ImportError, RuntimeError):
                try:
                    winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
                except (NameError, RuntimeError):
                    pass

        threading.Thread(target=play, daemon=True).start()

    def next_step(self) -> None:
        if self.current_model == "mediapipe":
            self.select_model("yolo")
        elif self.current_model == "yolo":
            self.select_model("yolo_onnx")
        elif self.current_model == "yolo_onnx" and self.openvino_available:
            self.select_model("yolo_openvino")
        else:
            self.show_report()

    def show_report(self) -> None:
        missing = missing_logs(PROJECT_DIR)
        if missing:
            self.instruction_label.configure(text="Nao e possivel gerar o relatorio. Logs ausentes: " + ", ".join(missing))
            return
        if self.protocol_run and not self.protocol_run.completed:
            self.instruction_label.configure(text="Conclua o teste em andamento antes de gerar o relatorio.")
            return
        self._hide_context_action("Gerando relatorio...")
        self._stop_worker()
        self.model_label.configure(text="Gerando relatorio final...")
        self.update_idletasks()
        try:
            camera_fps_limit = max(self.test_camera_fps_limits, default=None)
            result = generate_report(PROJECT_DIR, camera_fps_limit=camera_fps_limit)
            image = Image.open(result["image"])
            image.thumbnail((840, 570), Image.Resampling.LANCZOS)
            self.report_image = ctk.CTkImage(light_image=image, dark_image=image, size=image.size)
            self.video_label.configure(image=self.report_image, text="")
            self.model_label.configure(text="Relatorio final")
            self.status_label.configure(text="Relatorio gerado com sucesso", text_color="#55d98b")
            self.metrics_label.configure(text="comparativo_modelos.png  |  resumo_estatistico.csv  |  relatorio_benchmark.md")
            self.instruction_label.configure(text="O grafico esta exibido abaixo e os tres arquivos foram salvos na pasta do projeto.")
            self.progress.set(1)
            self._show_context_action("Executar novo teste  →", self.start_new_experiment)
            self._flash_guide()
        except Exception as exc:
            self.model_label.configure(text="Falha ao gerar o relatorio")
            self.instruction_label.configure(text=str(exc))
            self._hide_context_action("Corrija os logs para gerar o relatorio")

    def start_new_experiment(self) -> None:
        self.test_camera_fps_limits.clear()
        self.select_model("mediapipe")

    def archive_results(self) -> None:
        if self.protocol_run and not self.protocol_run.completed:
            self.instruction_label.configure(text="Conclua o teste atual antes de arquivar resultados.")
            return
        moved = archive_generated_files(PROJECT_DIR)
        if moved:
            self.test_camera_fps_limits.clear()
            # A geracao do relatorio encerra a pre-visualizacao. Ao arquivar os
            # resultados, ja preparar o proximo experimento e reabrir a camera.
            self.start_new_experiment()
            self.instruction_label.configure(text=f"{len(moved)} arquivo(s) movido(s) para {moved[0].parent.relative_to(PROJECT_DIR)}.")
            self.status_label.configure(text="Resultados arquivados. Camera pronta para um novo experimento.", text_color="#55d98b")
        else:
            self.instruction_label.configure(text="Nao ha logs ou relatorios para arquivar.")
        self._refresh_log_status()

    def _refresh_log_status(self) -> None:
        missing = missing_logs(PROJECT_DIR)
        if not missing:
            text = "Logs prontos: MediaPipe, YOLO/PyTorch e YOLO/ONNX Runtime."
        else:
            text = "Faltam logs: " + ", ".join(missing) + "."
        text += "\nLogs sao aditivos; arquive antes de um experimento novo."
        if self.openvino_available:
            text += "\nOpenVINO ativo: inclua-o no benchmark para aparecer no relatorio."
        self.log_status.configure(text=text)

    def _poll_video(self) -> None:
        if self.worker:
            observation = None
            while True:
                try:
                    observation = self.worker.output.get_nowait()
                except queue.Empty:
                    break
            if observation:
                self.latest = observation
                self._display_observation(observation)
        if self.backend and self.backend.calibrated and not self.protocol_run:
            self.calibrate_button.configure(state="normal")
            self.start_button.configure(state="normal")
            if not self.calibration_ready_announced:
                self.calibration_ready_announced = True
                self.instruction_label.configure(text="Calibracao concluida. Clique em Iniciar teste guiado quando estiver pronto.")
                self._show_context_action("Iniciar teste guiado", self.start_test)
                self._flash_guide()
                self._play_step_sound()
        if self.protocol_run and not self.protocol_run.completed:
            self._update_protocol_labels()
        self.after(50, self._poll_video)

    def _display_observation(self, observation: PoseObservation) -> None:
        rgb = cv2.cvtColor(observation.frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((850, 560), Image.Resampling.LANCZOS)
        self.camera_image = ctk.CTkImage(light_image=image, dark_image=image, size=image.size)
        self.video_label.configure(image=self.camera_image, text="")
        color = "#55d98b" if observation.status == "Boa Postura" else "#f0c75e"
        if observation.status.endswith("!"):
            color = "#ff6b6b"
        self.status_label.configure(text=observation.status, text_color=color)
        self.metrics_label.configure(text=f"FPS {observation.fps:.1f}  |  CPU {observation.cpu:.1f}%  |  RAM {observation.ram_mb:.0f} MB")
        if observation.error:
            self.instruction_label.configure(text=observation.error)
        if self.protocol_run and not self.protocol_run.completed and self.logger:
            self.logger.record(
                fps=observation.fps,
                cpu=observation.cpu,
                ram_mb=observation.ram_mb,
                status=observation.status,
                nose_x=observation.nose_x,
                nose_y=observation.nose_y,
                head_distance=observation.head_distance,
            )
            changed = self.protocol_run.update(observation.status)
            if changed:
                self._update_protocol_labels()
                self._flash_guide()
                self._play_step_sound()
            if self.protocol_run.completed:
                self._complete_test()

    def _stop_worker(self) -> None:
        if self.worker:
            self.worker.stop()
            self.worker.join(timeout=3)
        self.worker = None
        self.backend = None

    def on_close(self) -> None:
        if self.logger:
            self.logger.stop()
        self._stop_worker()
        self.destroy()


if __name__ == "__main__":
    BenchmarkApp().mainloop()
