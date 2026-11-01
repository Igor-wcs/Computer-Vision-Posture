"""Nucleo reutilizavel do benchmark guiado de postura.

Este modulo nao depende da webcam nem da interface grafica. Isso permite testar o
protocolo, o registro e a classificacao sem abrir janelas ou carregar os modelos.
"""

from __future__ import annotations

import csv
import os
import shutil
import statistics
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable


GOOD_POSTURE = "Boa Postura"
SHOULDERS = "Ombros Desalinhados!"
FORWARD = "Inclinado para Frente!"
HEAD_DOWN = "Cabeça Muito Baixa!"
TOO_FAR = "Muito Longe da Tela!"

LOG_COLUMNS = [
    "Timestamp",
    "FPS",
    "CPU_Usage_Percent",
    "RAM_Usage_MB",
    "Posture_Status",
    "Nose_X",
    "Nose_Y",
    "Head_Dist",
]


def classify_posture(
    left_shoulder: tuple[float, float],
    right_shoulder: tuple[float, float],
    head_distance: float,
    shoulder_width: float,
    base_head_distance: float,
    base_shoulder_width: float,
) -> str:
    """Classifica uma amostra com os mesmos limites dos scripts originais."""

    if abs(left_shoulder[1] - right_shoulder[1]) > 0.05:
        return SHOULDERS
    if head_distance < base_head_distance * 0.75:
        return HEAD_DOWN
    if shoulder_width > base_shoulder_width * 1.20:
        return FORWARD
    if shoulder_width < base_shoulder_width * 0.80:
        return TOO_FAR
    return GOOD_POSTURE


@dataclass(frozen=True)
class ProtocolStep:
    instruction: str
    expected_status: str
    hold_seconds: float = 1.5


DEFAULT_PROTOCOL = (
    ProtocolStep("Mantenha a postura correta para iniciar.", GOOD_POSTURE),
    ProtocolStep("Incline-se para a esquerda ate os ombros ficarem desalinhados.", SHOULDERS),
    ProtocolStep("Volte para a postura correta.", GOOD_POSTURE),
    ProtocolStep("Incline-se para a direita ate os ombros ficarem desalinhados.", SHOULDERS),
    ProtocolStep("Volte para a postura correta.", GOOD_POSTURE),
    ProtocolStep("Incline-se para frente, mantendo a cabeca reta.", FORWARD),
    ProtocolStep("Volte para a postura correta.", GOOD_POSTURE),
    ProtocolStep("Abaixe a cabeca.", HEAD_DOWN),
    ProtocolStep("Volte para a postura correta.", GOOD_POSTURE),
    ProtocolStep("Afaste-se da camera.", TOO_FAR),
    ProtocolStep("Volte para a postura correta para concluir.", GOOD_POSTURE),
)


class GuidedProtocol:
    """Maquina de estados do roteiro, avancando apos uma postura sustentada."""

    def __init__(
        self,
        steps: Iterable[ProtocolStep] = DEFAULT_PROTOCOL,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self.steps = tuple(steps)
        if not self.steps:
            raise ValueError("O protocolo precisa de ao menos uma etapa")
        self.time_fn = time_fn
        self.index = 0
        self.started_at = self.time_fn()
        self.match_started_at: float | None = None
        self.completed_at: float | None = None

    @property
    def current(self) -> ProtocolStep | None:
        return None if self.completed else self.steps[self.index]

    @property
    def completed(self) -> bool:
        return self.completed_at is not None

    @property
    def elapsed(self) -> float:
        end = self.completed_at if self.completed else self.time_fn()
        return max(0.0, end - self.started_at)

    @property
    def progress(self) -> float:
        return 1.0 if self.completed else self.index / len(self.steps)

    def update(self, status: str) -> bool:
        """Atualiza o roteiro e retorna True apenas quando muda de etapa."""

        if self.completed:
            return False
        now = self.time_fn()
        if status == self.current.expected_status:
            if self.match_started_at is None:
                self.match_started_at = now
            if now - self.match_started_at >= self.current.hold_seconds:
                self.index += 1
                self.match_started_at = None
                if self.index >= len(self.steps):
                    self.completed_at = now
                return True
        else:
            self.match_started_at = None
        return False


class SessionLogger:
    """Registra no maximo uma amostra por segundo, somente durante o teste."""

    def __init__(self, path: Path, interval_seconds: float = 1.0) -> None:
        self.path = Path(path)
        self.interval_seconds = interval_seconds
        self.active = False
        self.last_sample_at = 0.0
        self.session_id = ""
        self.samples: list[dict[str, float | str]] = []

    def start(self) -> str:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists() or self.path.stat().st_size == 0:
            with self.path.open("w", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=LOG_COLUMNS).writeheader()
        self.active = True
        self.last_sample_at = 0.0
        self.samples = []
        self.session_id = uuid.uuid4().hex[:8]
        return self.session_id

    def record(
        self,
        *,
        fps: float,
        cpu: float,
        ram_mb: float,
        status: str,
        nose_x: float,
        nose_y: float,
        head_distance: float,
        now: float | None = None,
    ) -> bool:
        now = time.monotonic() if now is None else now
        if not self.active or now - self.last_sample_at < self.interval_seconds:
            return False
        self.last_sample_at = now
        row: dict[str, float | str] = {
            "Timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "FPS": round(fps, 2),
            "CPU_Usage_Percent": round(cpu, 2),
            "RAM_Usage_MB": round(ram_mb, 2),
            "Posture_Status": status,
            "Nose_X": round(nose_x, 5),
            "Nose_Y": round(nose_y, 5),
            "Head_Dist": round(head_distance, 5),
        }
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=LOG_COLUMNS).writerow(row)
        self.samples.append(row)
        return True

    def stop(self) -> dict[str, dict[str, float]]:
        self.active = False
        return summarize_samples(self.samples)


def summarize_samples(rows: Iterable[dict[str, float | str]]) -> dict[str, dict[str, float]]:
    rows = list(rows)
    definitions = {
        "FPS": "FPS",
        "CPU": "CPU_Usage_Percent",
        "RAM": "RAM_Usage_MB",
    }
    summary: dict[str, dict[str, float]] = {}
    for label, column in definitions.items():
        values = [float(row[column]) for row in rows]
        if values:
            summary[label] = {
                "min": min(values),
                "mean": statistics.fmean(values),
                "max": max(values),
            }
    nose_y = [float(row["Nose_Y"]) for row in rows]
    jitter = [abs(current - previous) for previous, current in zip(nose_y, nose_y[1:])]
    if jitter:
        summary["Jitter"] = {
            "min": min(jitter),
            "mean": statistics.fmean(jitter),
            "max": max(jitter),
        }
    return summary


GENERATED_FILES = (
    "log_mediapipe.csv",
    "log_yolo.csv",
    "log_yolo_onnx.csv",
    "log_yolo_openvino.csv",
    "comparativo_modelos.png",
    "resumo_estatistico.csv",
    "relatorio_benchmark.md",
)


def archive_generated_files(project_dir: Path) -> list[Path]:
    """Move resultados para um arquivo recuperavel em vez de apaga-los."""

    project_dir = Path(project_dir)
    existing = [project_dir / name for name in GENERATED_FILES if (project_dir / name).exists()]
    if not existing:
        return []
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    destination = project_dir / "logs" / "archive" / stamp
    destination.mkdir(parents=True, exist_ok=True)
    moved = []
    for source in existing:
        target = destination / source.name
        shutil.move(str(source), str(target))
        moved.append(target)
    return moved


def cpu_name() -> str:
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        ) as key:
            value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            if value.strip():
                return value.strip()
    except (ImportError, OSError):
        pass
    return os.environ.get("PROCESSOR_IDENTIFIER") or os.environ.get("PROCESSOR_ARCHITECTURE", "Desconhecida")
