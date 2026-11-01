"""Deteccao de disponibilidade e inferencia YOLO por OpenVINO otimizado."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def check_availability(model_path: Path) -> tuple[bool, str, str | None]:
    """Retorna se o teste OpenVINO pode ser executado exclusivamente na CPU."""

    if not model_path.exists():
        return False, "modelo OpenVINO otimizado ausente", None
    try:
        import openvino as ov

        devices = ov.Core().available_devices
    except (ImportError, RuntimeError) as exc:
        return False, f"OpenVINO indisponivel: {exc}", None
    if "CPU" not in devices:
        return False, "CPU sem suporte OpenVINO neste computador", None
    core = ov.Core()
    try:
        model = core.read_model(str(model_path))
        core.compile_model(model, "CPU")
    except RuntimeError:
        return False, "CPU nao conseguiu compilar o modelo OpenVINO", None
    return True, "OpenVINO pronto em CPU", "CPU"


class OpenVinoRunner:
    """Executor direto do IR OpenVINO; nao usa o backend PyTorch nem ONNX Runtime."""

    def __init__(self, model_path: Path, device: str) -> None:
        import openvino as ov

        self.core = ov.Core()
        model = self.core.read_model(str(model_path))
        self.compiled_model = self.core.compile_model(model, device)
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)

    def infer(self, tensor: np.ndarray) -> np.ndarray:
        return self.compiled_model({self.input_layer: tensor})[self.output_layer]
