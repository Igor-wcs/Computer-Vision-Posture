"""Pre e pos-processamento de pose YOLO para inferencia direta com ONNX Runtime."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


POSE_EDGES = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
)


@dataclass
class PoseDetection:
    score: float
    box: tuple[int, int, int, int]
    keypoints: np.ndarray  # 17 x (x_normalizado, y_normalizado, confianca)


def preprocess(image: np.ndarray, size: int = 640) -> tuple[np.ndarray, float, int, int]:
    """Aplica letterbox e devolve tensor BCHW RGB, escala e preenchimento."""

    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    resized_width, resized_height = round(width * scale), round(height * scale)
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = size - resized_width, size - resized_height
    left, top = pad_x // 2, pad_y // 2
    canvas = cv2.copyMakeBorder(
        resized,
        top,
        pad_y - top,
        left,
        pad_x - left,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    tensor = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(tensor, axis=0), scale, left, top


def decode_pose_output(
    output: np.ndarray,
    frame_shape: tuple[int, int],
    scale: float,
    pad_left: int,
    pad_top: int,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> PoseDetection | None:
    """Decodifica a saida (1, 56, 8400) do YOLO pose exportado para ONNX."""

    height, width = frame_shape
    candidates = np.squeeze(output, axis=0).transpose()
    if candidates.ndim != 2 or candidates.shape[1] < 56:
        raise ValueError(f"Saida ONNX de pose inesperada: {output.shape}")
    scores = candidates[:, 4]
    valid_indices = np.flatnonzero(scores >= confidence_threshold)
    if not len(valid_indices):
        return None

    boxes: list[list[int]] = []
    confidences: list[float] = []
    for index in valid_indices:
        x, y, box_width, box_height = candidates[index, :4]
        boxes.append([int(x - box_width / 2), int(y - box_height / 2), int(box_width), int(box_height)])
        confidences.append(float(scores[index]))
    kept = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, iou_threshold)
    if len(kept) == 0:
        return None
    kept_indices = np.asarray(kept).reshape(-1)
    best = max((valid_indices[position] for position in kept_indices), key=lambda index: scores[index])
    raw_points = candidates[best, 5:56].reshape(17, 3).copy()
    raw_points[:, 0] = np.clip((raw_points[:, 0] - pad_left) / scale / width, 0.0, 1.0)
    raw_points[:, 1] = np.clip((raw_points[:, 1] - pad_top) / scale / height, 0.0, 1.0)
    x, y, box_width, box_height = candidates[best, :4]
    box = (
        int((x - box_width / 2 - pad_left) / scale),
        int((y - box_height / 2 - pad_top) / scale),
        int(box_width / scale),
        int(box_height / scale),
    )
    return PoseDetection(float(scores[best]), box, raw_points)


def draw_detection(frame: np.ndarray, detection: PoseDetection) -> np.ndarray:
    """Desenha caixa e esqueleto sem depender do executor PyTorch."""

    height, width = frame.shape[:2]
    x, y, box_width, box_height = detection.box
    cv2.rectangle(frame, (x, y), (x + box_width, y + box_height), (35, 190, 70), 2)
    cv2.putText(frame, f"person {detection.score:.2f}", (x, max(24, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (35, 190, 70), 2)
    points = detection.keypoints
    for first, second in POSE_EDGES:
        if points[first, 2] >= 0.25 and points[second, 2] >= 0.25:
            start = (int(points[first, 0] * width), int(points[first, 1] * height))
            end = (int(points[second, 0] * width), int(points[second, 1] * height))
            cv2.line(frame, start, end, (255, 170, 0), 2)
    for point in points:
        if point[2] >= 0.25:
            cv2.circle(frame, (int(point[0] * width), int(point[1] * height)), 4, (0, 255, 0), -1)
    return frame
