"""YOLO detector wrapping ultralytics. Returns plain Detection dataclasses so the
rest of the pipeline never sees ultralytics types.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from wow_alert.events import Detection

logger = logging.getLogger(__name__)


def detect_device() -> tuple[str, str]:
    if torch.cuda.is_available():
        return "cuda", torch.cuda.get_device_name(0)
    return "cpu", "CPU"


class YoloDetector:
    def __init__(self, model_path: Path, confidence: float, imgsz: int):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"YOLO model file not found at {self.model_path}. "
                f"Set `model_path` in config/app.yaml (or pass --model PATH) "
                f"to a .pt or .engine file."
            )
        self.confidence = confidence
        self.imgsz = imgsz
        self._device, self._device_name = detect_device()
        logger.info("Inference device: %s (%s)", self._device, self._device_name)
        logger.info("Loading YOLO model from %s", self.model_path)
        try:
            self._model = YOLO(str(self.model_path))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load YOLO weights at {self.model_path}: {exc}. "
                f"Verify the file is a valid ultralytics weights export."
            ) from exc

    def set_confidence(self, value: float) -> None:
        self.confidence = value

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self._model.predict(
            frame,
            conf=self.confidence,
            device=self._device,
            imgsz=self.imgsz,
            verbose=False,
        )
        if not results:
            return []
        result = results[0]
        names = result.names
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        out: list[Detection] = []
        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy().astype(float)
        for (x1, y1, x2, y2), cls_id, conf in zip(xyxy, cls_ids, confs):
            out.append(
                Detection(
                    class_name=names.get(int(cls_id), str(int(cls_id))),
                    confidence=float(conf),
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                )
            )
        return out
