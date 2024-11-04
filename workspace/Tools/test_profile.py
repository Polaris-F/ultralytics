# def test_model_profile():
"""Test profiling of the YOLO model with `profile=True` to assess performance and resource usage."""
import torch
from ultralytics.nn.tasks import DetectionModel

cfg_path = r''

model = DetectionModel(cfg=cfg_path)  # build model
im = torch.randn(1, 3, 640, 640)  # requires min imgsz=64
_ = model.predict(im, profile=True)