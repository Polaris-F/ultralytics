import warnings

from ultralytics.nn.tasks import DetectionModel
warnings.filterwarnings('ignore')
import torch
from ultralytics import YOLO
# use argparse set yaml file path
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='ultralytics/cfg/models/v8/yolov8n.yaml', help='path to yaml file')
args = parser.parse_args()

if __name__ == '__main__':
    # choose your yaml file
    model_yaml = args.model
    model = YOLO(model_yaml)
   
    model.info(detailed=False,imgsz=1024)

    model.fuse(detailed=False, imgsz=1024)
    # model.fuse()

    det_model = DetectionModel(model_yaml, verbose=False, imgsz=1024)
    im = torch.randn(1, 3, 1024, 1024)  # requires min imgsz=64
    det_model.fuse(detailed=False, imgsz=1024)
    _ = det_model.predict(im, profile=True)