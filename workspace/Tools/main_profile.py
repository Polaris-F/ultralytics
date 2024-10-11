import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
# use argparse set yaml file path
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--yaml_path', type=str, default='ultralytics/cfg/models/v8/yolov8n.yaml', help='path to yaml file')
args = parser.parse_args()

if __name__ == '__main__':
    # choose your yaml file
    model_yaml = args.yaml_path
    model = YOLO(model_yaml)
    model.info(detailed=True)
    try:
        model.profile(imgsz=[640, 640])
    except Exception as e:
        print(e)
        pass
    model.fuse()