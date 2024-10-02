




"""Validate YOLO model predictions on COCO dataset using pycocotools."""
from ultralytics.models.yolo.detect import DetectionValidator

args = {
    "model": "/userhome/lhf/Github/WorkSpace/VisDrone/ultralytics/runs/detect/yolov8n-640/weights/best.pt", 
    "data": "/userhome/lhf/Github/WorkSpace/ultralytics_cfg/cfg/datasets/VisDrone.yaml", 
    "save_json": True, 
    "imgsz": 640
    }

validator = DetectionValidator(args=args)
# validator()
validator.is_coco = True
_ = validator.eval_json_polaris(validator.stats,anno_json_path=r'')
