
"""Validate YOLO model predictions on COCO dataset using pycocotools."""
from ultralytics.models.yolo.detect import DetectionValidator


anno_json_path=r'/userhome/lhf/datasets/VisDrone2019/VisDrone2019-DET-test-dev/data.json'
root_path=r'/userhome/lhf/Github/WorkSpace/VisDrone/ultralytics/runs/detect/yolov11n-640'

args = {
    "model": root_path + "/weights/best.pt", 
    "data": "/userhome/lhf/Github/WorkSpace/ultralytics_cfg/cfg/datasets/VisDrone.yaml", 
    "save_json": True, 
    "imgsz": 640,
    "mode": "test_COCO", #保存文件夹名称
    "batch": 64,
    }

## ==========> run ultralytics validation ===========
print("============================>>> Start validating YOLO model predictions on COCO dataset using pycocotools. <<<=")
validator = DetectionValidator(args=args)
validator()
validator.is_coco = True
_,pred_json = validator.eval_json_polaris(validator.stats,anno_json_path=anno_json_path)


## ==========> run tidecv validation =
print("==========>>> Start validating YOLO model predictions on COCO dataset using tidecv. <<<=")
from tidecv import TIDE, datasets
tide = TIDE()
tide.evaluate_range(datasets.COCO(anno_json_path), datasets.COCOResult(pred_json), mode=TIDE.BOX)
tide.summarize()
# tide.plot(out_dir='result')