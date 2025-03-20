# update ultralytics settinngs
from ultralytics import settings

# update datasets_dir weights_dir and other all settings
settings.update({
    # 'datasets_dir': '/path/to/datasets', 
    'weights_dir': '/userhome/lhf/Github/WorkSpace/office_weights',
    'runs_dir': '/userhome/lhf/Github/WorkSpace/VisDrone/ultralytics/runs',
    'tensorboard': False,
    'wandb': False,
})

from ultralytics import YOLO, RTDETR


## =============>>>>>>>> 标准训练 <

if True: 
    # Load a model
    # model = YOLO("yolov8n.yaml")  # build a new model from YAML
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
    # model = YOLO("yolov8n.yaml").load("yolov8n.pt")  # build from YAML and transfer weights

    # Train the model
    model.train(
        data="VisDrone.yaml", 
        epochs=100, 
        imgsz=640,
        device="7",
        batch=32,
        name="yolov8n-640",
        )

## =============>>>>>>>> 标准训练yolov11 640<
if False:
    # import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    # Load a model
    model = YOLO("yolo11n.yaml")  # build a new model from YAML
    # Train the model
    model.train(
        data="VisDrone.yaml", 
        epochs=100, 
        imgsz=1024,
        device='0,1',
        batch=32,
        name="yolov11n-1024",
        exist_ok=True,
        )

## =============>>>>>>>> 标准训练yolov11 640<
if False:
    # import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    # Load a model
    model = RTDETR("yolov8l-rtdetr.yaml")  # build a new model from YAML
    # Train the model
    model.train(
        data="VisDrone.yaml", 
        epochs=150, 
        imgsz=1024,
        device='4,5,6,7',
        batch=16,
        name="yolov8l-rtdetr-1024",
        exist_ok=True,
        )
    
## =============>>>>>>>> /userhome/lhf/Github/WorkSpace/ultralytics_cfg/cfg/models/v8/yolov8-SOEP.yaml <
if False:
    # Load a model
    model = YOLO("yolov8l-RSCD.yaml")  # build a new model from YAML
    # Train the model
    model.train(
        data="VisDrone.yaml", 
        epochs=100, 
        imgsz=1024,
        device='0,1,2,3',
        batch=32,
        name="yolov8l-RSCD",
        exist_ok=True,
        # amp=False,
        # half=False
        )

if False:
    # Load a model
    model = YOLO("/userhome/lhf/Github/WorkSpace/ultralytics_cfg/cfg/models/HR_FPN_Detect_n.yaml")  # build a new model from YAML
    # Train the model
    model.train(
        data="VisDrone.yaml", 
        epochs=100, 
        imgsz=1024,
        device='0,1,2,3',
        batch=16,
        name="HR_FPN_Detect_s",
        exist_ok=True,
        # amp=False,
        # half=False
        )
